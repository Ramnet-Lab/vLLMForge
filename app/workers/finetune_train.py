"""Unsloth SFT/QLoRA trainer — runs inside the finetune image, never on the host.

Everything the run needs arrives in one JSON file (argv[1]) that the dashboard
writes next to the outputs, so the router never assembles a shell command and a
run can be reproduced by hand with `docker run ... python /worker/finetune_train.py
/out/config.json`.

Two lines are contracts with `app.finetune`'s log parser: `@@PROGRESS@@ {json}`
once per trainer log event, and a single closing `@@RESULT@@ {json}`. tqdm's own
bars still go to stdout for the log pane, but nothing structured is scraped from
them — they are carriage-return redraws and interleave badly.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# isort: off
# Unsloth patches trl/transformers/peft at import time and warns then silently
# deoptimises if it loses the race, so it goes first and stays first.
import unsloth  # noqa: F401
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
# isort: on

PROGRESS = "@@PROGRESS@@"
RESULT = "@@RESULT@@"

SHAREGPT_KEYS = ("from", "value")
MESSAGE_COLUMNS = ("messages", "conversations", "conversation", "chat")


def emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload)}", flush=True)


def phase(name: str) -> None:
    print(f"[finetune] {name}", flush=True)
    emit(PROGRESS, {"phase": name})


def load_model(cfg: dict[str, Any]):
    dtype = cfg.get("dtype") or "auto"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
        max_seq_length=int(cfg["max_seq_length"]),
        dtype=None if dtype == "auto" else dtype,
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        full_finetuning=bool(cfg.get("full_finetuning", False)),
        token=os.environ.get("HF_TOKEN") or None,
    )
    if cfg.get("full_finetuning"):
        return model, tokenizer
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["r"]),
        target_modules=list(cfg["target_modules"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias=cfg.get("bias", "none"),
        use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=int(cfg["seed"]),
        use_rslora=bool(cfg.get("use_rslora", False)),
        max_seq_length=int(cfg["max_seq_length"]),
    )
    return model, tokenizer


def _standardize(dataset):
    """ShareGPT from/value rows -> role/content, under whichever name unsloth uses."""
    from unsloth import chat_templates

    standardize = getattr(chat_templates, "standardize_data_formats", None)
    if standardize is None:
        standardize = chat_templates.standardize_sharegpt
    return standardize(dataset)


def build_dataset(cfg: dict[str, Any], tokenizer):
    spec = cfg["dataset"]
    split = spec.get("split") or "train"
    if spec["source"] == "local":
        dataset = load_dataset("json", data_files=spec["reference"], split=split)
    else:
        dataset = load_dataset(spec["reference"], spec.get("config") or None, split=split)

    limit = int(spec.get("max_rows") or 0)
    if limit and limit < len(dataset):
        dataset = dataset.select(range(limit))

    columns = list(dataset.column_names)
    text_field = spec.get("text_field") or "text"
    if text_field in columns:
        phase(f"dataset ready: {len(dataset)} rows, plain text column '{text_field}'")
        return dataset, text_field

    messages_field = spec.get("messages_field") or next(
        (c for c in MESSAGE_COLUMNS if c in columns), ""
    )
    if messages_field not in columns:
        raise SystemExit(
            f"dataset has no '{text_field}' column and no conversation column; "
            f"available columns: {', '.join(columns)}"
        )

    first = dataset[0][messages_field]
    if first and isinstance(first[0], dict) and all(k in first[0] for k in SHAREGPT_KEYS):
        if messages_field != "conversations":
            dataset = dataset.rename_column(messages_field, "conversations")
        dataset = _standardize(dataset)
        messages_field = "conversations"
        columns = list(dataset.column_names)

    if cfg.get("chat_template"):
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(
            "the dataset is conversational but the tokenizer has no chat template; "
            "set chat_template in the config"
        )

    def render(batch: dict[str, list]) -> dict[str, list]:
        return {
            "text": [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in batch[messages_field]
            ]
        }

    dataset = dataset.map(render, batched=True, remove_columns=columns)
    phase(f"dataset ready: {len(dataset)} rows, chat template applied to '{messages_field}'")
    return dataset, "text"


class ProgressCallback(TrainerCallback):
    """One structured line per trainer log event; elapsed lets the host compute an ETA."""

    def __init__(self) -> None:
        self.started = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        self.started = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        emit(
            PROGRESS,
            {
                "step": state.global_step,
                "total_steps": state.max_steps,
                "elapsed": round(time.monotonic() - self.started, 2),
                **(logs or {}),
            },
        )


def train(cfg: dict[str, Any], model, tokenizer, dataset, text_field: str, out: Path):
    max_steps = int(cfg.get("max_steps") or 0)
    args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        warmup_steps=int(cfg["warmup_steps"]),
        max_steps=max_steps if max_steps > 0 else -1,
        num_train_epochs=float(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        optim=cfg["optim"],
        weight_decay=float(cfg["weight_decay"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=int(cfg["seed"]),
        logging_steps=int(cfg["logging_steps"]),
        max_length=int(cfg["max_seq_length"]),
        dataset_text_field=text_field,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        callbacks=[ProgressCallback()],
    )
    return trainer.train()


def save(cfg: dict[str, Any], model, tokenizer, out: Path) -> dict[str, Any]:
    """Write the artefacts the export option asks for.

    adapter and merged_16bit have both been run on this box. merged_4bit and
    gguf are wired from Unsloth's documented API but are NOT verified here, and
    gguf additionally needs llama.cpp, which the NGC base image does not ship —
    Unsloth would have to fetch and build it at save time on aarch64.
    """
    artefacts: dict[str, Any] = {}
    export = cfg.get("export", "adapter")

    if cfg.get("full_finetuning"):
        target = out / "model"
        model.save_pretrained(str(target))
        tokenizer.save_pretrained(str(target))
        return {"model_dir": str(target), "export": "full"}

    adapter = out / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    artefacts["adapter_dir"] = str(adapter)
    config_path = adapter / "adapter_config.json"
    if config_path.exists():
        adapter_config = json.loads(config_path.read_text())
        # The base recorded here is the bnb-4bit repo unsloth actually loaded;
        # vLLM cannot serve that, which is the whole reason the host surfaces it.
        artefacts["adapter_base_model"] = adapter_config.get("base_model_name_or_path", "")
        artefacts["lora_rank"] = adapter_config.get("r")

    if export in ("merged_16bit", "merged_4bit"):
        phase(f"exporting {export}")
        merged = out / export
        model.save_pretrained_merged(str(merged), tokenizer, save_method=export)
        artefacts["merged_dir"] = str(merged)
    elif export == "gguf":
        phase("exporting gguf")
        gguf = out / "gguf"
        model.save_pretrained_gguf(
            str(gguf), tokenizer, quantization_method=cfg.get("gguf_quant", "q4_k_m")
        )
        artefacts["gguf_dir"] = str(gguf)
        artefacts["gguf_files"] = sorted(p.name for p in gguf.glob("*.gguf"))

    artefacts["export"] = export
    return artefacts


def chown_tree(path: Path, uid: int, gid: int) -> None:
    """Containers run as root here, so hand the artefacts back to the dashboard user."""
    if uid < 0 or gid < 0:
        return
    for root, dirs, files in os.walk(path):
        for name in (*dirs, *files):
            with contextlib.suppress(OSError):
                os.chown(os.path.join(root, name), uid, gid)
    with contextlib.suppress(OSError):
        os.chown(path, uid, gid)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: finetune_train.py <config.json>", file=sys.stderr)
        return 2
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    try:
        phase(f"loading {cfg['model']}")
        model, tokenizer = load_model(cfg)
        phase("preparing dataset")
        dataset, text_field = build_dataset(cfg, tokenizer)
        phase("training")
        result = train(cfg, model, tokenizer, dataset, text_field, out)
        phase("saving")
        artefacts = save(cfg, model, tokenizer, out)
        emit(
            RESULT,
            {
                "model": cfg["model"],
                "metrics": {k: v for k, v in (result.metrics or {}).items()},
                **artefacts,
            },
        )
    finally:
        chown_tree(out, int(cfg.get("uid", -1)), int(cfg.get("gid", -1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
