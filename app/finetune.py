"""Unsloth fine-tuning: run configuration, job assembly and progress parsing.

The container side is `app/workers/finetune_train.py`, mounted read-only at
/worker. Everything the run needs is written to config.json in the run's output
directory, so this module never builds a shell command and a failed run can be
replayed by hand from the artefacts it left behind.

Two GB10 facts shape the design. Containers here run as root, so the worker
chowns its outputs back to the dashboard user on the way out — hence uid/gid in
the config. And GPU memory *is* host memory, so a run is size-checked against
free host memory before it is allowed to start, the same way a vLLM launch is
checked against the utilisation budget.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app import db, docker_ctl, events, jobs, safety, servers
from app.config import settings

log = logging.getLogger("llmd.finetune")

GIB = 1024 ** 3
KIND = "finetune"
PROGRESS_TAG = "@@PROGRESS@@"
RESULT_TAG = "@@RESULT@@"

WORKER_DIR = Path(__file__).resolve().parent / "workers"
DOCKERFILE = "finetune.Dockerfile"

# vLLM only accepts these ranks, so an adapter trained outside the set can never
# be served without merging. Keep the form honest about that.
SERVABLE_RANKS = (8, 16, 32, 64, 128, 256)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

OPTIMIZERS = ("adamw_8bit", "adamw_torch", "adamw_torch_fused", "paged_adamw_8bit", "sgd")
SCHEDULERS = ("linear", "cosine", "constant", "constant_with_warmup", "cosine_with_restarts")
GGUF_QUANTS = ("f16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k")
CHAT_TEMPLATES = ("", "qwen3-instruct", "qwen25", "llama3", "llama31", "gemma3", "phi4",
                  "chatml", "mistral", "zephyr", "gptoss")

# Rows we know how to feed the trainer, in the order the detector prefers them.
DATASET_FORMATS = ("text", "messages", "prompt-completion", "alpaca", "unknown")

PREVIEW_ROWS = 3
PREVIEW_CHARS = 600


# --- configuration -------------------------------------------------------

class DatasetSpec(BaseModel):
    source: Literal["hub", "local"] = Field(
        "hub", description="'hub' loads a HuggingFace dataset id; 'local' a JSONL file."
    )
    reference: str = Field(
        "", description="Hub dataset id, or the file name of an uploaded JSONL in the dataset dir."
    )
    split: str = Field("train", description="Split to train on, e.g. train or train[:2000].")
    config: str = Field(
        "", description="Hub dataset config/subset name, when the dataset defines several."
    )
    text_field: str = Field(
        "text", description="Column holding ready-to-train text. Used as-is when present."
    )
    messages_field: str = Field(
        "",
        description=(
            "Column holding chat turns. Blank auto-detects messages/conversations; the "
            "tokenizer's chat template is applied to build the text column."
        ),
    )
    max_rows: int = Field(
        0, ge=0, description="Train on at most this many rows. 0 uses the whole split."
    )


class FinetuneConfig(BaseModel):
    name: str = Field("", description="Run name. Used for the output directory and the job title.")
    model: str = Field(
        "unsloth/Qwen3-0.6B",
        description=(
            "Base model repo. Unsloth's -unsloth-bnb-4bit variants download fastest. "
            "With both vLLM servers up, 8B is about the ceiling for 4-bit on this box."
        ),
    )
    max_seq_length: int = Field(
        2048, ge=128, le=131072, description="Sequence length. Activation memory scales with it."
    )
    dtype: Literal["auto", "bfloat16", "float16"] = Field(
        "auto", description="Compute dtype. 'auto' lets Unsloth pick bfloat16 on this GPU."
    )
    load_in_4bit: bool = Field(
        True, description="4-bit QLoRA base weights. Turning this off roughly doubles the memory."
    )
    full_finetuning: bool = Field(
        False,
        description=(
            "Train every weight instead of a LoRA. Needs several times the memory and "
            "produces a full model rather than an adapter."
        ),
    )

    r: int = Field(16, ge=1, le=512, description="LoRA rank. Stick to 8/16/32/64/128/256 — vLLM "
                                                "refuses to serve other ranks.")
    target_modules: list[str] = Field(
        default_factory=lambda: list(TARGET_MODULES),
        description="Projections to adapt. The default set covers attention and MLP.",
    )
    lora_alpha: int = Field(16, ge=1, le=512, description="LoRA scaling. Commonly equal to r.")
    lora_dropout: float = Field(
        0.0, ge=0.0, le=0.9, description="Dropout on the adapter. 0 is Unsloth's fast path."
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        "none", description="Which biases to train. 'none' is the fast path."
    )
    use_gradient_checkpointing: Literal["unsloth", "true", "false"] = Field(
        "unsloth",
        description="'unsloth' trades a little speed for a lot of memory on long sequences.",
    )
    use_rslora: bool = Field(
        False, description="Rank-stabilised LoRA: scales alpha by sqrt(r). Helps at high rank."
    )
    chat_template: Literal[CHAT_TEMPLATES] = Field(  # type: ignore[valid-type]
        "", description="Force a chat template for conversational rows. Blank uses the "
                        "tokenizer's own."
    )

    per_device_train_batch_size: int = Field(
        2, ge=1, le=64, description="Rows per step. The main lever on activation memory."
    )
    gradient_accumulation_steps: int = Field(
        4, ge=1, le=128, description="Steps accumulated before an update. Raises the effective "
                                     "batch without extra memory."
    )
    warmup_steps: int = Field(5, ge=0, le=1000, description="Linear warmup before the schedule.")
    max_steps: int = Field(
        60, ge=0, le=200000,
        description="Hard step cap. 0 means train for num_train_epochs instead.",
    )
    num_train_epochs: float = Field(
        1.0, gt=0.0, le=100.0, description="Epochs, used only when max_steps is 0."
    )
    learning_rate: float = Field(
        2e-4, gt=0.0, le=1.0, description="2e-4 is the usual LoRA starting point; 2e-5 for full "
                                          "fine-tuning."
    )
    optim: Literal[OPTIMIZERS] = Field(  # type: ignore[valid-type]
        "adamw_8bit", description="adamw_8bit keeps optimiser state small via bitsandbytes."
    )
    weight_decay: float = Field(0.01, ge=0.0, le=1.0, description="L2 regularisation.")
    lr_scheduler_type: Literal[SCHEDULERS] = Field(  # type: ignore[valid-type]
        "linear", description="Learning-rate decay shape."
    )
    seed: int = Field(3407, description="Seed for shuffling and LoRA init.")
    logging_steps: int = Field(
        1, ge=1, le=1000, description="How often the trainer logs — this is what drives the "
                                      "progress chart."
    )

    export: Literal["adapter", "merged_16bit", "merged_4bit", "gguf"] = Field(
        "adapter",
        description=(
            "adapter: LoRA only, needs an fp16 base to serve. merged_16bit: a standalone model "
            "vLLM can serve directly. merged_4bit and gguf are wired but unverified on this box."
        ),
    )
    gguf_quant: Literal[GGUF_QUANTS] = Field(  # type: ignore[valid-type]
        "q4_k_m", description="Quantisation for the GGUF export."
    )
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)


FORM_GROUPS = [
    {"id": "run", "title": "Run", "fields": ["name", "model", "dataset"]},
    {
        "id": "model",
        "title": "Model & loading",
        "blurb": "4-bit is what makes this fit next to the running vLLM servers.",
        "fields": ["max_seq_length", "dtype", "load_in_4bit", "full_finetuning", "chat_template"],
    },
    {
        "id": "lora",
        "title": "LoRA",
        "blurb": "Rank decides both adapter size and whether vLLM will load it.",
        "fields": ["r", "lora_alpha", "lora_dropout", "target_modules", "bias",
                   "use_rslora", "use_gradient_checkpointing"],
    },
    {
        "id": "training",
        "title": "Training",
        "blurb": "max_steps wins over num_train_epochs whenever it is non-zero.",
        "fields": ["per_device_train_batch_size", "gradient_accumulation_steps", "warmup_steps",
                   "max_steps", "num_train_epochs", "learning_rate", "optim", "weight_decay",
                   "lr_scheduler_type", "seed", "logging_steps"],
    },
    {
        "id": "export",
        "title": "Export",
        "blurb": "An adapter alone cannot be served: its recorded base is a bnb-4bit repo.",
        "fields": ["export", "gguf_quant"],
    },
]


def defaults() -> dict[str, Any]:
    """Field defaults without running validation, so a half-filled form still renders."""
    out: dict[str, Any] = {}
    for name, field in FinetuneConfig.model_fields.items():
        value = field.get_default(call_default_factory=True)
        out[name] = value.model_dump() if isinstance(value, BaseModel) else value
    return out


def ui_schema() -> dict[str, Any]:
    return {
        "schema": FinetuneConfig.model_json_schema(),
        "defaults": defaults(),
        "groups": FORM_GROUPS,
        "choices": {
            "optim": list(OPTIMIZERS),
            "lr_scheduler_type": list(SCHEDULERS),
            "gguf_quant": list(GGUF_QUANTS),
            "chat_template": list(CHAT_TEMPLATES),
            "target_modules": list(TARGET_MODULES),
            "r": list(SERVABLE_RANKS),
            "export": ["adapter", "merged_16bit", "merged_4bit", "gguf"],
        },
        "image": settings.finetune_image,
        "dataset_dir": str(settings.dataset_dir),
        "notes": [
            "A saved adapter records the bnb-4bit repo it was trained on as its base model. "
            "vLLM cannot serve that, so serving an adapter means pointing it at the fp16 base "
            "with --enable-lora, or exporting merged_16bit instead.",
            "The adapter and merged_16bit exports have been run end to end on this GB10. "
            "merged_4bit and gguf come straight from Unsloth's API and are unverified here; "
            "gguf additionally needs llama.cpp, which the image does not ship.",
        ],
    }


# --- job assembly --------------------------------------------------------

def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower() or "run"


def dataset_path(reference: str) -> Path:
    """Resolve an uploaded dataset reference to a host path inside the dataset dir."""
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = settings.dataset_dir / reference
    candidate = candidate.resolve()
    root = settings.dataset_dir.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"local datasets must live under {root}")
    return candidate


def build_job(config: FinetuneConfig) -> jobs.JobSpec:
    """Write the run directory and describe the container that will consume it."""
    if not config.dataset.reference.strip():
        raise ValueError("a dataset is required: give a hub id or upload a JSONL file")
    if config.full_finetuning and config.export != "adapter":
        raise ValueError("full fine-tuning produces a whole model; leave export as 'adapter'")

    run_id = uuid.uuid4().hex[:12]
    run_dir = jobs.output_dir(run_id, _slug(config.name or config.model.split("/")[-1]))

    payload = config.model_dump()
    payload["output_dir"] = "/out"
    # Container writes land as root otherwise, and the dashboard could then
    # neither read the adapter nor delete the run.
    payload["uid"], payload["gid"] = os.getuid(), os.getgid()
    if config.dataset.source == "local":
        host_path = dataset_path(config.dataset.reference)
        if not host_path.exists():
            raise ValueError(f"no such dataset file: {host_path}")
        payload["dataset"]["reference"] = f"/datasets/{host_path.name}"
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2))

    env = {"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1"}
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token

    return jobs.JobSpec(
        kind=KIND,
        title=f"finetune {config.model}",
        image=settings.finetune_image,
        command=["/worker/finetune_train.py", "/out/config.json"],
        env=env,
        mounts=[
            docker_ctl.Mount(settings.hf_cache, "/hf"),
            docker_ctl.Mount(settings.dataset_dir, "/datasets", read_only=True),
            docker_ctl.Mount(run_dir, "/out"),
            docker_ctl.Mount(WORKER_DIR, "/worker", read_only=True),
        ],
        gpu=True,
        entrypoint="python",
        workdir="/work",
        # Training needs the hub but binds no port; host networking here would
        # only put it in the way of the vLLM servers.
        network="bridge",
        meta={
            "run_id": run_id,
            "run_dir": str(run_dir),
            "model": config.model,
            "export": config.export,
            "dataset": config.dataset.model_dump(),
            "config": config.model_dump(),
        },
    )


# --- progress parsing ----------------------------------------------------

@jobs.register_parser(KIND)
def parse_progress(line: str, progress: dict) -> dict | None:
    """Turn the worker's tagged JSON lines into the dict the fine-tune view renders."""
    if line.startswith(RESULT_TAG):
        payload = _payload(line, RESULT_TAG)
        # RESULT_KEY moves this onto the job row rather than leaving the
        # finished artefact buried inside a progress reading.
        return {jobs.RESULT_KEY: payload, "percent": 100.0} if payload else None
    if not line.startswith(PROGRESS_TAG):
        return None
    payload = _payload(line, PROGRESS_TAG)
    if not payload:
        return None

    update: dict[str, Any] = {}
    if "phase" in payload:
        update["phase"] = payload["phase"]
    for key in ("learning_rate", "epoch", "grad_norm", "train_runtime", "train_loss"):
        if key in payload:
            update[key] = payload[key]

    step = int(payload.get("step") or 0)
    total = int(payload.get("total_steps") or progress.get("total_steps") or 0)
    if step:
        update["step"] = step
    if total:
        update["total_steps"] = total
    if step and total:
        update["percent"] = round(100.0 * step / total, 1)
        elapsed = float(payload.get("elapsed") or 0.0)
        if elapsed > 0:
            update["eta"] = round(elapsed / step * max(0, total - step), 1)

    if "loss" in payload:
        loss = float(payload["loss"])
        update["loss"] = loss
        # The sparkline wants the whole curve, and nothing else keeps it: the
        # log file is the only other copy and re-parsing it per poll is silly.
        history = list(progress.get("loss_history") or [])
        history.append([step, round(loss, 5)])
        update["loss_history"] = history[-500:]
    return update


def _payload(line: str, tag: str) -> dict | None:
    try:
        value = json.loads(line[len(tag):].strip())
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


# --- memory pre-flight ---------------------------------------------------

_PARAM_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])")


def parse_param_count(model: str) -> float | None:
    """Billions of parameters, read off the repo name — the only clue we have offline."""
    matches = _PARAM_SIZE.findall(model.replace("_", "-"))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def estimate_bytes(config: FinetuneConfig) -> tuple[int, float | None, str]:
    """Rough peak host memory for a run: weights, then activations, then runtime."""
    params_b = parse_param_count(config.model)
    if params_b is None:
        return 0, None, "unknown model size"

    if config.full_finetuning:
        # weights + grads + 8-bit optimiser state, all resident at once.
        weights = params_b * 8.0
        basis = "full fine-tuning at ~8 GiB per billion parameters"
    elif config.load_in_4bit:
        weights = params_b * 1.0
        basis = "4-bit QLoRA at ~1 GiB per billion parameters"
    else:
        weights = params_b * 2.0
        basis = "16-bit LoRA at ~2 GiB per billion parameters"

    tokens = config.per_device_train_batch_size * (config.max_seq_length / 2048)
    activations = 2.0 + 0.5 * tokens * max(1.0, params_b ** 0.5)
    if config.use_gradient_checkpointing != "false":
        activations *= 0.5
    runtime = 3.0
    return int((weights + activations + runtime) * GIB), params_b, basis


async def check_memory(config: FinetuneConfig) -> dict:
    """Verdict-shaped pre-flight. Fine-tuning takes no util fraction, so this
    compares an estimate against what the host actually has free right now."""
    budget = await safety.current_budget()
    payload = budget.as_dict()
    need, params_b, basis = estimate_bytes(config)
    tenants = ", ".join(f"{t.name}={t.util:g}" for t in budget.tenants) or "none"
    util = need / budget.total_bytes if budget.total_bytes else 0.0

    if params_b is None:
        return safety.Verdict(
            ok=True,
            level="warn",
            message=(
                f"Cannot tell how big '{config.model}' is from its name, so this run is "
                f"unchecked. {_gib(budget.available_bytes)} is free right now "
                f"(vLLM tenants: {tenants}). Watch the first minute of the log."
            ),
            budget=payload,
        ).as_dict()

    if need >= budget.available_bytes:
        return safety.Verdict(
            ok=False,
            level="block",
            message=(
                f"Refusing to start: {params_b:g}B {basis} plus activations for "
                f"{config.per_device_train_batch_size}x{config.max_seq_length} tokens needs about "
                f"{_gib(need)}, and only {_gib(budget.available_bytes)} is free. "
                f"Stop a vLLM server ({tenants}) or try {_cheaper(config)}."
            ),
            budget=payload,
            requested_bytes=need,
            requested_util=util,
        ).as_dict()

    if need > 0.75 * budget.available_bytes or budget.tenants:
        return safety.Verdict(
            ok=True,
            level="warn",
            message=(
                f"Tight: about {_gib(need)} needed ({params_b:g}B, {basis}) against "
                f"{_gib(budget.available_bytes)} free, with {tenants} also resident. "
                "A concurrent inference spike is what would OOM this — consider stopping "
                "the servers for the duration."
            ),
            budget=payload,
            requested_bytes=need,
            requested_util=util,
        ).as_dict()

    return safety.Verdict(
        ok=True,
        level="ok",
        message=(
            f"Fits: about {_gib(need)} needed ({params_b:g}B, {basis}), "
            f"{_gib(budget.available_bytes)} free."
        ),
        budget=payload,
        requested_bytes=need,
        requested_util=util,
    ).as_dict()


def _cheaper(config: FinetuneConfig) -> str:
    """The levers that are actually still available on this config."""
    options = []
    if config.full_finetuning:
        options.append("a LoRA instead of full fine-tuning")
    if not config.load_in_4bit:
        options.append("load_in_4bit")
    if config.per_device_train_batch_size > 1:
        options.append("batch size 1")
    if config.max_seq_length > 1024:
        options.append("a shorter max_seq_length")
    options.append("a smaller model")
    return ", ".join(options[:-1]) + f" or {options[-1]}" if len(options) > 1 else options[0]


def _gib(value: float) -> str:
    return f"{value / GIB:.1f} GiB"


# --- image ---------------------------------------------------------------

PROBE = (
    "import json, unsloth, torch, trl, peft, transformers, bitsandbytes;"
    "print('@@PROBE@@ ' + json.dumps({'unsloth': unsloth.__version__,"
    " 'trl': trl.__version__, 'peft': peft.__version__,"
    " 'transformers': transformers.__version__, 'bitsandbytes': bitsandbytes.__version__,"
    " 'torch': torch.__version__,"
    " 'capability': list(torch.cuda.get_device_capability())}))"
)


async def probe_image() -> dict:
    """Does unsloth actually import in the built image? Cached, because it costs ~40s."""
    code, out, err = await docker_ctl.run_capture(
        image=settings.finetune_image,
        command=["-c", PROBE],
        entrypoint="python",
        gpu=True,
        network="bridge",
        mounts=[docker_ctl.Mount(settings.hf_cache, "/hf")],
        timeout=300.0,
    )
    for line in out.splitlines():
        if line.startswith("@@PROBE@@"):
            payload = _payload(line, "@@PROBE@@")
            if payload:
                return {"ok": True, "checked_at": db.now(), **payload}
    tail = (err or out).strip().splitlines()[-6:]
    return {"ok": False, "checked_at": db.now(), "error": "\n".join(tail) or f"exit {code}"}


async def image_present() -> bool:
    return await docker_ctl.image_exists(settings.finetune_image)


async def status(*, probe: bool = False) -> dict:
    present = await image_present()
    imports = db.get_setting("finetune_probe", None)
    if probe and present:
        imports = await probe_image()
        db.set_setting("finetune_probe", imports)
    return {
        "image": settings.finetune_image,
        "image_present": present,
        "dockerfile": str(settings.docker_dir / DOCKERFILE),
        "base_image": settings.vllm_image,
        "unsloth": imports,
        "defaults": defaults(),
        "dataset_dir": str(settings.dataset_dir),
        "output_dir": str(settings.output_dir),
    }


_BUILD_STEP = re.compile(r"^#\d+\s+\[(\d+)/(\d+)\]")

# The job manager only tracks tasks that tail containers, so these are held here
# to keep them from being garbage-collected mid-build.


async def build_image_job(build_args: dict[str, str] | None = None) -> str:
    """Build the finetune image as a normal job row.

    The job runner tails *containers*, and `docker build` is a host-side process
    that produces one, so this drives the same row directly instead: the UI gets
    a build with logs, progress and history like every other job.
    """
    dockerfile = settings.docker_dir / DOCKERFILE
    if not dockerfile.exists():
        raise FileNotFoundError(dockerfile)
    tag = settings.finetune_image
    argv = ["docker", "build", "-t", tag, "-f", str(dockerfile), str(settings.docker_dir)]
    spec = jobs.JobSpec(
        kind="build",
        title=f"build {tag}",
        image=tag,
        command=argv,
        env={},
        mounts=(),
        gpu=False,
        meta={
            "role": KIND,
            "tag": tag,
            "dockerfile": str(dockerfile),
            "build_args": build_args or {},
        },
    )
    job_id = jobs.manager.create(spec)
    jobs.manager.adopt(job_id, _run_build(job_id, tag, dockerfile, build_args))
    return job_id


async def _run_build(
    job_id: str, tag: str, dockerfile: Path, build_args: dict[str, str] | None
) -> None:
    manager = jobs.manager
    manager.begin(job_id)
    manager.log(job_id, f"$ docker build -t {tag} -f {dockerfile} {settings.docker_dir}")
    progress: dict[str, Any] = {"phase": "building"}
    try:
        async for line in docker_ctl.build_image(
            tag, dockerfile, settings.docker_dir, build_args=build_args
        ):
            manager.log(job_id, line)
            match = _BUILD_STEP.match(line)
            if match:
                done, total = int(match.group(1)), int(match.group(2))
                progress = {"phase": line[:120], "step": done, "total_steps": total,
                            "percent": round(100.0 * done / total, 1)}
                manager.report(job_id, progress)
    except Exception as exc:
        log.exception("finetune image build failed")
        manager.log(job_id, f"[dashboard] build failed: {exc}")
        manager.finish(job_id, ok=False, error=str(exc)[:400])
        return
    manager.finish(
        job_id,
        ok=True,
        result={"tag": tag},
        progress={**progress, "phase": "built", "percent": 100.0},
    )
    await events.broker.publish(events.JOBS, {"type": "image", "image": tag})


# --- datasets ------------------------------------------------------------

def detect_format(row: dict) -> str:
    if "text" in row:
        return "text"
    if any(key in row for key in ("messages", "conversations", "conversation", "chat")):
        return "messages"
    if "prompt" in row and "completion" in row:
        return "prompt-completion"
    if "instruction" in row and "output" in row:
        return "alpaca"
    return "unknown"


def validate_jsonl(raw: bytes) -> dict[str, Any]:
    """Parse an upload before it is stored: a broken row must fail here, not mid-run."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"not UTF-8 text: {exc}") from None

    rows = 0
    preview: list[dict] = []
    formats: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"line {number} is not valid JSON: {exc}") from None
        if not isinstance(row, dict):
            raise ValueError(f"line {number} is a {type(row).__name__}, expected a JSON object")
        rows += 1
        formats.add(detect_format(row))
        if len(preview) < PREVIEW_ROWS:
            preview.append({k: _truncate(v) for k, v in row.items()})
    if not rows:
        raise ValueError("the file has no rows")
    fmt = next((f for f in DATASET_FORMATS if f in formats), "unknown")
    if fmt == "unknown":
        raise ValueError(
            "rows have none of the recognised shapes (text, messages/conversations, "
            "prompt+completion, instruction+output)"
        )
    return {"rows": rows, "format": fmt, "preview": preview, "size_bytes": len(raw)}


def _truncate(value: Any) -> Any:
    rendered = value if isinstance(value, str) else json.dumps(value, default=str)
    return rendered[:PREVIEW_CHARS] + ("…" if len(rendered) > PREVIEW_CHARS else "")


def store_upload(name: str, raw: bytes, *, replace: bool = False) -> dict:
    """Validate, write into the dataset dir, and record the row the UI lists."""
    info = validate_jsonl(raw)
    filename = f"{_slug(name)}.jsonl"
    path = settings.dataset_dir / filename
    if path.exists() and not replace:
        raise ValueError(f"{filename} already exists in {settings.dataset_dir}")
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    db.execute(
        "INSERT INTO datasets (name, source, reference, format, rows, size_bytes, preview,"
        " created_at) VALUES (?, 'upload', ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(name) DO UPDATE SET reference = excluded.reference,"
        " format = excluded.format, rows = excluded.rows, size_bytes = excluded.size_bytes,"
        " preview = excluded.preview, created_at = excluded.created_at",
        (
            filename,
            str(path),
            info["format"],
            info["rows"],
            info["size_bytes"],
            db.dumps(info["preview"]),
            db.now(),
        ),
    )
    return {**info, "name": filename, "path": str(path), "reference": filename}


def list_datasets() -> dict:
    """Recorded datasets, plus whatever else is lying in the dataset dir."""
    rows = [
        {**row, "preview": db.loads(row["preview"], [])}
        for row in db.query("SELECT * FROM datasets ORDER BY created_at DESC")
    ]
    known = {Path(row["reference"]).name for row in rows}
    files = []
    if settings.dataset_dir.exists():
        for path in sorted(settings.dataset_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (".jsonl", ".json", ".csv"):
                continue
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "registered": path.name in known,
                }
            )
    return {"datasets": rows, "files": files, "dataset_dir": str(settings.dataset_dir)}


# --- serving a finished run ----------------------------------------------

_QUANT_SUFFIX = re.compile(r"-(unsloth-)?(bnb-)?4bit$", re.IGNORECASE)


def fp16_base(repo: str) -> str:
    """The fp16 repo behind a bnb-4bit one — vLLM will not load the quantised repo."""
    return _QUANT_SUFFIX.sub("", repo or "")


def _base_repo(recorded: str, requested: str) -> str:
    """Prefer the repo id the user typed: unsloth records its 4-bit mirror
    lower-cased, and hub ids are case-sensitive."""
    stripped = fp16_base(recorded)
    if requested and stripped.lower() == fp16_base(requested).lower():
        return fp16_base(requested)
    return stripped or fp16_base(requested)


def _served_path(path: Path) -> str:
    """Where vLLM's container sees a path this job wrote on the host."""
    return servers.container_path(path)


def serve_plan(job: dict, *, name: str) -> dict:
    """What a server definition for this finished run should look like."""
    meta = ((job.get("spec") or {}).get("meta")) or {}
    result = job.get("result") or (job.get("progress") or {}).get("result") or {}
    run_dir = Path(meta.get("run_dir") or "").resolve()
    if not run_dir.exists():
        raise ValueError("this run left no output directory")

    config = meta.get("config") or {}
    export = meta.get("export") or result.get("export") or "adapter"
    merged = run_dir / export if export.startswith("merged") else None
    if merged is not None and merged.exists():
        return {
            "model": _served_path(merged),
            "args": {},
            "adapter": None,
            "note": (
                f"Serving the {export} export directly as a normal model from "
                f"{merged}. No LoRA flags needed."
            ),
        }

    adapter = run_dir / "adapter"
    if not adapter.exists():
        raise ValueError(f"no adapter or merged export under {run_dir}")
    base = _base_repo(result.get("adapter_base_model") or "", config.get("model") or "")
    rank = int(result.get("lora_rank") or config.get("r") or 16)
    if rank not in SERVABLE_RANKS:
        raise ValueError(f"LoRA rank {rank} is not one of {SERVABLE_RANKS}; merge instead")
    return {
        "model": base,
        "args": {
            "enable_lora": True,
            "max_lora_rank": rank,
            "max_loras": 1,
            "lora_modules": [f"{name}={_served_path(adapter)}"],
        },
        "adapter": str(adapter),
        "note": (
            f"This run exported an adapter only. Its adapter_config.json records "
            f"'{result.get('adapter_base_model') or config.get('model')}' as the base, which is a "
            f"bnb-4bit repo that vLLM cannot load — so the server is defined against the fp16 "
            f"base '{base}' with --enable-lora and the adapter attached as '{name}'. If that "
            f"repo id is wrong, edit the server's model field, or re-run the fine-tune with "
            f"export=merged_16bit to get a standalone model."
        ),
    }
