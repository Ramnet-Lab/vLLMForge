"""The llama.cpp parameter model.

The sibling of `app/vllm_spec.py`, and deliberately its mirror image: the same
public surface (`schema`, `by_dest`, `ui_model`, `build_argv`, `validate`) over
the same document shape, so the frontend renders both engines with one renderer
and neither knows the other exists.

`app/data/llamacpp_args.json` carries the same thirteen keys per flag that
`vllm_args.json` does, plus three llama.cpp needs and argparse never did:

* `negative_flag` — vLLM's parser *derives* `--no-<flag>` from the flag name.
  llama.cpp spells its negations, and they are not derivable: the negation of
  `--webui` is `--no-webui`, of `--cont-batching` is `--no-cont-batching`, and
  of `--mmap` is `--no-mmap`. Guessing would emit flags the binary rejects.
* `accepts` — the literal words a numeric flag also takes. `-ngl` accepts `auto`
  and `all`; a validator that only knows integers would reject its own default.
* `env` — llama.cpp reads most flags from `LLAMA_ARG_*`, and the mapping is not
  a transform of the flag name (`--reasoning-format` reads `LLAMA_ARG_THINK`),
  so it is recorded rather than computed.

The file is authored rather than generated, because generating it needs the
image and the image needs a GPU-less machine to have pulled a multi-gigabyte
CUDA build. `tools/gen_llamacpp_schema.py` regenerates it from any real
`llama-server --help`, and the `image` key is what tells `scripts/setup.sh`
whether the checked-in copy still describes the image you configured.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import settings

# Ordered UI sections. Anything not listed here still appears under "All other
# parameters", grouped by the section llama.cpp's own --help puts it in, so no
# flag is ever unreachable.
FEATURED: list[dict[str, Any]] = [
    {
        "id": "offload",
        "title": "GPU offload",
        "blurb": (
            "The memory decision. llama.cpp holds the LAST N of the model's layers on "
            "the accelerator and runs the rest on the CPU, so -ngl trades speed for "
            "memory continuously rather than all-or-nothing. Left on 'auto' the engine "
            "measures the machine and fits itself; an explicit number turns that off "
            "and is attempted exactly."
        ),
        "flags": [
            "n_gpu_layers", "fit", "fit_target", "split_mode", "tensor_split",
            "main_gpu", "device", "cpu_moe", "n_cpu_moe", "load_mode",
        ],
    },
    {
        "id": "context",
        "title": "Context & KV cache",
        "blurb": (
            "The KV cache is linear in --ctx-size and in the model's layer and KV-head "
            "count, and it is allocated up front. Quantising it with -ctk/-ctv is the "
            "cheapest way to buy context back; -ub is what to lower when a launch runs "
            "out of memory while processing a prompt rather than while loading."
        ),
        "flags": [
            "ctx_size", "cache_type_k", "cache_type_v", "flash_attn", "ubatch_size",
            "batch_size", "parallel", "kv_unified", "swa_full", "cache_prompt",
            "cache_reuse", "context_shift",
        ],
    },
    {
        "id": "model",
        "title": "Model & loading",
        "blurb": "What to serve, and what it is called to a client.",
        "flags": ["alias", "n_predict", "keep", "threads", "threads_batch", "mmap", "mlock"],
    },
    {
        "id": "chat",
        "title": "Chat template & reasoning",
        "blurb": (
            "--jinja is on by default and is what makes tool calling work: without it "
            "llama.cpp falls back to a built-in template that cannot express tool calls."
        ),
        "flags": [
            "jinja", "chat_template", "chat_template_file", "chat_template_kwargs",
            "reasoning", "reasoning_format", "reasoning_effort", "reasoning_budget",
            "prefill_assistant",
        ],
    },
    {
        "id": "lora",
        "title": "LoRA adapters",
        "blurb": (
            "GGUF adapters only. The PEFT directory the Fine-tune tab writes has to be "
            "converted before llama.cpp can read it."
        ),
        "flags": ["lora", "lora_scaled", "lora_init_without_apply"],
    },
    {
        "id": "sampling",
        "title": "Sampling defaults",
        "blurb": (
            "What a client gets when it asks for nothing. Every one of these is also a "
            "per-request field, so these are defaults rather than limits."
        ),
        "flags": [
            "temp", "top_k", "top_p", "min_p", "repeat_penalty", "repeat_last_n",
            "presence_penalty", "frequency_penalty", "seed", "samplers",
            "grammar", "grammar_file", "json_schema",
        ],
    },
    {
        "id": "multimodal",
        "title": "Multimodal",
        "blurb": "Only meaningful for a model that ships a projector.",
        "flags": [
            "mmproj", "mmproj_url", "mmproj_offload", "mmproj_auto",
            "image_min_tokens", "image_max_tokens",
        ],
    },
    {
        "id": "speculative",
        "title": "Speculative decoding",
        "blurb": (
            "A small draft model proposes tokens the real one verifies in a single "
            "pass. Its weights come out of the same memory the main model is spending."
        ),
        "flags": [
            "spec_draft_model", "spec_draft_ngl", "spec_draft_n_max",
            "spec_draft_n_min", "spec_draft_p_min", "spec_type",
        ],
    },
    {
        "id": "frontend",
        "title": "HTTP frontend",
        "blurb": "Host, port and metrics are managed for you; these cover the rest.",
        "flags": [
            "api_key", "api_key_file", "embedding", "pooling", "reranking",
            "slots", "props", "webui", "timeout", "threads_http",
            "sse_ping_interval", "ssl_cert_file", "ssl_key_file",
        ],
    },
]

# Flags the dashboard owns. Setting them by hand would fight the launcher.
#
# `--metrics` is here for a different reason from the rest: it is off by default
# in llama.cpp — /metrics then answers 501, "start it with --metrics" — the
# Metrics tab has nothing to read without it, and enabling an endpoint costs
# nothing. So the launcher turns it on rather than leaving the operator with a
# panel that can never fill.
MANAGED_FLAGS = {"model", "hf_repo", "hf_file", "host", "port", "metrics"}

# Flags whose value is something on disk, rendered as a list of what is actually
# there rather than a text box. Values are what the *container* sees: the model
# cache is mounted at /hf and the output directory at /outputs.
PATH_KINDS = {
    "model": "model",
    "mmproj": "model",
    "spec_draft_model": "model",
    "lora": "adapter",
    "lora_scaled": "adapter",
    "chat_template_file": "template",
    "ssl_certfile": "cert",
    "ssl_cert_file": "cert",
    "ssl_key_file": "cert",
    "api_key_file": "cert",
    "slot_save_path": "directory",
    "path": "directory",
}


@lru_cache(maxsize=1)
def schema() -> dict[str, Any]:
    """The parameter document, or an empty one that says which image it is about.

    The fallback stamps the *llama.cpp* image rather than a generic placeholder,
    because `scripts/setup.sh` compares this key against the configured tag to
    decide whether to offer a regeneration, and the editor renders it as the
    form's subtitle. An empty document naming the wrong image is worse than an
    empty document.
    """
    path = settings.data_dir / "llamacpp_args.json"
    if not path.exists():
        return {"image": settings.llamacpp_image, "llamacpp_version": "unknown",
                "args": [], "sections": {}}
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def by_dest() -> dict[str, dict[str, Any]]:
    index = {arg["dest"]: arg for arg in schema().get("args", [])}
    for dest, kind in PATH_KINDS.items():
        if dest in index:
            index[dest] = {**index[dest], "path_kind": kind}
    return index


def ui_model() -> dict[str, Any]:
    """The shape the frontend renders: featured sections first, then the rest."""
    index = by_dest()
    used: set[str] = set()
    sections = []
    for section in FEATURED:
        entries = []
        for dest in section["flags"]:
            arg = index.get(dest)
            if arg is None or dest in MANAGED_FLAGS:
                continue
            entries.append(arg)
            used.add(dest)
        if entries:
            sections.append({**section, "flags": entries})

    rest: dict[str, list] = {}
    for dest, arg in index.items():
        if dest in used or dest in MANAGED_FLAGS:
            continue
        rest.setdefault(arg.get("group", "options"), []).append(arg)

    advanced = [
        {
            "id": f"group-{group}",
            "title": group,
            "blurb": schema().get("sections", {}).get(group, ""),
            "flags": sorted(args, key=lambda a: a["dest"]),
        }
        for group, args in sorted(rest.items())
    ]
    version = schema().get("llamacpp_version", "unknown")
    return {
        "image": schema().get("image", settings.llamacpp_image),
        "engine": "llamacpp",
        "label": "llama.cpp",
        "version": version,
        # The frontend read `vllm_version` before there was a second engine, and
        # still falls back to it. Answering both costs a key and no drift.
        "vllm_version": version,
        "featured": sections,
        "advanced": advanced,
        "managed": sorted(MANAGED_FLAGS),
    }


def _is_default(arg: dict[str, Any], value: Any) -> bool:
    """Mirrors vllm_spec._is_default: these values are never rendered into argv."""
    default = arg.get("default")
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return value == default


def _render(arg: dict[str, Any], value: Any) -> list[str]:
    """One flag and its value, as llama.cpp spells them.

    The one real divergence from vLLM's renderer is the boolean: argparse builds
    `--no-<flag>` for you, and llama.cpp does not, so a false value can only be
    emitted when the schema records the actual negative spelling.
    """
    flag = arg["flag"]
    widget = arg["widget"]

    if widget == "bool":
        truthy = (
            value if isinstance(value, bool)
            else str(value).lower() in ("1", "true", "yes", "on")
        )
        if truthy:
            return [flag]
        negative = arg.get("negative_flag")
        return [negative] if negative else []

    if widget == "list":
        # llama.cpp takes several values as one comma-separated argument, not as
        # repeated tokens the way an argparse nargs='+' flag does.
        if isinstance(value, (list, tuple, set)):
            items = [str(v) for v in value]
        else:
            items = [part for part in str(value).replace(",", " ").split() if part]
        return [flag, ",".join(items)] if items else []

    if widget == "json":
        rendered = value if isinstance(value, str) else json.dumps(value)
        return [flag, rendered]

    return [flag, str(value)]


def build_argv(
    model: str,
    params: dict[str, Any],
    *,
    host: str = "0.0.0.0",
    port: int | None = None,
) -> list[str]:
    """Turn a stored parameter dict into a full `llama-server` command.

    The model is a flag here, not a positional, and which flag it is depends on
    what the operator chose: a path inside the container is `-m`, and anything
    else is treated as a Hub reference and handed to `-hf`, which llama.cpp
    resolves and downloads itself through the mounted cache.
    """
    index = by_dest()
    argv = ["llama-server", "--host", host]
    if port is not None:
        argv += ["--port", str(port)]
    argv += model_argv(model)
    # Not negotiable and not offered in the form: without it /metrics answers
    # 501 and the Metrics tab is a panel that can never fill.
    argv += ["--metrics"]

    for dest, value in params.items():
        if dest in MANAGED_FLAGS:
            continue
        arg = index.get(dest)
        if arg is None:
            continue
        if _is_default(arg, value):
            continue
        argv += _render(arg, value)
    return argv


def looks_like_path(model: str) -> bool:
    """Whether this model reference names a file the container can open.

    A leading slash, or a `.gguf` anywhere in it. Everything else — `org/repo`,
    `org/repo:Q4_K_M` — is a Hub reference for llama.cpp to fetch.
    """
    text = str(model or "").strip()
    return text.startswith("/") or text.lower().endswith(".gguf")


def model_argv(model: str) -> list[str]:
    text = str(model or "").strip()
    if not text:
        return []
    return ["-m", text] if looks_like_path(text) else ["-hf", text]


def validate(params: dict[str, Any]) -> list[str]:
    """Cheap client-side-grade validation; llama-server remains the authority."""
    index = by_dest()
    problems: list[str] = []
    for dest, value in params.items():
        if dest in MANAGED_FLAGS:
            problems.append(f"{dest} is managed by the dashboard and cannot be set here")
            continue
        arg = index.get(dest)
        if arg is None:
            problems.append(f"unknown parameter '{dest}' for this llama.cpp build")
            continue
        if value is None or value == "":
            continue
        widget = arg["widget"]
        accepts = {str(word).lower() for word in arg.get("accepts") or ()}
        text = str(value).strip()
        if widget == "list" and isinstance(value, dict):
            problems.append(f"{arg['flag']}: expected a list or a value, not an object")
        elif widget == "enum" and arg.get("choices") and text not in arg["choices"]:
            problems.append(
                f"{arg['flag']}: '{value}' is not one of {', '.join(arg['choices'][:12])}"
            )
        elif widget in ("int", "size"):
            if text.lower() not in accepts and whole_number(text) is None:
                extra = f" or one of {', '.join(sorted(accepts))}" if accepts else ""
                problems.append(f"{arg['flag']}: '{value}' is not a whole number{extra}")
        elif widget == "float":
            try:
                number = float(text)
            except (TypeError, ValueError):
                problems.append(f"{arg['flag']}: '{value}' is not a number")
            else:
                # A value that parses and is still not a number. llama.cpp would
                # take it and then behave unpredictably.
                if number != number or number in (float("inf"), float("-inf")):
                    problems.append(f"{arg['flag']}: '{value}' is not a finite number")
        elif widget == "json" and isinstance(value, str) and text:
            try:
                json.loads(text)
            except ValueError:
                problems.append(f"{arg['flag']}: not valid JSON")
        elif widget == "bool" and not _boolean(value) and not arg.get("negative_flag"):
            problems.append(
                f"{arg['flag']}: this build has no way to turn this off — llama.cpp "
                "spells its negations and this flag has none, so leave it unset instead")
    problems.extend(_cross_flag_problems(params))
    return problems


def whole_number(value: Any) -> int | None:
    """One integer, or None. Never raises.

    `int(float(text))` is the obvious spelling and it has two escapes that are
    not ValueError: '1e400' floats to infinity and raises OverflowError, and
    'nan' floats to NaN and raises ValueError only on some paths. Both reached
    a pydantic field validator, where an uncaught exception is a 500 rather than
    the 422 the operator should have got.
    """
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    try:
        return int(number)
    except (ValueError, OverflowError):
        return None


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _cross_flag_problems(params: dict[str, Any]) -> list[str]:
    """Combinations that are individually legal and together fatal."""
    problems: list[str] = []
    # llama.cpp refuses outright rather than silently degrading: a quantised V
    # cache has no non-fused attention kernel, so with flash attention explicitly
    # off the context throws during initialisation, minutes after the launch
    # looked fine.
    v_type = str(params.get("cache_type_v") or "").strip().lower()
    # f32, f16 and bf16 are the unquantized types. llama.cpp gates the
    # flash-attention requirement on quantization, not on "anything but f16" —
    # refusing bf16 would block a combination the binary accepts.
    unquantized = ("f32", "f16", "bf16")
    if v_type and v_type not in unquantized \
            and str(params.get("flash_attn") or "").lower() == "off":
        problems.append(
            f"-ctv {v_type} needs flash attention, and --flash-attn is off. llama.cpp "
            "raises during context creation rather than falling back, so the container "
            "exits after the weights are read")
    return problems


def cross_flag_warnings(params: dict[str, Any]) -> list[str]:
    """Combinations that are legal, start fine, and do not do what was meant."""
    warnings: list[str] = []
    if params.get("jinja") is False:
        warnings.append(
            "--no-jinja falls back to a built-in chat template, which cannot express "
            "tool calls — requests carrying `tools` come back as ordinary text.")
    parallel = params.get("parallel")
    ctx = params.get("ctx_size")
    try:
        slots, context = int(parallel), int(ctx)
    except (TypeError, ValueError):
        slots = context = 0
    if slots > 1 and context and params.get("kv_unified") is False:
        warnings.append(
            f"--parallel {slots} with --no-kv-unified gives each slot {context // slots} "
            f"tokens, not {context}. The cache costs the same either way; only the "
            "division changes.")
    return warnings


def n_gpu_layers(params: dict[str, Any]) -> int | None:
    """The declared `-ngl`, or None for 'auto', 'all' and anything unset.

    None is not "zero": it means the operator did not pin a number, which
    app/engines/llamacpp.py prices as the whole model.
    """
    raw = params.get("n_gpu_layers")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("", "auto", "all", "-1"):
        return None
    return whole_number(text)


def ctx_size(params: dict[str, Any]) -> int | None:
    """The declared `--ctx-size`, or None when it is unset or 0 (from the model)."""
    value = whole_number(params.get("ctx_size"))
    return value if value and value > 0 else None
