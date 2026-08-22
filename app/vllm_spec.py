"""The vLLM parameter model.

`app/data/vllm_args.json` is generated from the actual image by
`tools/gen_vllm_schema.py`, so the parameter form in the UI always matches the
build that will run. This module turns that schema into (a) something the
frontend can render and (b) a `vllm serve` argv.

Only a curated subset is promoted to the top of the form; the remaining ~200
flags stay available under "All parameters" so nothing is unreachable.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import settings

# Ordered UI sections. Anything not listed here still appears under "Advanced",
# searchable by flag name.
FEATURED: list[dict[str, Any]] = [
    {
        "id": "model",
        "title": "Model & loading",
        "blurb": "What to serve and how the weights are read.",
        "flags": [
            "served_model_name", "tokenizer", "revision", "dtype", "quantization",
            "load_format", "trust_remote_code", "max_model_len", "hf_overrides",
            "config_format", "tokenizer_mode", "seed", "download_dir",
        ],
    },
    {
        "id": "memory",
        "title": "Memory & KV cache",
        "blurb": (
            "On a unified-memory host these decide whether the machine survives. "
            "gpu-memory-utilization is a fraction of total HOST memory."
        ),
        "flags": [
            "gpu_memory_utilization", "kv_cache_dtype", "kv_cache_memory_bytes",
            "block_size", "swap_space", "cpu_offload_gb", "num_gpu_blocks_override",
            "enable_prefix_caching", "prefix_caching_hash_algo",
        ],
    },
    {
        "id": "scheduling",
        "title": "Scheduling & concurrency",
        "blurb": (
            "max-num-seqs is an admission cap, not a request limit: extra requests "
            "queue. It is the real safety valve against activation spikes."
        ),
        "flags": [
            "max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill",
            "scheduling_policy", "long_prefill_token_threshold",
            "max_num_partial_prefills", "disable_log_stats",
        ],
    },
    {
        "id": "parallel",
        "title": "Parallelism",
        "blurb": (
            "One GPU per Spark, so prefer pipeline parallel across nodes over tensor parallel."
        ),
        "flags": [
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
            "distributed_executor_backend", "enable_expert_parallel",
            "max_parallel_loading_workers",
        ],
    },
    {
        "id": "lora",
        "title": "LoRA adapters",
        "blurb": "Serve adapters produced by the fine-tuning tab without merging them.",
        "flags": [
            "enable_lora", "lora_modules", "max_loras", "max_lora_rank",
            "max_cpu_loras", "fully_sharded_loras", "default_mm_loras",
        ],
    },
    {
        "id": "tools",
        "title": "Tools, reasoning & templates",
        "blurb": "Parsers must match the model family or tool calls come back as plain text.",
        "flags": [
            "enable_auto_tool_choice", "tool_call_parser", "tool_parser_plugin",
            "reasoning_parser", "chat_template", "chat_template_content_format",
            "response_role",
        ],
    },
    {
        "id": "structured",
        "title": "Structured outputs",
        "blurb": "Grammar-constrained decoding backend.",
        "flags": ["structured_outputs_config", "logits_processors"],
    },
    {
        "id": "multimodal",
        "title": "Multimodal",
        "blurb": "Only meaningful for vision/audio models.",
        "flags": [
            "limit_mm_per_prompt", "mm_processor_kwargs", "mm_processor_cache_gb",
            "media_io_kwargs", "skip_mm_profiling",
        ],
    },
    {
        "id": "frontend",
        "title": "HTTP frontend",
        "blurb": "Host and port are managed for you; these cover auth and logging.",
        "flags": [
            "api_key", "allowed_origins", "allowed_methods", "allowed_headers",
            "max_log_len", "uvicorn_log_level", "enable_log_requests",
            "enable_server_load_tracking", "enable_prompt_tokens_details",
        ],
    },
]

# Flags the dashboard owns. Setting them by hand would fight the launcher.
MANAGED_FLAGS = {"host", "port", "model", "model_tag", "config"}

# Flags whose value is something on disk. The form renders these as a list of
# what is actually there rather than a text box, because a mistyped path is only
# discovered minutes later, when the engine has already started and failed.
#
# The values offered are what the *container* sees: vLLM runs with the model
# cache at /hf and the output directory at /outputs, so a host path would not
# resolve inside it.
PATH_KINDS = {
    "model": "model",
    "tokenizer": "model",
    "hf_config_path": "model",
    "generation_config": "model",
    "lora_modules": "adapter",
    "chat_template": "template",
    "tool_parser_plugin": "plugin",
    "reasoning_parser_plugin": "plugin",
    "logits_processors": "plugin",
    "download_dir": "directory",
    "allowed_local_media_path": "directory",
    "ssl_certfile": "cert",
    "ssl_keyfile": "cert",
    "ssl_ca_certs": "cert",
    "mm_encoder_fp8_scale_path": "json",
    "mm_encoder_fp8_scale_save_path": "directory",
    "speculative_config": "json",
}


# vLLM leaves a handful of Frontend flags with no help text of their own.
HELP_OVERLAY = {
    "enable_auto_tool_choice": (
        "Let the model choose tools automatically. Requires --tool-call-parser."
    ),
    "tool_call_parser": (
        "Parser that turns the model's tool-call syntax into OpenAI tool_calls. "
        "Must match the model family."
    ),
    "tool_parser_plugin": "Path to a Python file registering a custom tool parser.",
    "chat_template": (
        "Path to, or inline text of, a Jinja chat template overriding the model's own."
    ),
    "chat_template_content_format": (
        "How multimodal content is rendered into the template: 'string' flattens to text, "
        "'openai' keeps the parts list."
    ),
    "response_role": "Role reported for generated messages when add_generation_prompt is used.",
    "lora_modules": (
        "Adapters to preload, as name=path pairs or JSON objects. "
        "Runtime loading also needs VLLM_ALLOW_RUNTIME_LORA_UPDATING=1."
    ),
    "max_log_len": "Truncate logged prompts to this many characters.",
    "return_tokens_as_token_ids": "Report tokens as 'token_id:N' strings instead of text.",
    "enable_prompt_tokens_details": "Include cached-token details in the usage block.",
    "enable_server_load_tracking": "Expose a server-load gauge on /metrics.",
    "enable_log_outputs": "Log generated completions as well as requests.",
}


@lru_cache(maxsize=1)
def schema() -> dict[str, Any]:
    path = settings.data_dir / "vllm_args.json"
    if not path.exists():
        return {"image": settings.vllm_image, "vllm_version": "unknown", "args": [], "sections": {}}
    data = json.loads(Path(path).read_text())
    for arg in data["args"]:
        if not arg.get("help") and arg["dest"] in HELP_OVERLAY:
            arg["help"] = HELP_OVERLAY[arg["dest"]]
        if arg["dest"] in PATH_KINDS:
            arg["path_kind"] = PATH_KINDS[arg["dest"]]
    return data


@lru_cache(maxsize=1)
def by_dest() -> dict[str, dict[str, Any]]:
    return {arg["dest"]: arg for arg in schema()["args"]}


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
    for arg in schema()["args"]:
        if arg["dest"] in used or arg["dest"] in MANAGED_FLAGS:
            continue
        rest.setdefault(arg["group"], []).append(arg)

    advanced = [
        {
            "id": f"group-{group}",
            "title": group,
            "blurb": schema()["sections"].get(group, ""),
            "flags": sorted(args, key=lambda a: a["dest"]),
        }
        for group, args in sorted(rest.items())
    ]
    return {
        "image": schema()["image"],
        "vllm_version": schema()["vllm_version"],
        "featured": sections,
        "advanced": advanced,
        "managed": sorted(MANAGED_FLAGS),
    }


def _is_default(arg: dict[str, Any], value: Any) -> bool:
    default = arg.get("default")
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return value == default


def _render(arg: dict[str, Any], value: Any) -> list[str]:
    flag = arg["flag"]
    widget = arg["widget"]

    if widget == "bool":
        truthy = (
            value if isinstance(value, bool)
            else str(value).lower() in ("1", "true", "yes", "on")
        )
        if truthy:
            return [flag]
        if arg.get("negatable"):
            return [f"--no-{flag.lstrip('-')}"]
        return []

    if widget == "list":
        if isinstance(value, str):
            items = [part for part in value.replace(",", " ").split() if part]
        elif isinstance(value, (list, tuple, set)):
            items = [str(v) for v in value]
        else:
            # A bare scalar for a list flag is what a hand-edited config looks
            # like; one value is a perfectly good list of one.
            items = [str(value)]
        return [flag, *items] if items else []

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
    """Turn a stored parameter dict into a full `vllm serve` command."""
    index = by_dest()
    argv = ["vllm", "serve", model, "--host", host]
    if port is not None:
        argv += ["--port", str(port)]

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


def validate(params: dict[str, Any]) -> list[str]:
    """Cheap client-side-grade validation; vLLM remains the real authority."""
    index = by_dest()
    problems: list[str] = []
    for dest, value in params.items():
        if dest in MANAGED_FLAGS:
            problems.append(f"{dest} is managed by the dashboard and cannot be set here")
            continue
        arg = index.get(dest)
        if arg is None:
            problems.append(f"unknown parameter '{dest}' for this vLLM build")
            continue
        if value is None or value == "":
            continue
        widget = arg["widget"]
        if widget == "list" and isinstance(value, dict):
            problems.append(f"{arg['flag']}: expected a list or a value, not an object")
        elif widget == "enum" and arg.get("choices") and str(value) not in arg["choices"]:
            problems.append(
                f"{arg['flag']}: '{value}' is not one of {', '.join(arg['choices'][:12])}"
            )
        elif widget in ("int", "size"):
            text = str(value).strip()
            # vLLM accepts human sizes and, for several flags, 'auto'.
            if text.lower() != "auto" and parse_size(text) is None:
                problems.append(
                    f"{arg['flag']}: '{value}' is not an integer or a size like 256k or 1M"
                )
        elif widget == "float":
            try:
                float(value)
            except (TypeError, ValueError):
                problems.append(f"{arg['flag']}: '{value}' is not a number")
        elif widget == "json" and isinstance(value, str) and value.strip():
            try:
                json.loads(value)
            except ValueError:
                problems.append(f"{arg['flag']}: not valid JSON")
    return problems


def gpu_memory_utilization(params: dict[str, Any]) -> float | None:
    raw = params.get("gpu_memory_utilization")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


_SIZE_SUFFIX = {"k": 10 ** 3, "m": 10 ** 6, "g": 10 ** 9, "t": 10 ** 12,
                "ki": 2 ** 10, "mi": 2 ** 20, "gi": 2 ** 30, "ti": 2 ** 40}


def parse_size(value: Any) -> int | None:
    """vLLM's human-readable byte sizes: 25.6k, 100G, 8Gi, or a plain integer."""
    if value in (None, "", "auto"):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().rstrip("Bb")
    for suffix, scale in sorted(_SIZE_SUFFIX.items(), key=lambda kv: -len(kv[0])):
        if text.lower().endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * scale)
            except ValueError:
                return None
    try:
        return int(float(text))
    except ValueError:
        return None


def footprint_bytes(params: dict[str, Any], total_bytes: int, *, default_util: float) -> int:
    """A floor on the host memory a launch with these parameters will take.

    The utilisation fraction is not the whole story. --kv-cache-memory sizes the
    KV cache explicitly and *overrides* the fraction, so a config can pair a
    tiny util with a huge cache and look free. --cpu-offload-gb moves weights to
    system RAM, which on a unified-memory part is the same pool. Both are added
    here so the guard cannot be walked past.
    """
    util = gpu_memory_utilization(params)
    engine = int((default_util if util is None else util) * total_bytes)

    kv = parse_size(params.get("kv_cache_memory_bytes") or params.get("kv_cache_memory"))
    if kv is not None:
        engine = max(engine, kv)

    try:
        offload = float(params.get("cpu_offload_gb") or 0)
    except (TypeError, ValueError):
        offload = 0.0
    return engine + int(offload * 1024 ** 3)
