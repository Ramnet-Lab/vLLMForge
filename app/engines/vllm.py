"""vLLM, as an engine object.

This file is an adapter and nothing else. Every member either delegates to
`app/vllm_spec.py` — which is not modified — or is a function moved out of
`app/safety.py` with its body unchanged. That is deliberate and it is the whole
compatibility argument: the second engine cannot alter the first if the first's
code is the same code.

Two things live here rather than staying in safety.py, and both moved because
leaving them would have made the import graph circular:

* `matches()` is `safety.is_vllm_command`. That name survives in safety.py as a
  delegate, because five call sites and eight tests read it, and because the
  point of `engines.recognise()` is that this predicate never has to widen.
* `implicit_util()` is `safety.default_util`, which reads only `vllm_spec`. Same
  arrangement: the name stays in safety.py as a one-line delegate so the budget
  payload's `default_util` key and every caller are untouched.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from app import vllm_spec
from app.engines import flag_value

# The three flags that decide what a vLLM container is really holding. The util
# is the fraction; the other two walk past it — --kv-cache-memory sizes the cache
# explicitly and overrides the fraction, and --cpu-offload-gb lands in the same
# pool on a unified part — which is why the footprint is not the fraction alone.
UTIL_FLAG = re.compile(r"--gpu[-_]memory[-_]utilization(?:=(\S+))?")
KV_BYTES_FLAG = re.compile(r"--kv[-_]cache[-_]memory(?:[-_]bytes)?(?:=(\S+))?$")
OFFLOAD_FLAG = re.compile(r"--cpu[-_]offload[-_]gb(?:=(\S+))?$")

# `vllm serve` as a word, anywhere in a string that may be a whole shell line.
_VLLM_SERVE = re.compile(r"(?:^|[\s;&|])vllm\s+serve(?:\s|$)")

# vLLM's own metrics, cherry-picked for the UI. Everything else stays in the raw
# /metrics passthrough.
INTERESTING_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:num_preemptions_total",
)


class VLLMEngine:
    name = "vllm"
    label = "vLLM"
    binary = "vllm serve"
    # The images this repo builds run docker/entrypoint.sh, which execs "$@", so
    # the full argv passed as the container command runs as written.
    entrypoint = None
    gpu = True
    supports_pooling = True
    served_name_dest = "served_model_name"
    interesting_metrics = INTERESTING_METRICS

    @property
    def default_image(self) -> str:
        from app.config import settings

        return settings.vllm_image

    # --- schema -----------------------------------------------------------

    def ui_model(self) -> dict[str, Any]:
        model = vllm_spec.ui_model()
        # Additive. `vllm_version` is retained because the frontend and
        # scripts/setup.sh both read it; `version`, `engine` and `label` are what
        # a second engine can also answer.
        return {**model, "engine": self.name, "label": self.label,
                "version": model.get("vllm_version", "unknown")}

    def validate(self, params: dict[str, Any]) -> list[str]:
        return vllm_spec.validate(params)

    def managed_flags(self) -> frozenset[str]:
        return frozenset(vllm_spec.MANAGED_FLAGS)

    def path_kinds(self) -> dict[str, str]:
        return dict(vllm_spec.PATH_KINDS)

    # --- command assembly -------------------------------------------------

    def build_argv(self, model: str, params: dict[str, Any], *,
                   host: str = "0.0.0.0", port: int | None = None) -> list[str]:
        return vllm_spec.build_argv(model, params, host=host, port=port)

    def env_overlay(self, params: dict[str, Any]) -> dict[str, str]:
        # Verbatim from servers.build_env: preloading adapters is one thing,
        # loading them at runtime is another, and vLLM gates the second on this.
        if (params or {}).get("enable_lora"):
            return {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}
        return {}

    # --- memory -----------------------------------------------------------

    def argv_of(self, command: list[str] | None) -> list[str]:
        """A container's command as tokens, whether or not a shell is in the way.

        A pooled engine is launched as `bash -lc "ray start ... && exec vllm
        serve ..."` — one element, because the Ray head and the engine have to
        share a container. Treating that as opaque made the dashboard's own
        pooled servers weigh nothing in the budget, so a second launch could be
        admitted on top of memory that was already spoken for.
        """
        if not command:
            return []
        for token in command:
            if not _VLLM_SERVE.search(token):
                continue
            if len(token.split()) == 1:
                break
            try:
                parts = shlex.split(token)
            except ValueError:
                continue
            for index, part in enumerate(parts):
                if part.endswith("vllm") and parts[index + 1:index + 2] == ["serve"]:
                    return parts[index:]
        return list(command)

    def matches(self, command: list[str] | None) -> bool:
        argv = self.argv_of(command)
        return bool(argv) and any(token == "serve" for token in argv) and any(
            "vllm" in token for token in argv[:2]
        )

    def command_params(self, command: list[str] | None) -> dict[str, Any]:
        """The memory-relevant flags of a running container, shaped like stored
        args, so a hand-launched container is accounted exactly like a managed
        one."""
        argv = self.argv_of(command)
        params: dict[str, Any] = {}
        util = self.parse_util(command)
        if util is not None:
            params["gpu_memory_utilization"] = util
        kv = flag_value(argv, KV_BYTES_FLAG)
        if kv is not None:
            params["kv_cache_memory_bytes"] = kv
        offload = flag_value(argv, OFFLOAD_FLAG)
        if offload is not None:
            params["cpu_offload_gb"] = offload
        return params

    def parse_util(self, command: list[str] | None) -> float | None:
        """Pull --gpu-memory-utilization out of a container's argv."""
        try:
            return float(flag_value(self.argv_of(command), UTIL_FLAG))
        except (TypeError, ValueError):
            return None

    async def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        """Nothing to look up: everything vLLM's footprint needs is in the argv."""
        return params

    def footprint_bytes(self, params: dict[str, Any], total_bytes: int) -> int:
        return vllm_spec.footprint_bytes(params, total_bytes,
                                         default_util=self.implicit_util())

    def declared_util(self, params: dict[str, Any]) -> float | None:
        return vllm_spec.gpu_memory_utilization(params or {})

    def implicit_util(self) -> float:
        """vLLM's own default, read from the image's schema so it tracks upgrades."""
        arg = vllm_spec.by_dest().get("gpu_memory_utilization") or {}
        try:
            return float(arg.get("default"))
        except (TypeError, ValueError):
            return 0.92

    def notes(self, params: dict[str, Any], *, implicit: bool) -> list[str]:
        notes: list[str] = []
        if implicit:
            # A serve command with no --gpu-memory-utilization is not free: vLLM
            # applies its own default, which on this host is over 100 GiB.
            notes.append(
                "no --gpu-memory-utilization set, so vLLM uses its default of "
                f"{self.implicit_util():g}"
            )
        if "kv_cache_memory_bytes" in (params or {}):
            notes.append("--kv-cache-memory overrides the utilisation fraction")
        if "cpu_offload_gb" in (params or {}):
            notes.append("--cpu-offload-gb lands in the same unified pool")
        return notes

    # --- discovery and status ---------------------------------------------

    def model_from_argv(self, argv: list[str] | None) -> str:
        """`vllm serve <model>` — the model is the positional after the verb."""
        argv = self.argv_of(argv)
        for index, token in enumerate(argv):
            if token == "serve" and index + 1 < len(argv):
                candidate = argv[index + 1]
                return "" if candidate.startswith("-") else candidate
        return ""

    def is_loading(self, *, reachable: bool, healthy: bool, **_extra: Any) -> bool:
        """vLLM does not bind its port until the weights are loaded and the CUDA
        graphs captured, so a refused connection means "still loading", not
        "broken"."""
        return not reachable


ENGINE = VLLMEngine()
