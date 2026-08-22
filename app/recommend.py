"""A configuration that starts on the first try.

The Serve form offers around 190 flags. Almost all of them are tuning; a handful
decide whether the engine comes up at all, and getting those wrong costs four
minutes of loading followed by a traceback. This module answers, for one model
on one machine, which of that handful to set and to what.

Three principles, each of which removes work rather than adding it:

* **Set as little as possible.** vLLM already detects the quantisation method,
  already resolves the dtype, already reads generation_config.json, and already
  loads a chat template that transformers put on the tokenizer. A flag set to
  what vLLM would have chosen is not help — it is a second source of truth that
  goes stale when the image is upgraded.
* **Prefer a flag that lets vLLM decide over a number computed here.**
  `--max-model-len auto` makes vLLM fit the context to whatever KV cache is
  actually left after it has profiled the model. No offline arithmetic can beat
  a measurement, and the arithmetic is not even derivable for hybrid or MLA
  models.
* **Say what cannot be fixed.** A LoRA adapter, GGUF-only weights, or a model
  whose weights exceed free memory are not configuration problems, and offering
  flags for them wastes the operator's afternoon.

Every rule below was checked against vLLM 0.24.0's own source in the image this
dashboard runs, and against the models actually cached on this box.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

from app import model_profile, safety, telemetry, vllm_spec
from app.model_profile import Profile

# vLLM compares its requested memory against a snapshot of free memory taken as
# the engine starts, and that number drifts while a container image is pulled
# and a model is read off disk. Three points of headroom is what keeps a
# recommendation from being refused by a machine that got a little busier.
UTIL_MARGIN = 0.03

# Flags a recommendation deliberately leaves alone, with the reason. Shown to
# the operator, because "why didn't it set the quantisation" is the obvious
# question and the answer is that setting it makes things worse.
LEFT_ALONE = (
    ("quantization", "the checkpoint declares its own method and vLLM detects it; "
                     "naming it here can only disagree"),
    ("dtype", "vLLM resolves it from the checkpoint and refuses float16 where a model "
              "cannot take it"),
    ("kv_cache_dtype", "a quantised checkpoint can carry its own KV scheme, and an "
                       "explicit value overrides it"),
    ("chat_template", "transformers loads the template the repo shipped; a path here "
                      "replaces it"),
    ("served_model_name", "the Served name field above already sets it"),
)


@dataclass
class Suggestion:
    dest: str
    value: Any
    why: str


@dataclass
class Finding:
    level: str
    """ok, warn or block."""
    text: str


@dataclass
class Recommendation:
    model: str = ""
    node: str = ""
    ok: bool = True
    level: str = "ok"
    headline: str = ""
    suggestions: list[Suggestion] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    left_alone: list[dict[str, str]] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "node": self.node,
            "ok": self.ok,
            "level": self.level,
            "headline": self.headline,
            "args": {s.dest: s.value for s in self.suggestions},
            "suggestions": [{"dest": s.dest, "value": s.value, "why": s.why}
                            for s in self.suggestions],
            "findings": [{"level": f.level, "text": f.text} for f in self.findings],
            "left_alone": self.left_alone,
            "profile": self.profile,
        }


def _gib(value: float) -> str:
    return f"{value / 1024 ** 3:.1f} GiB"


def safe_utilisation(total_bytes: int, available_bytes: int, budget_free_util: float) -> float:
    """The largest --gpu-memory-utilization that will actually start.

    vLLM's startup check is `ceil(total_memory * util) > free_memory -> refuse`,
    and on a unified-memory box total_memory is host MemTotal while free_memory
    is MemAvailable. So the ceiling is MemAvailable/MemTotal, not 1.0, and the
    0.92 default fails on any machine holding more than 8% of its RAM — which
    this one always is.

    The dashboard's own budget is the other bound: it holds a reserve back for
    the OS and for torch.compile, and refusing to exceed that is the whole point
    of the guard. The recommendation takes whichever is tighter.
    """
    if total_bytes <= 0:
        return 0.0
    vllm_ceiling = max(0.0, available_bytes / total_bytes - UTIL_MARGIN)
    allowed = min(vllm_ceiling, max(0.0, budget_free_util))
    return math.floor(allowed * 100) / 100


async def build(model: str, node: str = "", args: dict[str, Any] | None = None) -> Recommendation:
    """What to set for this model on this node, and what stands in the way."""
    from app import nodes

    target = nodes.by_name(node)
    rec = Recommendation(model=model, node=target.name)
    if not str(model or "").strip():
        rec.headline = "Choose a model first."
        return rec

    if target.is_local:
        profile = await asyncio.to_thread(model_profile.read, model)
    else:
        profile = await model_profile.read_remote(model, target.name)
    rec.profile = profile.to_dict()

    budget = await safety.current_budget(node=target)
    memory = telemetry.read_meminfo() if target.is_local else None
    total = memory.total_bytes if memory else budget.total_bytes
    available = memory.available_bytes if memory else budget.available_bytes

    _blockers(rec, profile, available)
    if not rec.ok:
        return rec

    # The memory advice is about this machine and holds whether or not the model
    # is here yet. Everything else is read from files, so with no files there is
    # nothing to say — and saying "no chat template" about a repo nobody has
    # pulled would be a guess dressed as a fact.
    _memory(rec, profile, total, available, budget.free_util)
    if profile.found:
        _loading(rec, profile)
        _serving(rec, profile)

    rec.left_alone = [{"dest": dest, "why": why} for dest, why in LEFT_ALONE]
    _finish(rec, profile, args or {})
    return rec


def _blockers(rec: Recommendation, profile: Profile, available: int) -> None:
    """The cases no flag rescues. Offering settings for these wastes an afternoon."""
    if not profile.found:
        rec.headline = "Not on this machine yet."
        rec.findings.append(Finding("warn", (
            "Nothing is cached here under that name, so there are no files to read. The first "
            "start downloads it, and this page can say more once it has.")))
        rec.level = "warn"
        return

    if profile.is_adapter:
        rec.ok = False
        rec.level = "block"
        rec.headline = "This is a LoRA adapter, not a servable model."
        rec.findings.append(Finding("block", (
            f"Serve {profile.base_model or 'its base model'} with --enable-lora and attach this "
            "on top of it.")))
        return

    if profile.has_gguf and not profile.has_safetensors:
        rec.ok = False
        rec.level = "block"
        rec.headline = "GGUF weights only."
        rec.findings.append(Finding("block", (
            "This image loads safetensors. There is no flag that makes it read GGUF.")))
        return

    if profile.weight_bytes and available and profile.weight_bytes >= available:
        rec.ok = False
        rec.level = "block"
        rec.headline = "The weights alone do not fit."
        rec.findings.append(Finding("block", (
            f"{_gib(profile.weight_bytes)} of weights against {_gib(available)} free. No "
            "--gpu-memory-utilization serves it on this node right now — stop something, or "
            "pool it across machines.")))


def _memory(rec: Recommendation, profile: Profile, total: int, available: int,
            free_util: float) -> None:
    util = safe_utilisation(total, available, free_util)
    if util > 0:
        rec.suggestions.append(Suggestion(
            "gpu_memory_utilization", util,
            f"vLLM refuses to start when the fraction it asks for exceeds free memory, and "
            f"{_gib(available)} of {_gib(total)} is free — so its own default of "
            f"{safety.default_util():g} would be refused on this machine."))

    # Letting vLLM fit the context beats computing it: it sizes max_model_len
    # against the KV cache left after it has profiled the real model, which is
    # a measurement no offline arithmetic can match — and for hybrid attention
    # or MLA the arithmetic is not derivable from config.json at all.
    rec.suggestions.append(Suggestion(
        "max_model_len", "auto",
        "vLLM measures what is left after loading and fits the context to it, instead of "
        "refusing to start because the config advertises more than the KV cache can hold."))

    if profile.max_position_embeddings and profile.kv_bytes(profile.max_position_embeddings):
        full = profile.kv_bytes(profile.max_position_embeddings) or 0
        budget_bytes = int(util * total) - profile.weight_bytes
        if budget_bytes > 0 and full > budget_bytes:
            rec.findings.append(Finding("warn", (
                f"Its full {profile.max_position_embeddings:,}-token context needs {_gib(full)} "
                f"of KV cache and about {_gib(budget_bytes)} is left after the weights, so the "
                "served context will be shorter than the config advertises.")))


def _loading(rec: Recommendation, profile: Profile) -> None:
    if profile.requires_remote_code:
        rec.suggestions.append(Suggestion(
            "trust_remote_code", True,
            "config.json maps its modelling code into the repo, and vLLM will not execute that "
            "without being told it may."))

    if profile.supported is False:
        arch = (profile.architectures or ["it"])[0]
        rec.findings.append(Finding("warn", (
            f"This image has no native implementation of {arch}. It may still load through the "
            "Transformers backend, which is slower; nothing here can tell in advance.")))

    if not profile.architectures and profile.model_type:
        rec.findings.append(Finding("warn", (
            "config.json names no architecture, so vLLM guesses one from model_type — and the "
            "name it guesses often resolves to an embedding server.")))


def _serving(rec: Recommendation, profile: Profile) -> None:
    if profile.runner == "pooling":
        rec.findings.append(Finding("warn", (
            f"This will serve embeddings, not chat: {profile.runner_reason}. /v1/embeddings "
            "works; /v1/chat/completions refuses.")))
        return

    if not profile.chat_template:
        rec.findings.append(Finding("warn", (
            "The repo ships no chat template. The server starts and /v1/completions works, but "
            "every /v1/chat/completions request fails until a template is supplied.")))


def _finish(rec: Recommendation, profile: Profile, args: dict[str, Any]) -> None:
    worst = "ok"
    for finding in rec.findings:
        if finding.level == "block":
            worst = "block"
        elif finding.level == "warn" and worst == "ok":
            worst = "warn"
    rec.level = worst

    changes = [s for s in rec.suggestions if args.get(s.dest) != s.value]
    if not rec.headline:
        if not changes:
            rec.headline = "Already configured to start."
        else:
            names = ", ".join(vllm_spec.by_dest().get(s.dest, {}).get("flag", s.dest)
                              for s in changes)
            rec.headline = f"Set {names} and it should start on the first try."
