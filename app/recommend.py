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

# What a first-try configuration plans for. vLLM will still fit the context to
# whatever is actually left, so this only decides how much of the machine to
# reserve — and reserving all of it for a model that wants a fraction is how a
# box ends up holding one engine when it could hold three.
TARGET_CONTEXT = 32768

# Everything resident that is neither weights nor KV cache: the CUDA context,
# torch's allocator, compiled graphs, activations. vLLM measures this with a
# real profiling run and no offline figure can match it, so this is a constant
# with the concurrency factor below absorbing the error.
OVERHEAD_BYTES = 6 * 1024 ** 3
MULTIMODAL_OVERHEAD_BYTES = 2 * 1024 ** 3

# KV for more than one sequence at a time. A server that can only hold a single
# conversation is not one anybody wanted.
CONCURRENCY = 2

# Flags a recommendation deliberately leaves alone, with the reason. Shown to
# the operator, because "why didn't it set the quantisation" is the obvious
# question and the answer is that setting it makes things worse.
LEFT_ALONE = (
    ("kv_cache_memory_bytes", "vLLM sizes the cache from what is left after loading, and a "
                              "number here overrides that measurement with a guess"),
    ("enforce_eager", "cuda graphs are worth their memory; turn this on only to diagnose"),
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


def floor_utilisation(profile: Profile, total_bytes: int) -> float | None:
    """The least utilisation that could possibly work.

    vLLM's budget has to cover the weights before it covers anything else. Below
    this, `--max-model-len auto` gives up with "not enough GPU memory available
    to serve even a single token", and no other flag rescues it — so a value
    under this floor is not a tight configuration, it is a broken one.
    """
    if total_bytes <= 0 or not profile.weight_bytes:
        return None
    overhead = OVERHEAD_BYTES + (MULTIMODAL_OVERHEAD_BYTES if profile.is_multimodal else 0)
    return math.ceil((profile.weight_bytes + overhead) / total_bytes * 100) / 100


def needed_utilisation(profile: Profile, total_bytes: int, context: int) -> float | None:
    """The fraction this model actually wants, as opposed to the most it may have.

    Weights, plus a fixed overhead for the CUDA context and compiled graphs,
    plus KV cache for a couple of concurrent sequences at the target context.
    Rounded up: the point is to fit, not to be exact.
    """
    if total_bytes <= 0 or not profile.weight_bytes:
        return None
    kv = profile.kv_bytes(context) or 0
    overhead = OVERHEAD_BYTES + (MULTIMODAL_OVERHEAD_BYTES if profile.is_multimodal else 0)
    wanted = profile.weight_bytes + overhead + CONCURRENCY * kv
    return math.ceil(wanted / total_bytes * 100) / 100


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


async def build(model: str, node: str = "", args: dict[str, Any] | None = None,
                pool: list[str] | None = None) -> Recommendation:
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

    _placement(rec, profile, total, available, pool or [])

    # The memory advice is about this machine and holds whether or not the model
    # is here yet. Everything else is read from files, so with no files there is
    # nothing to say — and saying "no chat template" about a repo nobody has
    # pulled would be a guess dressed as a fact.
    _memory(rec, profile, total, available, budget.free_util)
    if profile.found:
        _loading(rec, profile)
        _quantisation(rec, profile)
        _serving(rec, profile)
        if profile.runner != "pooling":
            _parsers(rec, profile, args or {})
    _existing(rec, args or {})

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
            f"{_gib(profile.weight_bytes)} of weights against {_gib(available)} available. No "
            "--gpu-memory-utilization serves it on this node right now — stop something, or "
            "pool it across machines.")))


def _placement(rec: Recommendation, profile: Profile, total: int, available: int,
               pool: list[str]) -> None:
    """Whether spreading this across machines is buying anything.

    Pooling is not free and it is not merely slower: it costs a network hop per
    token per stage boundary, it fixes the world size so losing a node aborts
    the engine, and it runs pipeline-parallel code paths a single-node launch
    never touches — which is its own source of failures. A model that fits on
    one box should stay on one box.
    """
    if len(pool) < 2 or not profile.weight_bytes:
        return
    overhead = OVERHEAD_BYTES + (MULTIMODAL_OVERHEAD_BYTES if profile.is_multimodal else 0)
    kv = profile.kv_bytes(min(profile.max_position_embeddings or TARGET_CONTEXT,
                              TARGET_CONTEXT)) or 0
    wanted = profile.weight_bytes + overhead + CONCURRENCY * kv
    if wanted <= available:
        rec.findings.append(Finding("warn", (
            f"This does not need {len(pool)} machines. {_gib(profile.weight_bytes)} of weights "
            f"and its cache want about {_gib(wanted)}, and {_gib(available)} is free on one — so "
            "pooling buys nothing and costs a network hop per token, an engine that aborts if "
            "either node drops, and the pipeline-parallel paths, which is a class of failure a "
            "single-machine launch never meets.")))


def _memory(rec: Recommendation, profile: Profile, total: int, available: int,
            free_util: float) -> None:
    ceiling = safe_utilisation(total, available, free_util)
    context = min(profile.max_position_embeddings or TARGET_CONTEXT, TARGET_CONTEXT)
    wanted = needed_utilisation(profile, total, context) if profile.found else None
    floor = floor_utilisation(profile, total) if profile.found else None

    # A value under the floor is not a tight configuration, it is one that
    # cannot load the weights. Better to say so than to hand over a number that
    # is certain to fail after four minutes of reading safetensors.
    if floor and floor > ceiling:
        rec.ok = False
        rec.level = "block"
        rec.headline = "Not enough room for this right now."
        rec.findings.append(Finding("block", (
            f"The weights alone need {floor:g} of this machine and only {ceiling:g} can be "
            f"given — {_gib(available)} is available of {_gib(total)}. vLLM would read the whole "
            "checkpoint and then give up with \"not enough GPU memory available to serve even a "
            "single token\". Stop something, or pool it across machines.")))
        return

    util = min(ceiling, wanted) if wanted else ceiling
    if floor:
        util = max(util, floor)
    if util > 0:
        if wanted and wanted <= ceiling:
            why = (
                f"Enough for {_gib(profile.weight_bytes)} of weights and a {context:,}-token "
                f"context for a couple of callers at once. The ceiling here is {ceiling:g} — "
                "raise it for a longer context, or leave the rest for another engine.")
        else:
            why = (
                f"The most this machine can give: vLLM refuses to start when the fraction it "
                f"asks for exceeds available memory, and {_gib(available)} of {_gib(total)} is "
                f"available, so its own default of {safety.default_util():g} would be refused.")
        rec.suggestions.append(Suggestion("gpu_memory_utilization", util, why))

    # Letting vLLM fit the context beats computing it: it sizes max_model_len
    # against the KV cache left after it has profiled the real model, which is
    # a measurement no offline arithmetic can match — and for hybrid attention
    # or MLA the arithmetic is not derivable from config.json at all.
    rec.suggestions.append(Suggestion(
        "max_model_len", "auto",
        "vLLM measures what is left after loading and fits the context to it, instead of "
        "refusing to start because the config advertises more than the KV cache can hold."))

    _context_note(rec, profile, util, total)


def _context_note(rec: Recommendation, profile: Profile, util: float, total: int) -> None:
    """Roughly what context this will actually serve.

    A statement, not a warning: with --max-model-len auto a context shorter than
    the config advertises is the normal, correct outcome, and calling it a
    problem trains the operator to ignore the panel.
    """
    advertised = profile.max_position_embeddings
    per_token = profile.kv_bytes_per_token()
    if not (advertised and per_token):
        return

    overhead = OVERHEAD_BYTES + (MULTIMODAL_OVERHEAD_BYTES if profile.is_multimodal else 0)
    for_kv = int(util * total) - profile.weight_bytes - overhead
    if for_kv <= 0:
        return
    fits = int(for_kv / CONCURRENCY / per_token)
    if fits >= advertised:
        return
    rec.findings.append(Finding("ok", (
        f"That fits roughly {fits // 1024:,}k tokens of context, of the {advertised // 1024:,}k "
        "the config advertises. vLLM measures the real figure at startup; raise the "
        "utilisation for more.")))


def _quantisation(rec: Recommendation, profile: Profile) -> None:
    """Two checkpoint shapes that load and then fail, both present in this cache.

    Neither is a configuration problem — no flag rescues either — so they are
    said before a launch rather than discovered from an abort partway through
    weight loading.
    """
    quant = profile.quantization or {}
    groups = quant.get("config_groups") or {}
    strategies = {
        str((group.get("weights") or {}).get("strategy") or "")
        for group in groups.values() if isinstance(group, dict)
    }

    if profile.quant_method == "compressed-tensors" and "block" in strategies:
        rec.findings.append(Finding("warn", (
            "These are FP8 block-quantised weights, and block layouts hit a DeepGEMM assertion "
            "on this GPU in vLLM 0.24 — the engine loads them and then aborts. A per-channel or "
            "per-tensor FP8 build of the same model avoids it, as does the unquantised one.")))

    # No ignore-list clause here, and that is deliberate. ModelOpt never
    # quantises embed_tokens — get_quant_method returns None for a
    # VocabParallelEmbedding, which becomes UnquantizedEmbeddingMethod and does
    # implement tie_weights. The object that raises is lm_head, and excluding it
    # does not help: an excluded ParallelLMHead gets UnquantizedLinearMethod,
    # which lacks tie_weights just as the quantised method does. Treating the
    # ignore list as an escape hatch would silence a checkpoint that still fails.
    if profile.quant_method == "modelopt" and profile.tie_word_embeddings:
        rec.findings.append(Finding("warn", (
            "This ModelOpt checkpoint ties lm_head to the input embedding, and no ModelOpt "
            "quantisation method implements tie_weights — construction raises "
            "NotImplementedError before any weights are read. A compressed-tensors build of the "
            "same model works, because its excluded lm_head falls through to the unquantised "
            "embedding method, which does implement it. So does the unquantised base model.")))


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


def _parsers(rec: Recommendation, profile: Profile, args: dict[str, Any]) -> None:
    """Which tool and reasoning parser this template expects.

    vLLM picks these by name, and the name is written down nowhere in the repo —
    but the tokens the template emits are, and each parser keys on its own. Only
    the unambiguous cases are offered as flags; a bare <think>/</think> is claimed
    by five parsers that differ in what they do with the tokens, so it is named
    rather than set.
    """
    markers = set(profile.template_markers or [])
    if not markers:
        return

    tool = ""
    if "<|tool_call>" in markers:
        tool = "gemma4"
    elif ({"<tool_call>", "<function=", "<parameter="} <= markers
            and profile.model_type.startswith("qwen3")):
        # step3p5's parser reads the same three tokens, so the three-token lock
        # alone does not identify qwen3 — the model family is what disambiguates.
        tool = "qwen3_xml"

    if tool and not args.get("tool_call_parser"):
        rec.suggestions.append(Suggestion(
            "tool_call_parser", tool,
            f"the template emits the tokens the {tool} parser reads; without it a tool call "
            "comes back as ordinary text in the reply"))
        rec.suggestions.append(Suggestion(
            "enable_auto_tool_choice", True,
            "the parser is ignored unless this is on, and vLLM does not say so"))

    reasoning = ""
    if "<|channel>" in markers and profile.model_type == "gemma4":
        reasoning = "gemma4"
    elif {"<think>", "</think>"} <= markers and profile.model_type.startswith("qwen3"):
        reasoning = "qwen3"

    if reasoning and not args.get("reasoning_parser"):
        rec.suggestions.append(Suggestion(
            "reasoning_parser", reasoning,
            "separates the model's thinking from its answer; without it the reasoning arrives "
            "as part of the reply"))
    elif {"<think>", "</think>"} <= markers and not args.get("reasoning_parser"):
        rec.findings.append(Finding("warn", (
            "The template emits <think>/</think>, which five registered parsers claim and "
            "handle differently. Pick one under Reasoning rather than guessing here.")))

    if "if not enable_thinking" in markers:
        rec.findings.append(Finding("warn", (
            "This template hides the model's reasoning unless the caller passes "
            'enable_thinking. Set --default-chat-template-kwargs to \'{"enable_thinking": '
            'true}\' to turn it on for every request.')))


def _serving(rec: Recommendation, profile: Profile) -> None:
    if profile.runner == "pooling":
        rec.findings.append(Finding("warn", (
            f"This will serve embeddings, not chat: {profile.runner_reason}. /v1/embeddings "
            "works; /v1/chat/completions is never registered, so the route does not exist and "
            "the Playground cannot talk to it.")))
        return

    if not profile.chat_template:
        rec.findings.append(Finding("warn", (
            "The repo ships no chat template. The server starts and /v1/completions works, but "
            "every /v1/chat/completions request fails until a template is supplied.")))


def _existing(rec: Recommendation, args: dict[str, Any]) -> None:
    """Flags already set that this machine makes worse rather than better."""
    try:
        offload = float(args.get("cpu_offload_gb") or 0)
    except (TypeError, ValueError):
        offload = 0.0
    if offload > 0:
        rec.findings.append(Finding("warn", (
            "--cpu-offload-gb frees nothing here: GPU memory is host memory on this machine, so "
            "the offloaded weights sit in the same pool, pinned and unreclaimable, and every "
            "forward pass reads them back over the same bus.")))

    for warning in vllm_spec.cross_flag_warnings(args):
        rec.findings.append(Finding("warn", warning))


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
