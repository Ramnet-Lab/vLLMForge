"""A llama.cpp configuration that starts on the first try.

The sibling of `app/recommend.py`, and it exists for the same reason: the form
offers a hundred flags, two of them decide whether the engine comes up, and
getting those two wrong costs minutes of loading followed by an abort. It is a
separate module rather than a branch because almost none of the vLLM advisor
transfers — its three principles invert here.

* **vLLM: prefer a flag that lets the engine decide.** `--max-model-len auto`
  makes vLLM fit the context to whatever cache is left after it has profiled the
  model, and no offline arithmetic beats a measurement. llama.cpp has the same
  instinct — `-ngl auto` with `--fit on` measures and fits — but it aims to
  leave 1 GiB free per device, where this host wants tens. So on a shared box
  the offline arithmetic that vLLM's advisor avoids is exactly what is needed,
  and the numbers are computable: everything except the context length is in
  the GGUF's own header.

* **vLLM: set as little as possible.** Still true, and there are only two flags
  worth setting. But `--ctx-size` cannot be left alone the way `--max-model-len`
  can: unset, llama.cpp takes the model's *trained* length, which on a modern
  checkpoint is 128k or more — several times the weights in KV cache, for a
  context nobody asked for.

* **vLLM: say what cannot be fixed.** The same, with the sign flipped. GGUF is
  what this engine reads; safetensors is what it cannot, and a model that is
  only safetensors is a vLLM model no flag rescues here.

Every number below comes from `app/gguf.py` reading the file, and the estimate
it feeds is the same one `app/engines/llamacpp.py` prices a launch with — so the
advice and the verdict cannot disagree.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app import gguf, llamacpp_spec, safety
from app.engines.llamacpp import ENGINE as LLAMACPP
from app.engines.llamacpp import host_path
from app.recommend import Finding, Recommendation, Suggestion, _gib

# What a first-try configuration plans for. Long enough to be a working chat
# server, short enough that a model does not spend its whole share of the box on
# a context nobody sends. Raising it is one field.
TARGET_CONTEXT = 16384

# Contexts worth suggesting, in preference order. Powers of two because that is
# what everything else in this space is quantised to, and an operator reading
# "12288" wonders where it came from.
CONTEXT_STEPS = (65536, 32768, 16384, 8192, 4096, 2048)

# Leave this much of what is free unspent. llama.cpp measures its real buffers
# at startup and they are never exactly the estimate; a recommendation that
# lands on the ceiling is refused by the very check that produced it.
MARGIN = 0.10


async def build(model: str, node: str = "", args: dict[str, Any] | None = None,
                server_id: int | None = None) -> Recommendation:
    """Which of llama.cpp's flags to set for this .gguf on this machine."""
    rec = Recommendation(model=model, node=node, engine="llamacpp")
    args = args or {}

    reference = str(model or "").strip()
    if not reference:
        rec.headline = "Choose a model."
        return rec

    if not llamacpp_spec.looks_like_path(reference):
        # A `-hf` reference. llama.cpp will fetch it itself, and nothing can be
        # sized until it has — which is a real answer, not a failure.
        rec.level = "warn"
        rec.headline = "llama.cpp will download this itself."
        rec.findings.append(Finding("warn", (
            f"'{reference}' is a Hub reference, so llama-server resolves and pulls it on the "
            "first start. Nothing about its size is knowable until then, which means the memory "
            "guard cannot vouch for this launch either. Pull it from the Models tab and pick the "
            ".gguf file to get real numbers.")))
        return rec

    path = host_path(reference)
    header = await asyncio.to_thread(gguf.read_cached, path) if path else None
    if header is None:
        rec.ok = False
        rec.level = "block"
        rec.headline = "Not a readable GGUF on this machine."
        rec.findings.append(Finding("block", (
            f"'{reference}' could not be opened and read as a GGUF from here. If the download is "
            "still running, wait for it; if this is a safetensors model, it is a vLLM model and "
            "there is no llama.cpp flag that changes that.")))
        rec.engine_hint = "vllm"
        return rec

    rec.profile = _profile(header, reference)

    replacing = None
    if server_id is not None:
        from app import servers as server_service

        existing = await asyncio.to_thread(server_service.get_server, server_id)
        if existing is not None:
            replacing = server_service.container_name(existing)

    from app import nodes as node_registry

    target = node_registry.by_name(node) if node and node != node_registry.LOCAL else None
    budget = await safety.current_budget(exclude=replacing, node=target)

    # What this launch may spend: the budget's own ceiling, plus whatever the
    # container being replaced is about to release, less a margin for the
    # difference between this arithmetic and llama.cpp's own measurement.
    spendable = int(min(budget.free_bytes_to_commit,
                        budget.available_after_replacement) * (1 - MARGIN))
    if spendable <= 0:
        rec.ok = False
        rec.level = "block"
        rec.headline = "Nothing is free on this node."
        rec.findings.append(Finding("block", (
            f"{_gib(budget.occupied_bytes)} of {_gib(budget.total_bytes)} is already spoken for. "
            "Stop something on the Overview tab first.")))
        return rec

    _sizing(rec, header, args, spendable)
    _quality(rec, header, args)
    return rec


def _profile(header: gguf.Header, reference: str) -> dict[str, Any]:
    """The facts, in the shape the Serve page's profile card already renders.

    Deliberately filling the same keys `app/model_profile.py` fills rather than
    inventing a second shape: the card is one renderer, and a GGUF is not a
    different kind of model, only a different place to read it from.
    """
    return {
        "found": True,
        "reference": reference,
        "source": "gguf",
        "architecture": header.architecture,
        "architectures": [header.architecture] if header.architecture else [],
        "num_hidden_layers": header.block_count,
        "num_key_value_heads": header.head_count_kv,
        "num_attention_heads": header.head_count,
        "hidden_size": header.embedding_length,
        "max_position_embeddings": header.context_length,
        "weight_bytes": header.file_bytes,
        "disk_bytes": header.file_bytes,
        "quantization": header.quant,
        "num_experts": header.expert_count,
        "has_gguf": True,
        "has_safetensors": False,
        "has_chat_template": header.chat_template,
        "is_adapter": False,
        # `supported` here means "llama.cpp can load it", and the honest answer
        # from a header alone is "it is a GGUF and this build reads GGUF".
        # Claiming knowledge of ggml's architecture list would be a guess, and
        # app/model_profile.py's own precedent is that an unknown answer must
        # not read as a refusal.
        "supported": None,
        "notes": [f"read from the GGUF header of {header.path.split('/')[-1]}"],
    }


def _sizing(rec: Recommendation, header: gguf.Header, args: dict, spendable: int) -> None:
    """The two flags that decide whether it starts, and by how much it fits."""
    layers = (header.block_count or 0) + 1
    type_k = str(args.get("cache_type_k") or gguf.DEFAULT_CACHE_TYPE)
    type_v = str(args.get("cache_type_v") or gguf.DEFAULT_CACHE_TYPE)

    def cost(ngl: int, ctx: int) -> int:
        return LLAMACPP.footprint_bytes(
            {"_gguf": header, "n_gpu_layers": str(ngl), "ctx_size": str(ctx),
             "cache_type_k": type_k, "cache_type_v": type_v,
             "ubatch_size": args.get("ubatch_size"), "flash_attn": args.get("flash_attn")},
            0)

    # Everything on the accelerator is the fast case, so it is what is tried
    # first — and the context is stepped down before the layers are, because a
    # layer on the CPU costs speed on every token while a shorter context costs
    # only what it costs.
    for ctx in CONTEXT_STEPS:
        if ctx > (header.context_length or ctx):
            continue
        if cost(layers, ctx) <= spendable:
            rec.suggestions.append(Suggestion("n_gpu_layers", "all", (
                f"every layer fits: {_gib(cost(layers, ctx))} of the {_gib(spendable)} this node "
                "can give a new engine, weights and cache together")))
            _context(rec, header, ctx, cost(layers, ctx), spendable, type_k, type_v)
            return

    # It does not all fit. Find the layer count that does at the smallest
    # context worth serving, and say plainly what that costs.
    floor_ctx = CONTEXT_STEPS[-1]
    fitting = 0
    # A binary search rather than a walk. The layer count comes out of the file's
    # own header, and while app/gguf.py caps it, a linear descent over even a
    # legitimate few hundred layers is a few hundred footprint computations on
    # the request path for an answer that is monotonic in the count.
    low, high = 0, layers
    while low < high:
        middle = (low + high + 1) // 2
        if cost(middle, floor_ctx) <= spendable:
            low = middle
        else:
            high = middle - 1
    fitting = low

    if not fitting:
        rec.ok = False
        rec.level = "block"
        rec.headline = "This model does not fit at any layer count."
        rec.findings.append(Finding("block", (
            f"Even one layer plus a {floor_ctx}-token cache needs more than the "
            f"{_gib(spendable)} free here. Stop something, or use a smaller quantisation — "
            f"this file is {_gib(header.file_bytes)} at {header.quant or 'its quantisation'}.")))
        return

    # With the layer count settled, spend what is left on context.
    best = floor_ctx
    for ctx in CONTEXT_STEPS:
        if ctx > (header.context_length or ctx):
            continue
        if cost(fitting, ctx) <= spendable:
            best = ctx
            break

    rec.level = "warn"
    rec.headline = f"{fitting} of {layers} layers fit; the rest run on the CPU."
    rec.suggestions.append(Suggestion("n_gpu_layers", fitting, (
        f"{_gib(cost(fitting, best))} of the {_gib(spendable)} free. The remaining "
        f"{layers - fitting} layers run on the CPU, which is slower per token in proportion — "
        "this is the thing llama.cpp can do that vLLM cannot")))
    _context(rec, header, best, cost(fitting, best), spendable, type_k, type_v)


def _context(rec: Recommendation, header: gguf.Header, ctx: int, spent: int,
             spendable: int, type_k: str, type_v: str) -> None:
    trained = header.context_length or 0
    # The same cache types the fit decision above used. Quoting the f16 figure
    # while having chosen the context at q8_0 would tell the operator a number
    # nearly twice what their own configuration will take.
    kv = header.kv_bytes(ctx, type_k=type_k, type_v=type_v) or 0
    why = (f"{_gib(kv)} of KV cache. Left unset, llama.cpp takes the model's own trained "
           f"{trained:,} tokens" if trained else "left unset, llama.cpp takes the model's "
           "own trained length")
    if trained and trained > ctx:
        full = header.kv_bytes(trained, type_k=type_k, type_v=type_v) or 0
        why += f", which would be {_gib(full)} of cache for a context nobody asked for"
    rec.suggestions.append(Suggestion("ctx_size", ctx, why + "."))
    if not rec.headline:
        rec.headline = f"Fits whole, with {ctx:,} tokens of context."
    rec.findings.append(Finding("ok", (
        f"Estimated at {_gib(spent)} against {_gib(spendable)} free. llama.cpp measures its "
        "real buffers at startup and prints them in the log — this is arithmetic from the "
        "file's header, and it is deliberately a little high.")))


def _quality(rec: Recommendation, header: gguf.Header, args: dict) -> None:
    """The handful of things worth saying that are not about fitting."""
    if not args.get("flash_attn"):
        rec.suggestions.append(Suggestion("flash_attn", "on", (
            "fused attention. It does not shrink the KV cache, but it stops the compute buffer "
            "growing with context — which is what runs out on a long prompt rather than on "
            "loading — and a quantised V cache requires it")))

    if not header.chat_template:
        rec.findings.append(Finding("warn", (
            "This file carries no chat template, so llama.cpp will fall back to a built-in one "
            "and the model may not answer in the shape it was trained for. Set "
            "--chat-template-file, or use a GGUF built with the template included.")))

    if header.expert_count:
        rec.findings.append(Finding("ok", (
            f"{header.expert_count} experts. If it does not fit, --cpu-moe keeps the expert "
            "weights on the CPU — on a sparse model that is most of the file for a small speed "
            "cost, because only a few experts run per token. The estimate above does not "
            "discount it, so a launch with --cpu-moe has more room than it says.")))

    rec.left_alone = [
        {"dest": "cache_type_k", "why": "the file's own quantisation already made this "
                                        "tradeoff; quantise the cache only when the context "
                                        "you want will not otherwise fit"},
        {"dest": "parallel", "why": "the server picks four slots sharing one context, which is "
                                    "the right shape until you know your concurrency"},
        {"dest": "rope_freq_base", "why": "the header carries it, and overriding it changes how "
                                          "the model reads position"},
        {"dest": "batch_size", "why": "--ubatch-size is the one that sizes the compute buffer; "
                                      "this only decides how much is submitted at once"},
    ]
