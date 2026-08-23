"""llama.cpp, as an engine object.

The interesting half of this file is the memory model, because llama.cpp does
not have the thing every memory decision in this dashboard was built on. vLLM
declares its appetite as a fraction — `--gpu-memory-utilization 0.52` — and
takes that much whatever the model is. llama.cpp declares nothing: what it takes
is arithmetic over the model's own shape and three flags.

    weights = file bytes x (offloaded layers / (layer count + 1))
    kv      = ctx x layers x (n_embd_k_gqa x sizeof(-ctk)
                              + n_embd_v_gqa x sizeof(-ctv))
    compute = a working set sized mostly by --ubatch-size

Everything on the right but the flags comes from the .gguf's own header, which
`app/gguf.py` reads. So this engine's `resolve()` is where the file is opened,
and it exists as a separate async step precisely so `footprint_bytes()` can stay
synchronous for the memory watchdog's per-container ranking loop.

Two judgement calls are worth stating outright, because both are choices to be
wrong in the safe direction rather than facts:

* **An unset `-ngl` is priced as the whole model.** Recent llama.cpp defaults it
  to `auto` and fits itself to the device, leaving 1 GiB free per device. That
  is a smaller reserve than this dashboard's (32 GiB on a unified box), so our
  guard has to run first and cannot assume the engine will be modest. It is the
  same asymmetry `app/images.py` applies to base-image detection: only positive
  proof — an explicit layer count — moves off the pessimistic answer.

* **`--cpu-moe` is not discounted.** It keeps a sparse model's expert weights on
  the CPU, which really is most of the file, but how much cannot be known
  without walking the tensor table. The estimate is therefore high for those
  models, which refuses a launch that would have fit — recoverable, from the
  "start anyway" button — where the other direction is a frozen machine.

When the header cannot be read at all — a `-hf` reference that has not been
downloaded yet, a half-pulled file, a path this process cannot see — the
footprint is 0 and `_sizing` says why. That is not a claim the launch is free:
`safety` turns an unsized llama.cpp launch into a warning that says the guard
cannot vouch for it, and on the local node `measured_gpu_bytes` still sees the
container once it is running.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app import gguf, llamacpp_spec
from app.engines import as_int, flag_value

GIB = 1024 ** 3

# What a llama-server argv starts with, in every spelling that reaches a
# container: the binary the images this repo builds put on PATH, and the
# absolute path the upstream ggml images use as their ENTRYPOINT.
#
# llama.cpp's historical `./server` is deliberately absent. `/app/server` is the
# most common entrypoint in the containerised world — it is what a Go or Node
# service is built to — and claiming those would put an operator's own
# applications on the memory watchdog's kill list.
_BINARY = "llama-server"

# The same name as a word inside a whole shell line, for the reason vLLM needs
# one: a container launched as `bash -lc "... && exec llama-server ..."` is one
# opaque token, and a container that weighs nothing in the budget is how a
# second launch is admitted on memory that is already spoken for.
_LLAMA_SERVER = re.compile(r"(?:^|[\s;&|/])llama-server(?:\s|$)")

MODEL_FLAG = re.compile(r"(?:-m|--model)(?:=(\S+))?$")
HF_FLAG = re.compile(r"(?:-hf|-hfr|--hf-repo)(?:=(\S+))?$")
HF_FILE_FLAG = re.compile(r"(?:-hff|--hf-file)(?:=(\S+))?$")
NGL_FLAG = re.compile(r"(?:-ngl|--gpu-layers|--n-gpu-layers)(?:=(\S+))?$")
CTX_FLAG = re.compile(r"(?:-c|--ctx-size)(?:=(\S+))?$")
CTK_FLAG = re.compile(r"(?:-ctk|--cache-type-k)(?:=(\S+))?$")
CTV_FLAG = re.compile(r"(?:-ctv|--cache-type-v)(?:=(\S+))?$")
UBATCH_FLAG = re.compile(r"(?:-ub|--ubatch-size)(?:=(\S+))?$")
PARALLEL_FLAG = re.compile(r"(?:-np|--parallel)(?:=(\S+))?$")
FA_FLAG = re.compile(r"(?:-fa|--flash-attn)(?:=(\S+))?$")

# The mounts every engine container gets. A model reference is expressed the way
# the container sees it, and pricing has to open the file the way this process
# sees it, so the two are folded back here.
CONTAINER_MOUNTS = ("/hf", "/outputs")

# llama.cpp measures its own compute buffer by building the worst-case graph
# twice and asking the allocator what it needed — no closed form exists, and any
# offline figure is a guess. This is the guess, and the concurrency and batch
# factors below absorb its error. Its shape is taken from what actually drives
# the measurement: the physical batch size, and (without fused attention) the
# attention score matrix, which is the term that grows with context.
COMPUTE_FLOOR_BYTES = 1 * GIB
DEFAULT_UBATCH = 512
# A ceiling on the no-flash-attention term. Past a few GiB the estimate stops
# discriminating between configurations and starts refusing all of them.
MAX_SCORE_BYTES = 4 * GIB

INTERESTING_METRICS = (
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
    "llamacpp:n_busy_slots_per_decode",
    "llamacpp:prompt_tokens_seconds",
    "llamacpp:predicted_tokens_seconds",
    "llamacpp:prompt_tokens_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:n_decode_total",
    "llamacpp:n_tokens_max",
    "llamacpp:spec_decode_num_draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total",
)


def host_path(reference: str) -> Path | None:
    """Where this process can open a path a container was given.

    Model references are stored as the container sees them — `/hf/hub/…` — and a
    footprint has to stat the real file. Only the two known mounts are folded;
    anything else is returned as-is and simply may not exist here, which is the
    honest answer for a container someone else launched with their own mounts.
    """
    import os

    from app.config import settings

    text = str(reference or "").strip()
    if not text:
        return None
    for mount, root in (("/hf", settings.hf_cache), ("/outputs", settings.output_dir)):
        if text != mount and not text.startswith(mount + "/"):
            continue
        base = Path(root).resolve()
        # Normalised and then checked for containment, because this path is
        # opened and stat'd by the dashboard process and it arrives from a form
        # field. `/hf/../../etc/passwd` folds to something outside the mount, and
        # a prefix test alone would happily hand it over. `..` is not a legal
        # part of a model reference, so refusing is not a restriction anybody
        # runs into. Symlinks are deliberately NOT resolved: the cache is a tree
        # of them by design, and following one is how a snapshot path becomes an
        # unreadable blob hash.
        candidate = Path(os.path.normpath(base / text[len(mount):].lstrip("/")))
        if candidate != base and base not in candidate.parents:
            return None
        return candidate
    # An absolute path with no mount prefix belongs to a container somebody else
    # launched with mounts this process knows nothing about. It is returned so a
    # foreign engine can be priced when the path happens to be readable here —
    # but only when it lands inside a directory this dashboard already manages,
    # so a stray `-m /root/.ssh/id_rsa` is not opened on its say-so.
    if not text.startswith("/"):
        return None
    absolute = Path(os.path.normpath(text))
    for root in (settings.hf_cache, settings.output_dir):
        base = Path(root).resolve()
        if absolute == base or base in absolute.parents:
            return absolute
    return None


class LlamaCppEngine:
    name = "llamacpp"
    label = "llama.cpp"
    binary = "llama-server"
    # docker/llamacpp.Dockerfile installs the same entrypoint shim the vLLM image
    # uses, which execs "$@" — so the full argv passed as the container command
    # runs as written, and Config.Cmd[0] stays `llama-server`, which is what lets
    # our own containers be recognised without falling back to the entrypoint.
    entrypoint = None
    gpu = True
    # llama.cpp splits across machines with `--rpc` and rpc-server processes: no
    # world size, no rendezvous, no NCCL. app/cluster.py is torch.distributed
    # pipeline parallel end to end, so pooling is refused rather than faked.
    supports_pooling = False
    served_name_dest = "alias"
    interesting_metrics = INTERESTING_METRICS

    @property
    def default_image(self) -> str:
        from app.config import settings

        return settings.llamacpp_image

    # --- schema -----------------------------------------------------------

    def ui_model(self) -> dict[str, Any]:
        return llamacpp_spec.ui_model()

    def validate(self, params: dict[str, Any]) -> list[str]:
        return llamacpp_spec.validate(params)

    def managed_flags(self) -> frozenset[str]:
        return frozenset(llamacpp_spec.MANAGED_FLAGS)

    def path_kinds(self) -> dict[str, str]:
        return dict(llamacpp_spec.PATH_KINDS)

    # --- command assembly -------------------------------------------------

    def build_argv(self, model: str, params: dict[str, Any], *,
                   host: str = "0.0.0.0", port: int | None = None) -> list[str]:
        return llamacpp_spec.build_argv(model, params, host=host, port=port)

    def env_overlay(self, params: dict[str, Any]) -> dict[str, str]:
        # llama.cpp keeps what it downloads through `-hf` in LLAMA_CACHE, and
        # left unset that is a directory inside the container that dies with it —
        # so the same weights are pulled again on every restart. Pointing it at
        # the mounted cache makes a llama.cpp download as durable as a vLLM one.
        return {"LLAMA_CACHE": "/hf/llamacpp"}

    # --- memory -----------------------------------------------------------

    def argv_of(self, command: list[str] | None) -> list[str]:
        """Tokens, unwrapping a shell if one is in the way."""
        if not command:
            return []
        for token in command:
            if len(token.split()) == 1 or not _LLAMA_SERVER.search(token):
                continue
            try:
                import shlex

                parts = shlex.split(token)
            except ValueError:
                continue
            for index, part in enumerate(parts):
                if Path(part).name == _BINARY:
                    return parts[index:]
        return list(command)

    def matches(self, command: list[str] | None) -> bool:
        """Whether this argv is a llama-server.

        Anchored on the program name only, and only in the first two tokens, for
        the same reason vLLM's is: a container running something else that
        happens to mention llama.cpp in a later argument is not this engine, and
        mistaking it for one prices it into the memory budget and — once the
        watchdog can act — kills it.

        The name has to be `llama-server`. llama.cpp's historical `./server`
        binary is deliberately NOT accepted, and the asymmetry is on purpose:
        `/app/server` is the single most common entrypoint in the containerised
        world — it is what a Go or Node service is built to — and a recogniser
        that claimed those would hand the watchdog a list of an operator's own
        applications to kill. Missing a five-year-old binary name costs a budget
        line; the other direction costs somebody's database.
        """
        argv = self.argv_of(command)
        return any(Path(token).name == _BINARY for token in argv[:2])

    def command_params(self, command: list[str] | None) -> dict[str, Any]:
        """A running container's sizing flags, shaped like stored args."""
        argv = self.argv_of(command)
        params: dict[str, Any] = {}

        model = flag_value(argv, MODEL_FLAG) or flag_value(argv, HF_FLAG)
        if model:
            params["model"] = model
        ngl = flag_value(argv, NGL_FLAG)
        if ngl is not None:
            params["n_gpu_layers"] = ngl
        for key, pattern in (("ctx_size", CTX_FLAG), ("cache_type_k", CTK_FLAG),
                             ("cache_type_v", CTV_FLAG), ("ubatch_size", UBATCH_FLAG),
                             ("parallel", PARALLEL_FLAG), ("flash_attn", FA_FLAG)):
            value = flag_value(argv, pattern)
            if value is not None:
                params[key] = value
        return params

    async def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        """Open the .gguf and fold its header into the params.

        This is the one member that does I/O, and it is separate from
        `footprint_bytes` for that reason: the watchdog prices every container on
        every tick and the launch guard prices one while assembling a message,
        and neither may block on a filesystem read.
        """
        import asyncio

        params = dict(params or {})
        reference = str(params.get("model") or "").strip()
        if not reference:
            params["_sizing"] = "no model in this command, so its size is unknown"
            return params

        path = host_path(reference)
        if path is None:
            params["_sizing"] = (
                f"{reference} is a Hub reference llama.cpp downloads itself, so its "
                "size is not known until it has been pulled")
            return params

        header = await asyncio.to_thread(gguf.read_cached, path)
        if header is None:
            params["_sizing"] = (
                f"{reference} could not be read as a GGUF from this machine, so its "
                "footprint is an unknown")
            return params

        params["_gguf"] = header
        params.pop("_sizing", None)
        return params

    def footprint_bytes(self, params: dict[str, Any], total_bytes: int) -> int:
        """A floor on the accelerator memory these parameters will take.

        `total_bytes` is unused: unlike vLLM, nothing here is a fraction of the
        machine. It stays in the signature because it is the shared contract
        every engine's pricer is called through.
        """
        header = (params or {}).get("_gguf")
        if not isinstance(header, gguf.Header):
            # Unsized. Not "free" — see the module docstring; safety turns this
            # into a warning rather than an approval.
            return 0

        ngl = llamacpp_spec.n_gpu_layers(params)
        weights = int(header.file_bytes * header.offload_fraction(ngl))

        ctx = llamacpp_spec.ctx_size(params) or header.context_length or 0
        kv = header.kv_bytes(
            ctx,
            type_k=str(params.get("cache_type_k") or gguf.DEFAULT_CACHE_TYPE),
            type_v=str(params.get("cache_type_v") or gguf.DEFAULT_CACHE_TYPE),
        )
        # NOT scaled by the offload fraction, and this is the one place the
        # discrete/unified distinction leaks into the estimate. Only the
        # offloaded layers keep their cache in the framebuffer — but on the
        # machine this dashboard is most careful about, GPU memory IS host
        # memory, so the cache of a CPU-resident layer lands in the very pool
        # being guarded. Discounting it priced `-ngl 0` on a 70 GiB model at the
        # compute floor alone and answered "Fits" for a launch that takes the
        # machine. On a discrete GPU this over-charges a partial offload, which
        # is the recoverable direction.
        if kv is None and ctx:
            # The head geometry could not be read — a hybrid architecture
            # declaring per-layer head counts as an array longer than the header
            # reader keeps, most often. Folding that to zero would price the
            # model weights-only and report a confident "Fits" while the largest
            # single term went uncounted, so it is reported as unsized instead.
            params["_sizing"] = (
                "this model's attention geometry could not be read from its header, so the "
                "KV cache — usually the largest term after the weights — cannot be sized")
            return 0
        kv = kv or 0

        compute = self._compute_bytes(params, header, ctx)
        extra = self._companions(params)
        return weights + kv + compute + extra

    def _companions(self, params: dict[str, Any]) -> int:
        """The other GGUFs a launch loads into the same pool.

        A draft model for speculative decoding and a multimodal projector are
        separate files, each fully resident, and neither appears anywhere in the
        main model's header. Left out, a config with a 2 GiB draft model was
        priced identically to one without.
        """
        total = 0
        for key in ("spec_draft_model", "mmproj"):
            reference = str((params or {}).get(key) or "").strip()
            if not reference:
                continue
            path = host_path(reference)
            if path is None:
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _compute_bytes(self, params: dict[str, Any], header: gguf.Header, ctx: int) -> int:
        """The activation working set, estimated rather than derived.

        llama.cpp measures this by reserving the real graph twice and reading the
        allocator back, so no offline number matches it. What is modelled is the
        two things that actually move it: the physical batch, and — with fused
        attention off — the attention score matrix, which is the term that grows
        with context and is why long-context launches fail during prompt
        processing rather than during loading.
        """
        ubatch = as_int(params.get("ubatch_size")) or DEFAULT_UBATCH
        compute = int(COMPUTE_FLOOR_BYTES * max(1.0, ubatch / DEFAULT_UBATCH))

        # llama.cpp's own values here are `on`, `off` and `auto`; anything else
        # is a typo, and treating a typo as "on" would under-charge. Only an
        # explicit `off` disables fused attention — `auto` turns it on wherever
        # the backend supports it, which every CUDA build does.
        if str(params.get("flash_attn") or "auto").strip().lower() in ("off", "0", "false") \
                and ctx:
            heads = header.head_count or 1
            # The unfused path materialises an [n_kv x n_tokens] score matrix per
            # head. ggml reuses the buffer across layers, so this is one graph
            # node's worth rather than the whole model's — but it is the term
            # that grows with context, and it is why a long-context launch fails
            # during prompt processing rather than during loading.
            scores = ctx * ubatch * heads * 4
            compute += min(scores, MAX_SCORE_BYTES)
        return compute

    def declared_util(self, params: dict[str, Any]) -> float | None:
        """Always None. llama.cpp declares no fraction, and reporting bytes/total
        here would make two engines' Util columns look summable when they are
        not — an operator will add them, and the answer would be wrong."""
        return None

    def implicit_util(self) -> float | None:
        return None

    def notes(self, params: dict[str, Any], *, implicit: bool) -> list[str]:
        notes: list[str] = []
        sizing = (params or {}).get("_sizing")
        if sizing:
            notes.append(str(sizing))
            return notes

        header = (params or {}).get("_gguf")
        if isinstance(header, gguf.Header):
            ngl = llamacpp_spec.n_gpu_layers(params)
            total = (header.block_count or 0) + 1
            if ngl is None:
                notes.append(
                    "no --n-gpu-layers set, so this is priced as the whole model on the "
                    "accelerator — llama.cpp would fit itself to a 1 GiB margin, which "
                    "is smaller than this host's reserve")
            elif header.block_count is not None:
                notes.append(f"{min(ngl, total)} of {total} layers offloaded")
            ctx = llamacpp_spec.ctx_size(params) or header.context_length
            if ctx:
                k = str(params.get("cache_type_k") or gguf.DEFAULT_CACHE_TYPE)
                v = str(params.get("cache_type_v") or gguf.DEFAULT_CACHE_TYPE)
                cache = f"{k}" if k == v else f"{k}/{v}"
                notes.append(f"{ctx} tokens of context at {cache}")
        if params.get("cpu_moe") or params.get("n_cpu_moe"):
            notes.append(
                "--cpu-moe keeps the expert weights on the CPU, so the real footprint "
                "is smaller than this — how much smaller is not readable from the header")
        return notes

    # --- discovery and status ---------------------------------------------

    def model_from_argv(self, argv: list[str] | None) -> str:
        """The model a llama-server is serving. A flag, never a positional."""
        tokens = self.argv_of(argv)
        model = flag_value(tokens, MODEL_FLAG)
        if model:
            return model
        repo = flag_value(tokens, HF_FLAG)
        if not repo:
            return ""
        found = flag_value(tokens, HF_FILE_FLAG)
        return f"{repo}/{found}" if found else repo

    def is_loading(self, *, reachable: bool, healthy: bool, **_extra: Any) -> bool:
        """The exact inverse of vLLM's rule.

        llama-server binds its socket before it loads anything and answers
        /health with 503 "Loading model" until the weights are in. So reachable
        but not healthy is what loading looks like, and treating an unreachable
        port as loading — vLLM's rule — would leave a llama.cpp server that is
        genuinely coming up labelled "running" and "unhealthy", which reads as
        broken for the several minutes a large model takes.
        """
        return bool(reachable) and not bool(healthy)


ENGINE = LlamaCppEngine()
