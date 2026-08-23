"""What a .gguf file says about itself, read from its own header.

A HuggingFace repo puts a model's shape in config.json, beside the weights,
where app/model_profile.py reads it. A GGUF puts it *inside* the weights: the
first few kilobytes of the file are a typed key/value table carrying the
architecture, the layer count, the attention head geometry and the training
context length. Nothing else in this dashboard opens a weights file, so nothing
knew any of it — a GGUF profiled as "not found" for a model sitting on disk.

That table is the whole reason this module exists, because llama.cpp's memory
footprint is arithmetic rather than a declared fraction:

    weights ≈ file bytes x (offloaded layers / (block_count + 1))
    kv      = ctx x block_count x (n_embd_k_gqa x sizeof(ctk)
                                   + n_embd_v_gqa x sizeof(ctv))

Every term on the right except `ctx` and the cache dtypes comes from here.

The format (ggml-org/ggml docs/gguf.md, version 3):

    magic          uint32   'GGUF' little-endian, 0x46554747
    version        uint32
    tensor_count   uint64
    kv_count       uint64
    kv_count x { key: uint64 length + UTF-8 bytes; type: uint32; value }

The tensor-info block follows and is deliberately never read: it is one record
per tensor and answers nothing this file is asked. Reading stops at the end of
the key/value table.

Three bounds are load-bearing, because this parses a length-prefixed binary
format that a truncated download produces a perfectly plausible-looking prefix
of. A corrupt uint64 length is indistinguishable from a real one until it is
acted on, so every count and every length is checked against a ceiling before a
single byte is allocated, and the whole read is capped. Failure is always None —
a model whose header cannot be read is a model whose footprint is unknown, and
app/engines/llamacpp.py says so rather than inventing a number.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

log = logging.getLogger("llmd.gguf")

MAGIC = 0x46554747

# gguf_metadata_value_type. Ordinals are the wire format and cannot be reordered.
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

# (struct format, byte width) for every fixed-width type. STRING and ARRAY are
# absent on purpose: they are variable-length and are handled by name.
_FIXED: dict[int, tuple[str, int]] = {
    U8: ("<B", 1), I8: ("<b", 1),
    U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4),
    F32: ("<f", 4),
    BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8),
    F64: ("<d", 8),
}

# gguf-py's own defensive ceilings, which the C reader also applies. They are not
# part of the spec — they are what stops a garbage length from being believed.
MAX_STRING = 1 << 30
MAX_ARRAY = 1 << 30
# The spec's own limit on a key name. A "key" longer than this is not one.
MAX_KEY = 65535
# Nothing legitimate carries this many keys; a header that claims to is corrupt.
MAX_KV = 8192
# The header of a real model is a few hundred KB at most, dominated by the
# tokenizer's token list. Past this, stop rather than walk a corrupt file to its
# end — the answer is not in there.
MAX_HEADER_BYTES = 64 * 1024 * 1024

# A string this long is a chat template or a tokenizer entry. Neither is needed
# for a footprint, and materialising a 1 MB Jinja template on every budget survey
# is waste, so long strings are skipped rather than decoded.
MAX_KEPT_STRING = 64 * 1024

# Element counts above this are token lists and merge tables. They are skipped
# wholesale — but note skipping still costs a walk when the elements are
# variable-length, because there is no length in front of the array as a whole.
MAX_KEPT_ARRAY = 64

# A ceiling on any geometry read out of the header. Nothing real has a million
# layers or a billion attention heads, and an absurd one is not merely a wrong
# estimate: it is a loop bound in the recommender and a multiplication in the
# pricer, both of which run on the request path.
MAX_DIMENSION = 1 << 24

# Layers get their own, much tighter, ceiling. The number is a loop bound in
# app/recommend_llamacpp.py, which walks candidate layer counts downward — so an
# absurd one is not a wrong estimate but seconds of CPU on the event loop. The
# largest real model is a few hundred blocks deep.
MAX_LAYERS = 4096

# Arrays may nest, so a crafted file can nest them as deeply as it likes and
# walk this parser straight into a RecursionError — which is not a GGUFError and
# would escape `read_cached`'s net into a budget survey. Nothing real nests more
# than one deep.
MAX_ARRAY_DEPTH = 8


# The ggml block layout, as (elements per block, bytes per block), for the types
# `-ctk`/`-ctv` will accept. llama.cpp sizes a cache row as
# `type_size x elements / block_size`, so these two numbers are the whole answer.
# Read out of ggml/src/ggml-common.h; llama.cpp's own GGML_QUANT_SIZES agrees.
CACHE_TYPES: dict[str, tuple[int, int]] = {
    "f32": (1, 4),
    "f16": (1, 2),
    "bf16": (1, 2),
    "q8_0": (32, 34),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "iq4_nl": (32, 18),
}

DEFAULT_CACHE_TYPE = "f16"


def cache_bytes_per_element(name: str | None) -> float:
    """Bytes one KV element costs at this `-ctk`/`-ctv` setting.

    Unknown names fall back to f16 rather than to zero: a cache type this build
    does not recognise is far more likely to be a newer quant than to be free,
    and pricing it at nothing is the direction that freezes a machine.
    """
    block, size = CACHE_TYPES.get(str(name or DEFAULT_CACHE_TYPE).lower(),
                                  CACHE_TYPES[DEFAULT_CACHE_TYPE])
    return size / block


# llama_ftype, from include/llama.h. This is NOT the ggml_type enum — it names
# the mixed-precision *scheme* a whole file was quantised with, which is what an
# operator recognises ("Q4_K_M"), and several of its values have no single-tensor
# equivalent at all.
FILE_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}


class GGUFError(ValueError):
    """The file is not a GGUF, or its header does not survive its own bounds."""


@dataclass
class Header:
    """The handful of facts a launch decision actually needs.

    Everything is optional because a GGUF is allowed to omit any of it, and a
    partial answer is still worth having: a file whose `block_count` is readable
    but whose head geometry is not can still be priced for weights, and saying
    "the KV term is unknown" beats reporting a total that quietly excludes it.
    """

    path: str = ""
    file_bytes: int = 0
    version: int = 0
    tensor_count: int = 0

    architecture: str = ""
    name: str = ""
    size_label: str = ""
    quant: str = ""                       # 'Q4_K_M' — from general.file_type

    block_count: int | None = None        # transformer layers
    context_length: int | None = None     # what the model was trained for
    embedding_length: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    key_length: int | None = None         # per-head K dim, when declared
    value_length: int | None = None
    expert_count: int | None = None

    chat_template: bool = False
    keys: dict[str, Any] = field(default_factory=dict)

    # --- derived geometry -------------------------------------------------

    @property
    def n_embd_head_k(self) -> int | None:
        """Per-head K width. llama-hparams.cpp falls back to n_embd/n_head when
        the file does not declare it, which is the pre-GQA-era shape."""
        if self.key_length:
            return self.key_length
        if self.embedding_length and self.head_count:
            return self.embedding_length // self.head_count
        return None

    @property
    def n_embd_head_v(self) -> int | None:
        if self.value_length:
            return self.value_length
        return self.n_embd_head_k

    @property
    def n_head_kv(self) -> int | None:
        """KV heads. Absent means no GQA — one KV head per attention head."""
        return self.head_count_kv or self.head_count

    @property
    def n_embd_k_gqa(self) -> int | None:
        head, count = self.n_embd_head_k, self.n_head_kv
        return head * count if head and count else None

    @property
    def n_embd_v_gqa(self) -> int | None:
        head, count = self.n_embd_head_v, self.n_head_kv
        return head * count if head and count else None

    def kv_bytes(self, ctx: int, *, type_k: str = DEFAULT_CACHE_TYPE,
                 type_v: str = DEFAULT_CACHE_TYPE) -> int | None:
        """Bytes the KV cache takes at this context length.

        llama.cpp allocates, per attended layer, one K tensor of
        (n_embd_k_gqa x kv_size) and one V of (n_embd_v_gqa x kv_size), each
        sized through ggml_row_size so a quantised cache type costs its block
        size rather than its element count.

        `-np`/`--kv-unified` are deliberately absent from the arithmetic. They
        decide whether the context is one shared pool or is pre-partitioned into
        equal slices, and in both cases n_ctx_seq x n_stream comes back to n_ctx
        (llama-context.cpp) up to a 256-token pad. The total is the same either
        way, which is the only thing a budget is asking.
        """
        if not ctx or self.block_count is None:
            return None
        k, v = self.n_embd_k_gqa, self.n_embd_v_gqa
        if not k or not v:
            return None
        per_token = (k * cache_bytes_per_element(type_k)
                     + v * cache_bytes_per_element(type_v))
        return int(ctx * self.block_count * per_token)

    def offload_fraction(self, n_gpu_layers: int | None) -> float:
        """How much of the file lands in accelerator memory at this `-ngl`.

        llama.cpp treats the model as block_count + 1 virtual layers — the
        blocks, plus one standing for the output head — and offloads the LAST N
        of them (llama-model.cpp, `i_gpu_start = max(n_layer_all + 1 - N, 0)`).
        So the divisor is block_count + 1, not block_count, and the extra slot is
        not a rounding detail: on a small model the output tensor is
        vocab x n_embd and is one of the largest tensors in the file.

        The token-embedding tensor stays on the CPU at every setting, which this
        does not model — it makes the estimate slightly high, and high is the
        direction a guard should err in.

        None means the operator did not say. That is priced as everything, and
        the reason is in app/engines/llamacpp.py.
        """
        total = (self.block_count or 0) + 1
        if not self.block_count or total <= 0:
            # No layer count to divide by. Everything, which is the pessimistic
            # answer and the one a guard should give when it cannot tell.
            return 1.0
        if n_gpu_layers is None or n_gpu_layers < 0:
            return 1.0
        return max(0.0, min(1.0, n_gpu_layers / total))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_bytes": self.file_bytes,
            "version": self.version,
            "tensor_count": self.tensor_count,
            "architecture": self.architecture,
            "name": self.name,
            "size_label": self.size_label,
            "quant": self.quant,
            "block_count": self.block_count,
            "context_length": self.context_length,
            "embedding_length": self.embedding_length,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "n_embd_k_gqa": self.n_embd_k_gqa,
            "n_embd_v_gqa": self.n_embd_v_gqa,
            "expert_count": self.expert_count,
            "chat_template": self.chat_template,
        }


# --- the reader ----------------------------------------------------------

class _Cursor:
    """A bounded reader over the header. Every read is checked, so a corrupt
    length fails here rather than in an allocation."""

    def __init__(self, stream: BinaryIO, limit: int = MAX_HEADER_BYTES):
        self.stream = stream
        self.limit = limit
        self.read_bytes = 0

    def take(self, count: int) -> bytes:
        if count < 0 or count > self.limit - self.read_bytes:
            raise GGUFError(f"header wants {count} more bytes than it may have")
        data = self.stream.read(count)
        if len(data) != count:
            raise GGUFError("header ends before it says it does")
        self.read_bytes += count
        return data

    def skip(self, count: int) -> None:
        """Seek past a value rather than read it. Same bound; no allocation."""
        if count < 0 or count > self.limit - self.read_bytes:
            raise GGUFError(f"header wants to skip {count} bytes it may not have")
        self.stream.seek(count, 1)
        self.read_bytes += count

    def scalar(self, kind: int) -> Any:
        fmt, width = _FIXED[kind]
        return struct.unpack(fmt, self.take(width))[0]

    def u32(self) -> int:
        return self.scalar(U32)

    def u64(self) -> int:
        return self.scalar(U64)

    def string(self, *, keep: bool = True, cap: int = MAX_STRING) -> str:
        length = self.u64()
        if length > cap:
            raise GGUFError(f"string of {length} bytes is not credible")
        if not keep and length > MAX_KEPT_STRING:
            # A chat template or a tokenizer entry. Its presence is recorded by
            # the caller; its content is nothing a footprint needs.
            self.skip(length)
            return ""
        return self.take(length).decode("utf-8", errors="replace")


def _read_value(cur: _Cursor, kind: int, *, keep: bool = True, depth: int = 0) -> Any:
    if kind in _FIXED:
        return cur.scalar(kind)
    if kind == STRING:
        return cur.string(keep=keep)
    if kind == ARRAY:
        return _read_array(cur, keep=keep, depth=depth)
    raise GGUFError(f"unknown metadata value type {kind}")


def _read_array(cur: _Cursor, *, keep: bool, depth: int = 0) -> Any:
    if depth >= MAX_ARRAY_DEPTH:
        raise GGUFError(f"metadata arrays nested more than {MAX_ARRAY_DEPTH} deep")
    element = cur.u32()
    count = cur.u64()
    if count > MAX_ARRAY:
        raise GGUFError(f"array of {count} elements is not credible")

    if element in _FIXED:
        # Fixed width: the whole array can be skipped by arithmetic, which is
        # what keeps a 128k-entry table from costing 128k reads.
        _fmt, width = _FIXED[element]
        # A WANTED numeric array is kept however long it is, bounded only by the
        # layer ceiling. Hybrid and sliding-window architectures write
        # attention.head_count_kv as one entry per layer, and a model with more
        # than a handful of layers would otherwise have its head geometry
        # discarded — which is the whole KV term of its footprint.
        if keep and count <= MAX_LAYERS:
            return [cur.scalar(element) for _ in range(count)]
        if not keep or count > MAX_KEPT_ARRAY:
            cur.skip(count * width)
            return None
        return [cur.scalar(element) for _ in range(count)]

    # Variable width — strings, or nested arrays. There is no length in front of
    # the array as a whole, so skipping still means walking it. Elements are not
    # kept while walking, so the cost is the walk and not the memory.
    wanted = keep and count <= MAX_KEPT_ARRAY
    out: list[Any] = []
    for _ in range(count):
        value = _read_value(cur, element, keep=wanted, depth=depth + 1)
        if wanted:
            out.append(value)
    return out if wanted else None


# Keys worth keeping. Everything else is read (or skipped) and discarded, so the
# header of a model with a 150k-token vocabulary costs a walk rather than a
# hundred megabytes. Suffixes are matched against the `{arch}.` prefix the spec
# interpolates, because the architecture is not necessarily the first key.
_GENERAL = (
    "general.architecture", "general.name", "general.file_type",
    "general.size_label", "general.quantization_version", "general.alignment",
)
_PER_ARCH = (
    ".block_count", ".context_length", ".embedding_length",
    ".attention.head_count", ".attention.head_count_kv",
    ".attention.key_length", ".attention.value_length",
    ".expert_count", ".expert_used_count",
    ".rope.freq_base", ".rope.dimension_count",
)


def _wanted(key: str) -> bool:
    return key in _GENERAL or any(key.endswith(suffix) for suffix in _PER_ARCH)


def read_header(path: str | Path, *, limit: int = MAX_HEADER_BYTES) -> Header:
    """Parse one .gguf file's metadata table. Raises GGUFError on anything odd."""
    target = Path(path)
    size = target.stat().st_size
    with target.open("rb") as stream:
        cur = _Cursor(stream, limit=limit)
        if cur.u32() != MAGIC:
            raise GGUFError(f"{target.name} does not start with the GGUF magic")
        version = cur.u32()
        tensor_count = cur.u64()
        kv_count = cur.u64()
        if kv_count > MAX_KV:
            raise GGUFError(f"{kv_count} metadata keys is not credible")

        values: dict[str, Any] = {}
        chat_template = False
        for _ in range(kv_count):
            # keep=False: the spec caps a key at 65535 bytes, and nothing longer
            # than that is a key this reader wants. Reading it with keep=True
            # would let one 64 MiB "key" be materialised as bytes and again as a
            # string before being thrown away.
            key = cur.string(keep=False, cap=MAX_KEY)
            kind = cur.u32()
            keep = _wanted(key)
            value = _read_value(cur, kind, keep=keep)
            if key == "tokenizer.chat_template":
                # Recorded but never held: it is up to a megabyte of Jinja, and
                # the only question asked of it here is whether it is there.
                chat_template = True
            elif keep:
                values[key] = value

    arch = str(values.get("general.architecture") or "")

    def per_arch(suffix: str) -> Any:
        return values.get(f"{arch}{suffix}") if arch else None

    def as_int(value: Any) -> int | None:
        """One non-negative integer, or None. Never raises.

        Every number here came off the wire, and a key declared as F32 may hold
        NaN or infinity — `int()` on either raises ValueError or OverflowError,
        and neither is a GGUFError, so both would escape `read_cached`'s net into
        a budget survey. A negative count is refused for the same reason a NaN
        is: `block_count = -1` parses cleanly as a signed integer and then makes
        `offload_fraction`'s divisor zero inside the synchronous pricer.
        """
        if isinstance(value, list):
            numbers = [n for n in (as_int(v) for v in value) if n is not None]
            # head_count_kv is per-layer on some hybrid architectures. The
            # largest layer is the honest answer for a memory estimate: it is
            # the one that decides whether the cache fits.
            return max(numbers) if numbers else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            number = int(value)
        except (ValueError, OverflowError):
            return None
        return number if 0 <= number <= MAX_DIMENSION else None

    def _layers(value: int | None) -> int | None:
        return value if value is not None and 0 < value <= MAX_LAYERS else None

    ftype = as_int(values.get("general.file_type"))
    return Header(
        path=str(target),
        file_bytes=size,
        version=version,
        tensor_count=tensor_count,
        architecture=arch,
        name=str(values.get("general.name") or ""),
        size_label=str(values.get("general.size_label") or ""),
        quant=FILE_TYPES.get(ftype, "") if ftype is not None else "",
        block_count=_layers(as_int(per_arch(".block_count"))),
        context_length=as_int(per_arch(".context_length")),
        embedding_length=as_int(per_arch(".embedding_length")),
        head_count=as_int(per_arch(".attention.head_count")),
        head_count_kv=as_int(per_arch(".attention.head_count_kv")),
        key_length=as_int(per_arch(".attention.key_length")),
        value_length=as_int(per_arch(".attention.value_length")),
        expert_count=as_int(per_arch(".expert_count")),
        chat_template=chat_template,
        keys=values,
    )


# A read is a few hundred kilobytes of file I/O and a walk of the tokenizer's
# token list, and the budget survey asks for it on every poll. The key carries
# mtime and size so a re-quantised file at the same path is re-read rather than
# answered from a stale entry.
_CACHE: dict[tuple[str, int, int], Header | None] = {}
# Larger than the most files one request will look at (catalog.MAX_GGUF), so a
# single listing does not evict its own earlier entries and the next identical
# request is answered without re-reading anything. A Header is a few dozen small
# fields, so several hundred of them is negligible.
_CACHE_LIMIT = 512


def read_cached(path: str | Path) -> Header | None:
    """`read_header`, memoised on (path, mtime, size). None if it cannot be read.

    Never raises. A file that is missing, truncated, still downloading or simply
    not a GGUF is a footprint this dashboard does not know — which the caller
    reports, rather than guessing a number that would be acted on.
    """
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return None
    key = (str(target), int(stat.st_mtime), stat.st_size)
    if key in _CACHE:
        return _CACHE[key]
    try:
        header: Header | None = read_header(target)
    except (GGUFError, OSError, struct.error, ValueError, OverflowError,
            RecursionError, MemoryError) as exc:
        # Deliberately wide. This parses attacker-shaped input — a length-prefixed
        # binary format arriving over the network — and the contract every caller
        # relies on is that an unreadable model is an unknown footprint, not an
        # exception in the middle of a budget survey. GGUFError covers what this
        # reader anticipates; the rest covers what it did not.
        log.debug("could not read the GGUF header of %s: %s", target, exc)
        header = None
    if len(_CACHE) >= _CACHE_LIMIT:
        # Drop the oldest rather than the lot: clearing wiped the entries a walk
        # in progress was about to reuse, so a listing longer than the cache
        # scored zero hits and re-read every file on every request.
        for stale in list(_CACHE)[:_CACHE_LIMIT // 4]:
            _CACHE.pop(stale, None)
    _CACHE[key] = header
    return header


def is_gguf(path: str | Path) -> bool:
    """Whether this path names a file whose first four bytes are the magic.

    By content, not by suffix: `-m` takes whatever it is given, and a `.gguf`
    that is really an LFS pointer is the common way a repo arrives half-pulled.
    """
    try:
        with Path(path).open("rb") as stream:
            return struct.unpack("<I", stream.read(4))[0] == MAGIC
    except (OSError, struct.error):
        return False
