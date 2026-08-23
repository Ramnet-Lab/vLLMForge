"""Reading a .gguf's own header.

This parses a length-prefixed binary format, and the file it parses arrives over
the network in tens of gigabytes. A truncated download produces a perfectly
plausible prefix, so most of what is worth testing here is what happens when the
numbers in the file are lies.
"""

from __future__ import annotations

import struct

import pytest

from app import gguf

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)


def _s(raw: bytes) -> bytes:
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _s(key.encode()) + struct.pack("<I", kind) + payload


def write_gguf(path, *, keys=None, file_bytes=0, tensor_count=291, magic=gguf.MAGIC,
               kv_count=None, version=3):
    """A GGUF whose header says what the test needs it to say."""
    keys = keys if keys is not None else default_keys()
    body = b"".join(keys)
    head = (struct.pack("<I", magic) + struct.pack("<I", version)
            + struct.pack("<Q", tensor_count)
            + struct.pack("<Q", len(keys) if kv_count is None else kv_count))
    with open(path, "wb") as handle:
        handle.write(head + body)
        if file_bytes > handle.tell():
            handle.truncate(file_bytes)
    return path


def default_keys(arch="llama", layers=32, heads=32, kv_heads=8, embd=4096, ctx=131072):
    return [
        _kv("general.architecture", STRING, _s(arch.encode())),
        _kv("general.name", STRING, _s(b"Test Model")),
        _kv("general.file_type", U32, struct.pack("<I", 15)),      # Q4_K_M
        _kv("general.size_label", STRING, _s(b"8B")),
        _kv(f"{arch}.block_count", U32, struct.pack("<I", layers)),
        _kv(f"{arch}.context_length", U32, struct.pack("<I", ctx)),
        _kv(f"{arch}.embedding_length", U32, struct.pack("<I", embd)),
        _kv(f"{arch}.attention.head_count", U32, struct.pack("<I", heads)),
        _kv(f"{arch}.attention.head_count_kv", U32, struct.pack("<I", kv_heads)),
    ]


GIB = 1024 ** 3


@pytest.fixture
def model(served_dir):
    return write_gguf(served_dir / "Test-8B-Q4_K_M.gguf", file_bytes=int(4.92 * GIB))


def test_the_header_answers_what_a_launch_needs(model):
    header = gguf.read_header(model)
    assert header.architecture == "llama"
    assert header.block_count == 32
    assert header.head_count_kv == 8
    assert header.context_length == 131072
    # general.file_type is llama_ftype, NOT the ggml_type enum — 15 is Q4_K_M in
    # one and Q5_K in the other, and reading the wrong table mislabels the file.
    assert header.quant == "Q4_K_M"


def test_the_kv_cache_formula_matches_llama_cpps_own(model):
    """K and V are each n_embd_gqa x n_ctx per layer, sized through the cache
    type's block size. n_embd_k_gqa here is 4096/32 x 8 = 1024."""
    header = gguf.read_header(model)
    assert header.n_embd_k_gqa == 1024
    # 2 (K and V) x 1024 x 4096 tokens x 32 layers x 2 bytes = 0.5 GiB.
    assert header.kv_bytes(4096) == 512 * 1024 ** 2
    # Linear in context.
    assert header.kv_bytes(8192) == 2 * header.kv_bytes(4096)
    # q8_0 is 34 bytes per 32 elements against f16's 2 per 1, so roughly half.
    quantised = header.kv_bytes(4096, type_k="q8_0", type_v="q8_0")
    assert 0.5 < quantised / header.kv_bytes(4096) < 0.6


def test_a_missing_head_count_kv_means_no_grouped_attention(tmp_path):
    """llama-hparams.cpp falls back to head_count. Defaulting it to zero instead
    would make every pre-GQA model's cache come out as nothing."""
    path = write_gguf(tmp_path / "mha.gguf",
                      keys=[k for k in default_keys()
                            if b"head_count_kv" not in k], file_bytes=GIB)
    header = gguf.read_header(path)
    assert header.head_count_kv is None
    assert header.n_head_kv == 32
    assert header.n_embd_k_gqa == 4096


def test_offload_counts_the_output_layer_too(model):
    """llama.cpp offloads the last N of block_count + 1 virtual layers, the
    extra one standing for the output head. Dividing by block_count would
    under-count every partial offload."""
    header = gguf.read_header(model)
    assert header.offload_fraction(33) == 1.0
    assert header.offload_fraction(99) == 1.0
    assert header.offload_fraction(0) == 0.0
    # 16 of 33, not 16 of 32.
    assert header.offload_fraction(16) == pytest.approx(16 / 33)


def test_an_unset_layer_count_is_priced_as_everything(model):
    """None is not zero. It means the operator declared nothing, and llama.cpp
    would then fit itself to a 1 GiB margin — smaller than this host's reserve,
    so the guard has to assume the worst."""
    assert gguf.read_header(model).offload_fraction(None) == 1.0


def test_a_huge_tokenizer_is_walked_but_never_held(tmp_path):
    """A real model's header is dominated by a 150k-token vocabulary. Holding it
    would put hundreds of megabytes into a budget survey that runs every tick."""
    tokens = (struct.pack("<I", STRING) + struct.pack("<Q", 5000)
              + b"".join(_s(f"token{i}".encode()) for i in range(5000)))
    path = write_gguf(tmp_path / "vocab.gguf",
                      keys=[*default_keys(),
                            _kv("tokenizer.ggml.tokens", ARRAY, tokens),
                            _kv("tokenizer.chat_template", STRING, _s(b"x" * 200_000))],
                      file_bytes=GIB)
    header = gguf.read_header(path)
    assert header.block_count == 32          # the keys after the array still parse
    assert "tokenizer.ggml.tokens" not in header.keys
    # Its presence is recorded; its megabyte of Jinja is not.
    assert header.chat_template is True


def test_a_fixed_width_array_is_skipped_by_arithmetic(tmp_path):
    """Not by reading 5000 values. The cost of a header read is what makes it
    affordable on every poll."""
    types = struct.pack("<I", U32) + struct.pack("<Q", 5000) + b"\0" * 20_000
    path = write_gguf(tmp_path / "types.gguf",
                      keys=[*default_keys(), _kv("tokenizer.ggml.token_type", ARRAY, types)],
                      file_bytes=GIB)
    assert gguf.read_header(path).block_count == 32


@pytest.mark.parametrize("keys,kv_count,magic,why", [
    (None, None, 0x11111111, "wrong magic"),
    (None, gguf.MAX_KV + 1, gguf.MAGIC, "an incredible key count"),
])
def test_a_file_that_is_not_one_is_refused(tmp_path, keys, kv_count, magic, why):
    path = write_gguf(tmp_path / "bad.gguf", keys=keys, kv_count=kv_count, magic=magic)
    with pytest.raises(gguf.GGUFError):
        gguf.read_header(path)


def test_a_lying_string_length_fails_before_it_allocates(tmp_path):
    """The failure mode this format invites: a corrupt uint64 is indistinguishable
    from a real one until something acts on it."""
    body = _s(b"general.architecture") + struct.pack("<I", STRING) + struct.pack("<Q", 1 << 40)
    path = tmp_path / "lying.gguf"
    path.write_bytes(struct.pack("<I", gguf.MAGIC) + struct.pack("<I", 3)
                     + struct.pack("<Q", 1) + struct.pack("<Q", 1) + body)
    with pytest.raises(gguf.GGUFError):
        gguf.read_header(path)


def test_a_truncated_download_is_not_a_crash(tmp_path):
    """Half a header is what a pull in progress looks like. read_cached is the
    entry point every caller uses, and it answers None rather than raising —
    an unreadable model is an unknown footprint, which the caller reports."""
    path = tmp_path / "partial.gguf"
    path.write_bytes(struct.pack("<I", gguf.MAGIC) + struct.pack("<I", 3)
                     + struct.pack("<Q", 291) + struct.pack("<Q", 40))
    with pytest.raises(gguf.GGUFError):
        gguf.read_header(path)
    assert gguf.read_cached(path) is None
    assert gguf.read_cached(tmp_path / "absent.gguf") is None


def test_the_cache_is_keyed_on_the_file_not_the_path(tmp_path, model):
    """A re-quantised file at the same path has to be re-read, or its footprint
    is answered from a header describing different weights."""
    first = gguf.read_cached(model)
    assert gguf.read_cached(model) is first          # same file: cached
    write_gguf(model, keys=default_keys(layers=80), file_bytes=int(9 * GIB))
    again = gguf.read_cached(model)
    assert again is not None and again.block_count == 80


def test_a_cache_type_this_build_does_not_know_is_not_free(model):
    """Falling back to zero for an unrecognised -ctk would price a cache at
    nothing, which is the direction that freezes a machine."""
    assert gguf.cache_bytes_per_element("f16") == 2
    assert gguf.cache_bytes_per_element("q4_0") == 18 / 32
    assert gguf.cache_bytes_per_element("some_future_quant") == 2
    assert gguf.cache_bytes_per_element(None) == 2


def test_is_gguf_reads_the_bytes_not_the_name(tmp_path, model):
    """A .gguf that is really a git-lfs pointer is the normal shape of a repo
    that was cloned rather than pulled."""
    assert gguf.is_gguf(model)
    pointer = tmp_path / "pointer.gguf"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    assert not gguf.is_gguf(pointer)


# --- what a crafted file can throw ---------------------------------------
#
# read_cached's contract is that it never raises: an unreadable model is an
# unknown footprint, and every caller — the budget survey, the watchdog's
# per-container loop, the launch guard — relies on that. Each case below escaped
# the original net, and each reaches a synchronous pricer.

def _one_key(tmp_path, name, key, kind, payload):
    path = tmp_path / name
    body = _s(key.encode()) + struct.pack("<I", kind) + payload
    path.write_bytes(struct.pack("<I", gguf.MAGIC) + struct.pack("<I", 3)
                     + struct.pack("<Q", 0) + struct.pack("<Q", 2)
                     + _kv("general.architecture", STRING, _s(b"llama")) + body)
    return path


@pytest.mark.parametrize("kind,payload,why", [
    (F32, struct.pack("<f", float("nan")), "NaN layer count"),
    (F32, struct.pack("<f", float("inf")), "infinite layer count"),
    (I32, struct.pack("<i", -1), "negative layer count"),
    (U64, struct.pack("<Q", 5_000_000), "an absurd layer count"),
])
def test_a_nonsense_layer_count_is_not_a_layer_count(tmp_path, kind, payload, why):
    """int(NaN) raises ValueError, int(inf) raises OverflowError, -1 makes
    offload_fraction's divisor zero, and five million is a loop bound in the
    recommender. None of the four is a GGUFError, so all four escaped."""
    path = _one_key(tmp_path, "odd.gguf", "llama.block_count", kind, payload)
    header = gguf.read_cached(path)
    assert header is not None, why
    assert header.block_count is None, why
    # And the pricer these feed does not divide by zero or take forever.
    assert header.offload_fraction(8) == 1.0
    assert header.kv_bytes(4096) is None


def test_a_key_name_cannot_be_a_megabyte(tmp_path):
    """A key is capped by the spec at 65535 bytes. Read without that cap, one
    'key' could be as large as the whole cursor budget — allocated once as bytes
    and again as a string, on every budget survey."""
    path = tmp_path / "fatkey.gguf"
    path.write_bytes(struct.pack("<I", gguf.MAGIC) + struct.pack("<I", 3)
                     + struct.pack("<Q", 0) + struct.pack("<Q", 1)
                     + struct.pack("<Q", 1 << 20) + b"k" * 16)
    with pytest.raises(gguf.GGUFError):
        gguf.read_header(path)
    assert gguf.read_cached(path) is None


def test_deeply_nested_arrays_do_not_blow_the_stack(tmp_path):
    """RecursionError is not a GGUFError either."""
    payload = struct.pack("<I", U32) + struct.pack("<Q", 1) + b"\0\0\0\0"
    for _ in range(64):
        payload = struct.pack("<I", ARRAY) + struct.pack("<Q", 1) + payload
    path = _one_key(tmp_path, "nested.gguf", "llama.block_count", ARRAY, payload)
    assert gguf.read_cached(path) is None
