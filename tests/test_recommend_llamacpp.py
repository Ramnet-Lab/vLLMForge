"""The llama.cpp advisor.

Its whole job is the two numbers an operator would otherwise guess: how many
layers fit, and how much context is left over. What is pinned here is that it
degrades — full offload, then a shorter context, then layers on the CPU — rather
than answering "it fits" or "it does not".
"""

from __future__ import annotations

import pytest

from app import recommend_llamacpp, safety
from tests.test_gguf import write_gguf

GIB = 1024 ** 3
TOTAL = 48 * GIB


@pytest.fixture
def box(monkeypatch):
    """A node with a settable amount of memory free."""
    def free(gib: float):
        async def current(exclude=None, node=None):
            occupied = int((48 - gib - 2) * GIB)
            return safety.Budget(
                total_bytes=TOTAL, available_bytes=int(gib * GIB),
                free_bytes=int(gib * GIB), measured_gpu_bytes=occupied,
                reserve_bytes=2 * GIB, warn_reserve_bytes=3 * GIB)
        monkeypatch.setattr(recommend_llamacpp.safety, "current_budget", current)
    return free


@pytest.fixture
def model(served_dir):
    # A Llama-3-8B-shaped Q4_K_M: 4.92 GiB of weights, 32 layers, 8 KV heads.
    return write_gguf(served_dir / "Llama-3-8B-Q4_K_M.gguf", file_bytes=int(4.92 * GIB))


@pytest.mark.anyio
async def test_a_roomy_box_gets_every_layer_and_a_long_context(box, model):
    box(40)
    rec = await recommend_llamacpp.build(str(model))
    assert rec.ok and rec.level == "ok"
    args = {s.dest: s.value for s in rec.suggestions}
    assert args["n_gpu_layers"] == "all"
    assert args["ctx_size"] >= 32768


@pytest.mark.anyio
async def test_a_tight_box_shortens_the_context_before_it_moves_a_layer(box, model):
    """A layer on the CPU costs speed on every token forever; a shorter context
    costs only what it costs. So the context steps down first."""
    box(8)
    rec = await recommend_llamacpp.build(str(model))
    args = {s.dest: s.value for s in rec.suggestions}
    assert rec.ok
    assert args["n_gpu_layers"] == "all"
    assert args["ctx_size"] < 32768


@pytest.mark.anyio
async def test_a_model_bigger_than_the_box_is_split_with_the_cpu(box, model):
    """The thing llama.cpp can do that vLLM cannot, and the reason the advice is
    a layer count rather than a yes or a no."""
    box(3)
    rec = await recommend_llamacpp.build(str(model))
    args = {s.dest: s.value for s in rec.suggestions}
    assert rec.ok and rec.level == "warn"
    assert isinstance(args["n_gpu_layers"], int)
    assert 0 < args["n_gpu_layers"] < 33
    assert "CPU" in rec.headline or "CPU" in str(rec.suggestions[0].why)


@pytest.mark.anyio
async def test_a_full_box_refuses_rather_than_suggesting_nothing(box, model):
    box(0.1)
    rec = await recommend_llamacpp.build(str(model))
    assert not rec.ok and rec.level == "block"
    assert "free" in rec.headline.lower() or "fit" in rec.headline.lower()


@pytest.mark.anyio
async def test_a_safetensors_model_is_routed_back_to_vllm(box, served_dir):
    box(40)
    fake = served_dir / "model.gguf"
    fake.write_text("not a gguf at all")
    rec = await recommend_llamacpp.build(str(fake))
    assert not rec.ok
    assert rec.engine_hint == "vllm"


@pytest.mark.anyio
async def test_a_hub_reference_is_a_real_answer_not_a_failure(box):
    """llama.cpp resolves and pulls it itself, so nothing about its size is
    knowable yet — which is worth saying rather than guessing at."""
    box(40)
    rec = await recommend_llamacpp.build("bartowski/Qwen3-8B-GGUF:Q4_K_M")
    assert rec.ok and rec.level == "warn"
    assert "pulls it on the first start" in rec.findings[0].text


@pytest.mark.anyio
async def test_the_profile_fills_the_shape_the_card_already_renders(box, model):
    """A GGUF is not a different kind of model, only a different place to read
    it from — so it fills app/model_profile.py's keys rather than inventing a
    second shape for one card to have to know about."""
    box(40)
    rec = await recommend_llamacpp.build(str(model))
    profile = rec.profile
    assert profile["found"] and profile["has_gguf"] and not profile["has_safetensors"]
    assert profile["num_hidden_layers"] == 32
    assert profile["quantization"] == "Q4_K_M"
    # `supported` stays None: claiming knowledge of ggml's architecture list
    # from a header alone would be a guess, and an unknown answer must not read
    # as a refusal.
    assert profile["supported"] is None


@pytest.mark.anyio
async def test_the_advice_and_the_launch_verdict_cannot_disagree(box, model):
    """They are the same arithmetic — recommend_llamacpp prices candidates with
    the engine's own footprint function — so a recommendation the guard would
    then refuse is not expressible."""
    box(8)
    rec = await recommend_llamacpp.build(str(model))
    args = {s.dest: str(s.value) for s in rec.suggestions}
    verdict = await safety.check_launch(
        None, engine="llamacpp", params={"model": str(model), **args})
    assert verdict.ok, verdict.message
