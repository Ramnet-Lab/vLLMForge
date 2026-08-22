"""Turning a model on a machine into a configuration that starts.

The failure this exists to prevent is concrete: a server saved with
--gpu-memory-utilization 0.95 on a 121.7 GiB box with 107.8 GiB free, accepted
by the form, started by autostart, and refused by vLLM four minutes into
loading 58 GiB of weights.
"""

from __future__ import annotations

import pytest

from app import recommend

GIB = 1024 ** 3
TOTAL = 121.69 * GIB


def test_the_utilisation_ceiling_is_free_memory_not_all_memory():
    """vLLM's startup check is ceil(total * util) > free -> refuse, and on a
    unified-memory box total is host MemTotal while free is MemAvailable. Its
    own 0.92 default is therefore unusable on any machine holding more than 8%
    of its RAM, which this one always is."""
    # The exact numbers from the failure: 107.81 free of 121.69.
    util = recommend.safe_utilisation(int(TOTAL), int(107.81 * GIB), budget_free_util=1.0)
    assert util == 0.85
    assert util * TOTAL < 107.81 * GIB, "what is asked for has to fit in what is free"
    assert util < 0.92, "vLLM's default would have been refused"


def test_the_dashboards_own_reserve_still_binds():
    """Free memory is one bound; the guard's reserve for the OS and torch.compile
    is the other, and the recommendation takes whichever is tighter."""
    generous = recommend.safe_utilisation(int(TOTAL), int(120 * GIB), budget_free_util=1.0)
    guarded = recommend.safe_utilisation(int(TOTAL), int(120 * GIB), budget_free_util=0.73)
    assert generous > guarded == 0.73


def test_a_full_machine_recommends_nothing_rather_than_a_negative():
    assert recommend.safe_utilisation(int(TOTAL), 0, budget_free_util=0.5) == 0.0
    assert recommend.safe_utilisation(0, 0, budget_free_util=0.5) == 0.0


def test_the_margin_absorbs_drift():
    """MemAvailable moves while an image is pulled and weights are read. Without
    headroom a recommendation computed now is refused by a machine that got a
    little busier in between."""
    exact = int(0.80 * TOTAL)
    assert recommend.safe_utilisation(int(TOTAL), exact, budget_free_util=1.0) == 0.77


@pytest.fixture
def box(monkeypatch):
    """A machine with a known amount of memory free, and a model on its disk."""
    from app import model_profile, safety, telemetry

    async def budget(node=None, exclude=None):
        return safety.Budget(total_bytes=int(TOTAL), available_bytes=int(100 * GIB),
                             free_bytes=int(100 * GIB), reserve_bytes=int(32 * GIB))

    monkeypatch.setattr(safety, "current_budget", budget)
    monkeypatch.setattr(recommend.safety, "current_budget", budget)
    monkeypatch.setattr(telemetry, "read_meminfo", lambda: telemetry.HostMemory(
        total_bytes=int(TOTAL), available_bytes=int(100 * GIB)))
    monkeypatch.setattr(recommend.telemetry, "read_meminfo", lambda: telemetry.HostMemory(
        total_bytes=int(TOTAL), available_bytes=int(100 * GIB)))

    def place(profile):
        monkeypatch.setattr(recommend.model_profile, "read", lambda ref: profile)
        return profile

    return place, model_profile


@pytest.mark.anyio
async def test_a_healthy_model_gets_two_flags_and_no_complaints(box):
    place, mp = box
    place(mp.Profile(reference="org/m", found=True, source="cache",
                     architectures=["LlamaForCausalLM"], model_type="llama",
                     max_position_embeddings=131072, num_hidden_layers=32,
                     num_key_value_heads=8, head_dim=128,
                     weight_bytes=16 * GIB, has_safetensors=True,
                     chat_template=True, chat_template_source="tokenizer_config.json",
                     supported=True))

    rec = (await recommend.build("org/m")).to_dict()
    assert rec["ok"] and rec["level"] == "ok"
    assert set(rec["args"]) == {"gpu_memory_utilization", "max_model_len"}
    assert rec["args"]["max_model_len"] == "auto"
    assert 0 < rec["args"]["gpu_memory_utilization"] <= 0.74


@pytest.mark.anyio
async def test_quantisation_and_dtype_are_deliberately_not_set(box):
    """vLLM detects both. A flag set to what vLLM would have chosen is a second
    source of truth that goes stale when the image is upgraded."""
    place, mp = box
    place(mp.Profile(reference="org/q", found=True, architectures=["LlamaForCausalLM"],
                     quant_method="compressed-tensors", dtype="bfloat16",
                     weight_bytes=8 * GIB, has_safetensors=True, chat_template=True,
                     supported=True))

    rec = (await recommend.build("org/q")).to_dict()
    assert "quantization" not in rec["args"]
    assert "dtype" not in rec["args"]
    assert "kv_cache_dtype" not in rec["args"]
    left = {entry["dest"] for entry in rec["left_alone"]}
    assert {"quantization", "dtype", "kv_cache_dtype"} <= left


@pytest.mark.anyio
async def test_a_repo_that_maps_its_own_code_gets_trust_remote_code(box):
    place, mp = box
    place(mp.Profile(reference="org/x", found=True, architectures=["ExoticForCausalLM"],
                     requires_remote_code=True, weight_bytes=GIB, has_safetensors=True,
                     chat_template=True, supported=True))
    rec = (await recommend.build("org/x")).to_dict()
    assert rec["args"]["trust_remote_code"] is True


@pytest.mark.anyio
async def test_an_adapter_is_refused_with_the_model_to_serve_instead(box):
    place, mp = box
    place(mp.Profile(reference="org/lora", found=True, is_adapter=True, base_model="org/base"))
    rec = (await recommend.build("org/lora")).to_dict()
    assert rec["ok"] is False and rec["level"] == "block"
    assert rec["args"] == {}
    assert "org/base" in rec["findings"][0]["text"]


@pytest.mark.anyio
async def test_weights_larger_than_free_memory_is_a_block_not_a_setting(box):
    place, mp = box
    place(mp.Profile(reference="org/huge", found=True, architectures=["LlamaForCausalLM"],
                     weight_bytes=int(110 * GIB), has_safetensors=True, supported=True))
    rec = (await recommend.build("org/huge")).to_dict()
    assert rec["ok"] is False
    assert "pool it across machines" in rec["findings"][0]["text"]


@pytest.mark.anyio
async def test_gguf_only_is_a_block(box):
    place, mp = box
    place(mp.Profile(reference="org/gguf", found=True, has_gguf=True, has_safetensors=False))
    rec = (await recommend.build("org/gguf")).to_dict()
    assert rec["ok"] is False and "GGUF" in rec["headline"]


@pytest.mark.anyio
async def test_a_pooling_model_is_warned_about_before_it_is_started(box):
    place, mp = box
    place(mp.Profile(reference="org/emb", found=True, architectures=["Qwen3ForCausalLM"],
                     runner="pooling", runner_reason="modules.json declares a Pooling module",
                     weight_bytes=8 * GIB, has_safetensors=True, chat_template=True,
                     supported=True))
    rec = (await recommend.build("org/emb")).to_dict()
    assert rec["ok"] and rec["level"] == "warn"
    assert any("embeddings, not chat" in f["text"] for f in rec["findings"])
    # And it does not also complain about the chat template — a pooling server
    # was never going to answer /v1/chat/completions anyway.
    assert not any("chat template" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_a_generator_without_a_template_is_warned(box):
    place, mp = box
    place(mp.Profile(reference="org/bare", found=True, architectures=["LlamaForCausalLM"],
                     chat_template=False, weight_bytes=GIB, has_safetensors=True,
                     supported=True))
    rec = (await recommend.build("org/bare")).to_dict()
    assert rec["level"] == "warn"
    assert any("no chat template" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_a_model_nobody_has_pulled_gets_memory_advice_and_no_guesses(box):
    """The memory advice is about the machine and holds regardless. Saying 'no
    chat template' about a repo nobody has pulled would be a guess as a fact."""
    place, mp = box
    place(mp.Profile(reference="org/new", found=False, source="missing"))
    rec = (await recommend.build("org/new")).to_dict()
    assert rec["ok"] and rec["level"] == "warn"
    assert set(rec["args"]) == {"gpu_memory_utilization", "max_model_len"}
    assert not any("chat template" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_an_unregistered_architecture_warns_without_refusing(box):
    """It may still load through the Transformers backend, so this is a warning
    and not a block — nothing offline can tell which."""
    place, mp = box
    place(mp.Profile(reference="org/odd", found=True, architectures=["MadeUpForCausalLM"],
                     supported=False, weight_bytes=GIB, has_safetensors=True,
                     chat_template=True))
    rec = (await recommend.build("org/odd")).to_dict()
    assert rec["ok"] is True and rec["level"] == "warn"
    assert any("Transformers backend" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_the_utilisation_is_sized_to_the_model_not_to_the_machine(box):
    """Reserving the whole box for a model that wants a fraction is how a
    machine ends up holding one engine when it could hold three."""
    place, mp = box
    small = mp.Profile(reference="org/small", found=True, architectures=["LlamaForCausalLM"],
                       max_position_embeddings=40960, num_hidden_layers=36,
                       num_key_value_heads=8, head_dim=128, weight_bytes=14 * GIB,
                       has_safetensors=True, chat_template=True, supported=True)
    place(small)
    modest = (await recommend.build("org/small")).to_dict()

    big = mp.Profile(**{**small.__dict__, "reference": "org/big", "weight_bytes": 58 * GIB})
    place(big)
    large = (await recommend.build("org/big")).to_dict()

    assert modest["args"]["gpu_memory_utilization"] < large["args"]["gpu_memory_utilization"]
    # And neither takes the whole ceiling just because it is there.
    ceiling = recommend.safe_utilisation(int(TOTAL), int(100 * GIB), 0.73)
    assert modest["args"]["gpu_memory_utilization"] < ceiling


@pytest.mark.anyio
async def test_a_model_too_big_for_a_comfortable_fit_still_gets_the_ceiling(box):
    """When what it wants exceeds what the machine can give, the answer is the
    most the machine can give — not a number that cannot start."""
    place, mp = box
    place(mp.Profile(reference="org/big", found=True, architectures=["LlamaForCausalLM"],
                     max_position_embeddings=131072, num_hidden_layers=60,
                     num_key_value_heads=16, head_dim=256, weight_bytes=int(75 * GIB),
                     has_safetensors=True, chat_template=True, supported=True))
    rec = (await recommend.build("org/big")).to_dict()
    ceiling = recommend.safe_utilisation(int(TOTAL), int(100 * GIB), 0.73)
    assert rec["args"]["gpu_memory_utilization"] == ceiling
    assert "The most this machine can give" in rec["suggestions"][0]["why"]


@pytest.mark.anyio
async def test_a_shorter_context_than_advertised_is_stated_not_warned(box):
    """With --max-model-len auto, a context shorter than the config advertises is
    the normal outcome. Calling it a problem trains the operator to ignore the
    panel."""
    place, mp = box
    place(mp.Profile(reference="org/long", found=True, architectures=["LlamaForCausalLM"],
                     max_position_embeddings=262144, num_hidden_layers=60,
                     num_key_value_heads=16, head_dim=256, weight_bytes=20 * GIB,
                     has_safetensors=True, chat_template=True, supported=True))
    rec = (await recommend.build("org/long")).to_dict()
    context = [f for f in rec["findings"] if "of context" in f["text"]]
    assert context and context[0]["level"] == "ok"
    assert rec["level"] == "ok"


@pytest.mark.anyio
async def test_fp8_block_weights_are_flagged_for_this_gpu(box):
    place, mp = box
    place(mp.Profile(reference="org/fp8", found=True, architectures=["LlamaForCausalLM"],
                     quant_method="compressed-tensors", weight_bytes=30 * GIB,
                     quantization={"config_groups": {"FP8_BLOCK": {
                         "weights": {"strategy": "block", "block_structure": [128, 128]}}}},
                     has_safetensors=True, chat_template=True, supported=True))
    rec = (await recommend.build("org/fp8")).to_dict()
    assert any("DeepGEMM" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_a_quantised_tied_embedding_is_flagged_unless_it_is_exempt(box):
    """model.embed_vision* is a different tensor; matching 'embed' loosely would
    silence a real warning, and matching nothing would raise a false one."""
    place, mp = box
    base = dict(found=True, architectures=["LlamaForCausalLM"], quant_method="modelopt",
                tie_word_embeddings=True, weight_bytes=30 * GIB, has_safetensors=True,
                chat_template=True, supported=True)

    place(mp.Profile(reference="org/tied", **base,
                     quantization={"ignore": ["model.embed_vision*", "re:.*vision.*"]}))
    flagged = (await recommend.build("org/tied")).to_dict()
    assert any("tied input embedding" in f["text"] for f in flagged["findings"])

    place(mp.Profile(reference="org/exempt", **base,
                     quantization={"ignore": ["lm_head", "model.embed_vision*"]}))
    clean = (await recommend.build("org/exempt")).to_dict()
    assert not any("tied input embedding" in f["text"] for f in clean["findings"])


@pytest.mark.anyio
async def test_cpu_offload_is_called_out_on_unified_memory(box):
    place, mp = box
    place(mp.Profile(reference="org/m", found=True, architectures=["LlamaForCausalLM"],
                     weight_bytes=8 * GIB, has_safetensors=True, chat_template=True,
                     supported=True))
    rec = (await recommend.build("org/m", "", {"cpu_offload_gb": 20})).to_dict()
    assert any("frees nothing here" in f["text"] for f in rec["findings"])
    clean = (await recommend.build("org/m", "", {})).to_dict()
    assert not any("frees nothing here" in f["text"] for f in clean["findings"])
