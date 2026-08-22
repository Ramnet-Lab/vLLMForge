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
async def test_fp8_block_weights_ask_for_the_deep_gemm_switch(box):
    """The failure this prevents, measured on Qwen/Qwen3.8-27B-FP8: vLLM
    auto-disables DeepGemm for the layers on Blackwell and builds them for
    CUTLASS, but kernel_warmup is gated on VLLM_USE_DEEP_GEMM alone, so the
    warmup still hands DeepGEMM a layout it does not know and the engine dies on
    "Unknown recipe" — four and a half minutes in, with every shard read."""
    place, mp = box
    place(mp.Profile(reference="org/fp8", found=True, architectures=["LlamaForCausalLM"],
                     quant_method="compressed-tensors", weight_bytes=30 * GIB,
                     block_scaled_fp8=True,
                     quantization={"config_groups": {"FP8_BLOCK": {
                         "weights": {"strategy": "block", "block_structure": [128, 128]}}}},
                     has_safetensors=True, chat_template=True, supported=True))
    rec = (await recommend.build("org/fp8")).to_dict()
    assert any("DeepGEMM" in f["text"] for f in rec["findings"])
    assert rec["env"] == {"VLLM_USE_DEEP_GEMM": "0"}
    assert "VLLM_USE_DEEP_GEMM=0" in rec["headline"], "it has to be findable from the headline"


@pytest.mark.anyio
async def test_a_variable_already_in_the_environment_box_is_not_asked_for_again(box):
    """A recommendation says what to change. Repeating a variable the operator
    has already typed in is how the panel trains them to stop reading it."""
    place, mp = box
    profile = dict(found=True, architectures=["LlamaForCausalLM"], weight_bytes=30 * GIB,
                   block_scaled_fp8=True, quant_method="fp8", has_safetensors=True,
                   chat_template=True, supported=True)
    place(mp.Profile(reference="org/fp8", **profile))

    rec = (await recommend.build("org/fp8", env={"VLLM_USE_DEEP_GEMM": "0"})).to_dict()
    # Still explained — the reason it must stay set does not go away — but no
    # longer something the headline asks for.
    assert rec["env"] == {"VLLM_USE_DEEP_GEMM": "0"}
    assert "VLLM_USE_DEEP_GEMM" not in rec["headline"]


@pytest.mark.anyio
async def test_a_per_tensor_fp8_checkpoint_is_left_alone(box):
    """Only the block layout reaches the kernel that fails. Asking every FP8
    model to disable DeepGEMM would cost throughput on the ones it serves."""
    place, mp = box
    place(mp.Profile(reference="org/fp8-tensor", found=True,
                     architectures=["LlamaForCausalLM"], quant_method="fp8",
                     weight_bytes=30 * GIB, has_safetensors=True, chat_template=True,
                     supported=True))
    rec = (await recommend.build("org/fp8-tensor")).to_dict()
    assert rec["env"] == {}
    assert not any("DeepGEMM" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_a_modelopt_checkpoint_with_tied_embeddings_always_warns(box):
    """The ignore list is not an escape hatch. ModelOpt never quantises
    embed_tokens; the object that raises is lm_head, and excluding it does not
    help — an excluded ParallelLMHead gets UnquantizedLinearMethod, which lacks
    tie_weights exactly as the quantised method does. Treating the ignore list
    as an exemption would stay silent on a checkpoint that still fails."""
    place, mp = box
    base = dict(found=True, architectures=["LlamaForCausalLM"], quant_method="modelopt",
                tie_word_embeddings=True, weight_bytes=30 * GIB, has_safetensors=True,
                chat_template=True, supported=True)

    for ignore in (["model.embed_vision*"], ["lm_head", "model.embed_vision*"],
                   ["re:.*embed_tokens.*"], []):
        place(mp.Profile(reference="org/tied", **base, quantization={"ignore": ignore}))
        rec = (await recommend.build("org/tied")).to_dict()
        assert any("ties lm_head" in f["text"] for f in rec["findings"]), ignore

    # A compressed-tensors build of the same model is fine: its excluded lm_head
    # falls through to the unquantised embedding method, which does tie.
    place(mp.Profile(reference="org/ct", **{**base, "quant_method": "compressed-tensors"},
                     quantization={"ignore": ["lm_head"]}))
    clean = (await recommend.build("org/ct")).to_dict()
    assert not any("ties lm_head" in f["text"] for f in clean["findings"])


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


@pytest.mark.anyio
async def test_a_gemma_template_names_its_own_parsers(box):
    """vLLM picks parsers by name, and the name is written down nowhere in the
    repo — but the tokens the template emits are, and each parser keys on its
    own. Every gemma4 template in this cache carries these three markers."""
    place, mp = box
    place(mp.Profile(reference="org/g", found=True, architectures=["Gemma4ForConditionalGeneration"],
                     model_type="gemma4", weight_bytes=30 * GIB, has_safetensors=True,
                     chat_template=True, chat_template_source="chat_template.jinja",
                     template_markers=["<|tool_call>", "<|channel>", "if not enable_thinking"],
                     supported=True))
    rec = (await recommend.build("org/g")).to_dict()
    assert rec["args"]["tool_call_parser"] == "gemma4"
    assert rec["args"]["reasoning_parser"] == "gemma4"
    # A parser without this is silently ignored, and vLLM does not say so.
    assert rec["args"]["enable_auto_tool_choice"] is True
    assert any("enable_thinking" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_a_qwen3_xml_template_is_recognised(box):
    place, mp = box
    place(mp.Profile(reference="org/q", found=True, architectures=["Qwen3_5ForConditionalGeneration"],
                     model_type="qwen3_5", weight_bytes=20 * GIB, has_safetensors=True,
                     chat_template=True, template_markers=["<tool_call>", "<function=",
                                                           "<parameter=", "<think>", "</think>"],
                     supported=True))
    rec = (await recommend.build("org/q")).to_dict()
    assert rec["args"]["tool_call_parser"] == "qwen3_xml"
    assert rec["args"]["reasoning_parser"] == "qwen3"


@pytest.mark.anyio
async def test_the_same_three_tokens_do_not_make_every_model_qwen3(box):
    """step3p5's parser reads <tool_call>, <function= and <parameter= too, so the
    three-token lock alone picks the wrong engine — silently. The model family
    is what disambiguates."""
    place, mp = box
    place(mp.Profile(reference="org/s", found=True, architectures=["Step3p5ForCausalLM"],
                     model_type="step3p5", weight_bytes=20 * GIB, has_safetensors=True,
                     chat_template=True,
                     template_markers=["<tool_call>", "<function=", "<parameter="],
                     supported=True))
    rec = (await recommend.build("org/s")).to_dict()
    assert "tool_call_parser" not in rec["args"]


@pytest.mark.anyio
async def test_an_ambiguous_thinking_template_is_named_not_guessed(box):
    """<think>/</think> is claimed by five registered parsers that differ in what
    they do with the tokens. A wrong auto-set parser is worse than none."""
    place, mp = box
    place(mp.Profile(reference="org/t", found=True, architectures=["LlamaForCausalLM"],
                     model_type="llama", weight_bytes=8 * GIB, has_safetensors=True,
                     chat_template=True, template_markers=["<think>", "</think>"],
                     supported=True))
    rec = (await recommend.build("org/t")).to_dict()
    assert "reasoning_parser" not in rec["args"]
    assert any("five registered parsers" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_a_pooling_model_is_offered_no_parsers(box):
    """It has no chat endpoint to parse anything for."""
    place, mp = box
    place(mp.Profile(reference="org/e", found=True, architectures=["Qwen3ForCausalLM"],
                     runner="pooling", runner_reason="modules.json declares a Pooling module",
                     weight_bytes=8 * GIB, has_safetensors=True, chat_template=True,
                     template_markers=["<|tool_call>", "<|channel>"], supported=True))
    rec = (await recommend.build("org/e")).to_dict()
    assert "tool_call_parser" not in rec["args"]
    assert "reasoning_parser" not in rec["args"]


@pytest.mark.anyio
async def test_a_parser_set_without_auto_tool_choice_is_called_out(box):
    place, mp = box
    place(mp.Profile(reference="org/m", found=True, architectures=["LlamaForCausalLM"],
                     weight_bytes=8 * GIB, has_safetensors=True, chat_template=True,
                     supported=True))
    rec = (await recommend.build("org/m", "", {"tool_call_parser": "hermes"})).to_dict()
    assert any("ignored unless" in f["text"] for f in rec["findings"])


@pytest.mark.anyio
async def test_pooling_a_model_that_fits_on_one_machine_is_questioned(box):
    """Pooling costs a network hop per token, an engine that aborts if either
    node drops, and the pipeline-parallel code paths — a class of failure a
    single-machine launch never meets."""
    place, mp = box
    place(mp.Profile(reference="org/m", found=True, architectures=["LlamaForCausalLM"],
                     max_position_embeddings=131072, num_hidden_layers=32,
                     num_key_value_heads=8, head_dim=128, weight_bytes=48 * GIB,
                     has_safetensors=True, chat_template=True, supported=True))

    alone = (await recommend.build("org/m", "", {}, [])).to_dict()
    assert not any("does not need" in f["text"] for f in alone["findings"])

    spread = (await recommend.build("org/m", "", {}, ["local", "node2"])).to_dict()
    assert any("does not need 2 machines" in f["text"] for f in spread["findings"])


@pytest.mark.anyio
async def test_pooling_a_model_that_does_not_fit_is_left_alone(box):
    """When it genuinely does not fit, spreading it is the whole point."""
    place, mp = box
    place(mp.Profile(reference="org/huge", found=True, architectures=["LlamaForCausalLM"],
                     max_position_embeddings=131072, num_hidden_layers=60,
                     num_key_value_heads=16, head_dim=256, weight_bytes=90 * GIB,
                     has_safetensors=True, chat_template=True, supported=True))
    spread = (await recommend.build("org/huge", "", {}, ["local", "node2"])).to_dict()
    assert not any("does not need" in f["text"] for f in spread["findings"])


@pytest.mark.anyio
async def test_a_utilisation_below_the_weights_is_refused_not_offered(box, monkeypatch):
    """vLLM's budget has to cover the weights before it covers anything else.
    Below that, --max-model-len auto gives up with 'not enough GPU memory
    available to serve even a single token' — after reading the whole
    checkpoint. Handing over a number certain to fail that way is worse than
    saying it does not fit."""
    from app import safety, telemetry

    place, mp = box
    place(mp.Profile(reference="org/big", found=True, architectures=["LlamaForCausalLM"],
                     max_position_embeddings=131072, num_hidden_layers=30,
                     num_key_value_heads=8, head_dim=256, weight_bytes=int(48 * GIB),
                     has_safetensors=True, chat_template=True, supported=True))

    # A busy machine: the weights still nominally fit in what is available, so
    # the outright blocker does not fire — but the ceiling cannot cover them.
    async def busy(node=None, exclude=None):
        return safety.Budget(total_bytes=int(TOTAL), available_bytes=int(56 * GIB),
                             free_bytes=int(56 * GIB), reserve_bytes=int(32 * GIB))

    monkeypatch.setattr(recommend.safety, "current_budget", busy)
    monkeypatch.setattr(recommend.telemetry, "read_meminfo", lambda: telemetry.HostMemory(
        total_bytes=int(TOTAL), available_bytes=int(56 * GIB)))

    rec = (await recommend.build("org/big")).to_dict()
    assert rec["ok"] is False and rec["level"] == "block"
    assert rec["args"] == {}
    assert "serve even a single token" in rec["findings"][0]["text"]


@pytest.mark.anyio
async def test_the_floor_lifts_a_recommendation_that_would_be_too_small(box):
    """When the ceiling is generous but the model is large, the answer must
    still cover the weights — never the smaller of the two."""
    place, mp = box
    place(mp.Profile(reference="org/big", found=True, architectures=["LlamaForCausalLM"],
                     max_position_embeddings=1024, num_hidden_layers=30,
                     num_key_value_heads=8, head_dim=256, weight_bytes=int(48 * GIB),
                     has_safetensors=True, chat_template=True, supported=True))
    rec = (await recommend.build("org/big")).to_dict()
    util = rec["args"]["gpu_memory_utilization"]
    floor = recommend.floor_utilisation(
        mp.Profile(reference="x", weight_bytes=int(48 * GIB)), int(TOTAL))
    assert util >= floor
    assert util * TOTAL > 48 * GIB, "the budget has to cover the weights"


@pytest.mark.anyio
async def test_a_custom_sampler_model_cannot_be_pooled(box):
    """Measured, not guessed: this exact configuration loaded 48 GiB of weights,
    sized 1.1M tokens of KV cache and captured its graphs before the far rank
    died on `assert sampled_token_ids.dtype == torch.int64`. It is refused now,
    before anything starts."""
    place, mp = box
    profile = mp.Profile(reference="google/diffusion", found=True,
                         architectures=["DiffusionGemmaForBlockDiffusion"],
                         model_type="diffusion_gemma", weight_bytes=48 * GIB,
                         has_safetensors=True, chat_template=True, supported=True,
                         custom_sampler=True)
    place(profile)

    alone = (await recommend.build("google/diffusion", "", {}, [])).to_dict()
    assert alone["ok"] is True, "it serves perfectly well on one machine"

    spread = (await recommend.build("google/diffusion", "", {}, ["local", "node2"])).to_dict()
    assert spread["ok"] is False and spread["level"] == "block"
    assert spread["args"] == {}
    assert "cannot be split across machines" in spread["headline"]


@pytest.mark.anyio
async def test_a_pooled_engine_is_sized_per_machine(box):
    """Pipeline parallelism divides the layers, so the weights and the KV cache
    divide with them. Sizing a pooled server as if one machine held the whole
    model asks for roughly twice what each node needs — and on a large model
    that is the difference between a recommendation and a refusal."""
    place, mp = box
    place(mp.Profile(reference="org/big", found=True, architectures=["LlamaForCausalLM"],
                     max_position_embeddings=131072, num_hidden_layers=60,
                     num_key_value_heads=16, head_dim=256, weight_bytes=int(58 * GIB),
                     has_safetensors=True, chat_template=True, supported=True))

    alone = (await recommend.build("org/big", "", {}, [])).to_dict()
    spread = (await recommend.build("org/big", "", {}, ["local", "node2"])).to_dict()

    one = alone["args"]["gpu_memory_utilization"]
    two = spread["args"]["gpu_memory_utilization"]
    assert two < one, "each machine holds half the layers"
    # The overhead is per process, so the halving is of the model, not of the
    # whole figure — two nodes need more than half each, not exactly half.
    assert two > one / 2


def test_the_floor_and_the_need_both_divide_by_the_shards():
    from app import model_profile as mp

    profile = mp.Profile(reference="x", weight_bytes=int(58 * GIB), num_hidden_layers=60,
                         num_key_value_heads=16, head_dim=256,
                         max_position_embeddings=131072)
    total = int(TOTAL)
    assert recommend.floor_utilisation(profile, total, 1) > \
        recommend.floor_utilisation(profile, total, 2)
    assert recommend.needed_utilisation(profile, total, 32768, 1) > \
        recommend.needed_utilisation(profile, total, 32768, 2)
    # One machine is the default and must be unchanged by the new parameter.
    assert recommend.floor_utilisation(profile, total) == \
        recommend.floor_utilisation(profile, total, 1)


@pytest.mark.anyio
async def test_weights_that_do_not_fit_one_machine_may_fit_two(box):
    place, mp = box
    # 100 GiB of weights needs 0.88 of a 121.7 GiB machine once the CUDA
    # context is counted, and the guard's reserve caps a single node at 0.73.
    place(mp.Profile(reference="org/huge", found=True, architectures=["LlamaForCausalLM"],
                     max_position_embeddings=8192, num_hidden_layers=60,
                     num_key_value_heads=16, head_dim=256, weight_bytes=int(100 * GIB),
                     has_safetensors=True, chat_template=True, supported=True))

    alone = (await recommend.build("org/huge", "", {}, [])).to_dict()
    assert alone["ok"] is False

    spread = (await recommend.build("org/huge", "", {}, ["local", "node2"])).to_dict()
    assert spread["ok"] is True, "50 GiB a machine fits where 100 does not"


@pytest.mark.anyio
async def test_a_recurrent_model_gets_its_sequence_cap_set(box):
    """On a hybrid model --max-num-seqs stops being an admission cap and becomes
    a memory demand: one state block per sequence, allocated before the engine
    serves anything. vLLM's default of 256 has to fit or it refuses to capture a
    graph — which is how Qwen3.8-27B (48 of 64 layers linear_attention) died at
    the 0.34 this page recommended for it, with "max_num_seqs (256) exceeds
    available Mamba cache blocks (125)"."""
    place, mp = box
    place(mp.Profile(reference="org/hybrid", found=True, architectures=["Qwen3_5ForCausalLM"],
                     model_type="qwen3_5", weight_bytes=28 * GIB,
                     layer_types={"linear_attention": 48, "full_attention": 16},
                     num_hidden_layers=64, num_key_value_heads=4, head_dim=256,
                     max_position_embeddings=262144,
                     has_safetensors=True, chat_template=True, supported=True))

    rec = (await recommend.build("org/hybrid")).to_dict()
    assert rec["args"]["max_num_seqs"] == recommend.RECURRENT_SEQS
    assert rec["args"]["max_num_seqs"] < 256, "vLLM's default is the thing that fails"
    why = next(s["why"] for s in rec["suggestions"] if s["dest"] == "max_num_seqs")
    assert "48 of this model's 64 layers" in why


@pytest.mark.anyio
async def test_an_attention_only_model_keeps_vllms_own_sequence_cap(box):
    """Setting a flag to a number vLLM would have chosen better is the thing
    this module exists not to do. Without recurrent layers the cap costs no
    memory up front and there is nothing to say about it."""
    place, mp = box
    place(mp.Profile(reference="org/plain", found=True, architectures=["LlamaForCausalLM"],
                     weight_bytes=16 * GIB, num_hidden_layers=32, num_key_value_heads=8,
                     head_dim=128, max_position_embeddings=131072,
                     has_safetensors=True, chat_template=True, supported=True))
    rec = (await recommend.build("org/plain")).to_dict()
    assert "max_num_seqs" not in rec["args"]


@pytest.mark.anyio
async def test_a_cap_the_operator_chose_is_left_alone(box):
    """They may want more concurrency than this page would pick, and they can
    see the consequence in the memory guard. Overwriting it would be the page
    arguing with them."""
    place, mp = box
    place(mp.Profile(reference="org/hybrid", found=True, architectures=["Qwen3_5ForCausalLM"],
                     model_type="qwen3_5", weight_bytes=28 * GIB,
                     layer_types={"linear_attention": 48, "full_attention": 16},
                     num_hidden_layers=64, num_key_value_heads=4, head_dim=256,
                     has_safetensors=True, chat_template=True, supported=True))
    rec = (await recommend.build("org/hybrid", args={"max_num_seqs": 64})).to_dict()
    assert "max_num_seqs" not in rec["args"]
