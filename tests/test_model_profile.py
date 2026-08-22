"""Reading a model off the disk beside its weights.

The Serve page's job is to stop being a guessing game, and every guess it
removes rests on these files being read correctly. The cases here are the ones
that were wrong the first time or that real cached models actually exhibit.
"""

from __future__ import annotations

import json

import pytest

from app import model_profile


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A hub cache laid out the way huggingface_hub lays one out."""
    root = tmp_path / "hf"
    hub = root / "hub"
    hub.mkdir(parents=True)
    # Settings is frozen, so the seam is the accessor rather than the value.
    monkeypatch.setattr(model_profile, "_hub_root", lambda: hub.resolve())

    def add(repo_id: str, files: dict[str, object], *, sha: str = "abc123", ref: str = "abc123"):
        repo = root / "hub" / f"models--{repo_id.replace('/', '--')}"
        snapshot = repo / "snapshots" / sha
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            path = snapshot / name
            path.write_text(json.dumps(body) if isinstance(body, (dict, list)) else str(body))
        if ref:
            (repo / "refs").mkdir(exist_ok=True)
            (repo / "refs" / "main").write_text(ref)
        return snapshot

    return add


LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 131072,
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
}


def test_a_cached_repo_is_read_through_refs_main(cache):
    cache("org/model", {"config.json": LLAMA, "tokenizer_config.json": {
        "tokenizer_class": "PreTrainedTokenizerFast", "chat_template": "{{ x }}"}})

    profile = model_profile.read("org/model")

    assert profile.found and profile.source == "cache"
    assert profile.architectures == ["LlamaForCausalLM"]
    assert profile.max_position_embeddings == 131072
    assert profile.num_key_value_heads == 8
    assert profile.chat_template and profile.chat_template_source == "tokenizer_config.json"


def test_the_pinned_snapshot_wins_over_the_newest(cache):
    """A repo pulled at two revisions must resolve to the one refs/main names,
    not whichever directory was written last."""
    cache("org/model", {"config.json": LLAMA}, sha="old", ref="old")
    newer = cache("org/model", {"config.json": {**LLAMA, "max_position_embeddings": 4096}},
                  sha="new", ref="old")
    assert newer.exists()

    profile = model_profile.read("org/model")
    assert profile.max_position_embeddings == 131072


def test_without_refs_main_the_newest_snapshot_with_a_config_wins(cache):
    cache("org/model", {"README.md": "no config here"}, sha="empty", ref="")
    cache("org/model", {"config.json": LLAMA}, sha="real", ref="")

    profile = model_profile.read("org/model")
    assert profile.found and profile.architectures == ["LlamaForCausalLM"]


def test_a_gpt2_style_config_is_understood(cache):
    """GPT-2 spells every shape field differently. Reporting 'unknown' for a
    model whose config is complete would send the operator back to guessing."""
    cache("org/gpt2ish", {"config.json": {
        "model_type": "gpt2", "n_positions": 1024, "n_layer": 12,
        "n_embd": 768, "n_head": 12, "vocab_size": 50257,
    }})

    profile = model_profile.read("org/gpt2ish")
    assert profile.max_position_embeddings == 1024
    assert profile.num_hidden_layers == 12
    assert profile.hidden_size == 768
    assert profile.effective_head_dim() == 64
    # No num_key_value_heads means no grouping, so KV is sized by all the heads.
    assert profile.num_key_value_heads == 12
    assert any("names no architecture" in note for note in profile.notes)


def test_a_multimodal_config_reports_the_language_models_shape(cache):
    """Gemma-style repos nest the text model, and it is the text model that
    decides context length and KV cache size."""
    cache("org/vlm", {
        "config.json": {
            "architectures": ["SomeForConditionalGeneration"],
            "model_type": "somevlm",
            "vision_config": {"hidden_size": 1152},
            "text_config": {
                "max_position_embeddings": 262144, "num_hidden_layers": 60,
                "num_key_value_heads": 16, "head_dim": 256, "torch_dtype": "bfloat16",
            },
        },
        "preprocessor_config.json": {"image_processor_type": "SiglipImageProcessor"},
    })

    profile = model_profile.read("org/vlm")
    assert profile.is_multimodal
    assert profile.max_position_embeddings == 262144
    assert profile.num_hidden_layers == 60
    assert profile.kv_bytes_per_token() == 2 * 60 * 16 * 256 * 2


def test_kv_arithmetic_is_sized_by_kv_heads_not_attention_heads(cache):
    cache("org/gqa", {"config.json": LLAMA})
    profile = model_profile.read("org/gqa")
    # 32 layers x 8 kv heads x 128 head dim x 2 tensors x 2 bytes
    assert profile.effective_head_dim() == 128
    assert profile.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2
    assert profile.kv_bytes_per_token(kv_dtype_bytes=1) == 2 * 32 * 8 * 128


def test_a_template_shipped_as_its_own_file_counts(cache):
    cache("org/model", {"config.json": LLAMA,
                        "tokenizer_config.json": {"tokenizer_class": "X"},
                        "chat_template.jinja": "{% for m in messages %}{{ m }}{% endfor %}"})

    profile = model_profile.read("org/model")
    assert profile.chat_template and profile.chat_template_source == "chat_template.jinja"


def test_a_model_with_no_template_anywhere_says_so(cache):
    cache("org/model", {"config.json": LLAMA, "tokenizer_config.json": {"tokenizer_class": "X"}})
    profile = model_profile.read("org/model")
    assert profile.chat_template is False and profile.chat_template_source == ""


def test_an_empty_template_file_is_not_a_template(cache):
    cache("org/model", {"config.json": LLAMA, "chat_template.jinja": ""})
    assert model_profile.read("org/model").chat_template is False


def test_remote_code_is_detected_from_auto_map(cache):
    cache("org/exotic", {"config.json": {
        **LLAMA, "auto_map": {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}}})
    assert model_profile.read("org/exotic").requires_remote_code is True
    cache("org/plain", {"config.json": LLAMA})
    assert model_profile.read("org/plain").requires_remote_code is False


def test_quantization_method_is_normalised(cache):
    cache("org/q", {"config.json": {
        **LLAMA, "quantization_config": {"quant_method": "COMPRESSED-TENSORS", "bits": 8}}})
    profile = model_profile.read("org/q")
    assert profile.quant_method == "compressed-tensors"
    assert profile.quantization["bits"] == 8


def test_weights_are_measured_through_the_symlinks(cache, tmp_path):
    """A snapshot is symlinks into blobs/; sizing the link text instead of the
    blob would report a few hundred bytes for a 60 GiB model."""
    snapshot = cache("org/model", {"config.json": LLAMA})
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"x" * 5000)
    (snapshot / "model-00001-of-00001.safetensors").symlink_to(blob)

    profile = model_profile.read("org/model")
    assert profile.weight_bytes == 5000
    assert profile.shard_count == 1 and profile.has_safetensors


def test_an_adapter_is_recognised_and_names_its_base(cache):
    cache("org/lora", {"adapter_config.json": {
        "base_model_name_or_path": "org/base", "peft_type": "LORA"}})
    profile = model_profile.read("org/lora")
    assert profile.is_adapter and profile.base_model == "org/base"


def test_a_plain_directory_is_profiled_too(tmp_path):
    """What Heretic and fine-tuning write has no snapshot indirection."""
    out = tmp_path / "outputs" / "run-1"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(LLAMA))

    profile = model_profile.read(str(out))
    assert profile.found and profile.source == "directory"
    assert profile.num_hidden_layers == 32


def test_a_reference_that_is_not_here_is_missing_not_an_error(cache):
    profile = model_profile.read("org/never-pulled")
    assert profile.found is False and profile.source == "missing"
    assert profile.architectures == []


def test_a_traversal_never_escapes_the_cache(cache, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "config.json").write_text(json.dumps(LLAMA))

    assert model_profile.read("../elsewhere").found is False
    assert model_profile.read("org/../../elsewhere").found is False


def test_unreadable_json_does_not_take_the_profile_down(cache):
    cache("org/broken", {"config.json": "{ this is not json",
                         "tokenizer_config.json": "{ nor is this"})
    profile = model_profile.read("org/broken")
    assert profile.found is True
    assert profile.architectures == []
    assert any("config.json" in note for note in profile.notes)


# --- the shapes a real cache actually contains ----------------------------

GEMMA4 = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "dtype": "bfloat16",
    "vision_config": {"hidden_size": 1152},
    "text_config": {
        "max_position_embeddings": 262144,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "head_dim": 256,
        "num_global_key_value_heads": 4,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "layer_types": ["sliding_attention"] * 50 + ["full_attention"] * 10,
        "rope_parameters": {
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    },
}


def test_a_hybrid_models_kv_is_carried_by_its_full_attention_layers(cache):
    """Gemma-class models are mostly sliding-window layers, which stop growing
    at the window. Charging all sixty layers for the full context reports a
    20 GiB cache as 240 and sends the operator hunting for memory they have."""
    cache("google/gemma-like", {"config.json": GEMMA4})
    profile = model_profile.read("google/gemma-like")

    assert profile.layer_types == {"sliding_attention": 50, "full_attention": 10}
    # Only the 10 full layers grow, and they carry their own head geometry.
    assert profile.kv_bytes_per_token() == 10 * 2 * 4 * 512 * 2
    full = 10 * 2 * 4 * 512 * 2 * 262144
    sliding = 50 * 2 * 16 * 256 * 2 * 1024
    assert profile.kv_bytes(262144) == full + sliding
    assert profile.kv_bytes(262144) / 2 ** 30 == pytest.approx(20.78, abs=0.05)


def test_recurrent_layers_hold_no_kv_cache(cache):
    """A Qwen3.5-class model is 48 Mamba layers and 16 attention layers. The
    Mamba state is real memory but it is sized by concurrency, not context."""
    cache("org/hybrid", {"config.json": {
        "architectures": ["HybridForCausalLM"], "model_type": "hybrid",
        "num_hidden_layers": 64, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256,
        "max_position_embeddings": 262144,
        "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
    }})
    profile = model_profile.read("org/hybrid")

    assert profile.kv_bytes_per_token() == 16 * 2 * 4 * 256 * 2
    assert profile.kv_bytes(262144) == 16 * 2 * 4 * 256 * 2 * 262144
    assert any("recurrent" in note for note in profile.notes)


def test_rope_is_read_under_either_spelling(cache):
    cache("org/new", {"config.json": GEMMA4})
    cache("org/old", {"config.json": {
        **LLAMA, "rope_scaling": {"rope_type": "yarn", "factor": 4.0}, "rope_theta": 500000.0}})

    # Nested by layer type: several kinds at once, and the honest word is mixed.
    assert model_profile.read("org/new").rope_kind == "mixed"
    old = model_profile.read("org/old")
    assert old.rope_kind == "yarn" and old.rope_theta == 500000.0


def test_a_null_rope_scaling_is_not_a_rope_block(cache):
    """Qwen ships `rope_scaling: null` — present, and meaning nothing is scaled."""
    cache("org/qwen", {"config.json": {**LLAMA, "rope_scaling": None, "rope_theta": 1000000.0}})
    profile = model_profile.read("org/qwen")
    assert profile.rope_scaling is None and profile.rope_kind == ""
    assert profile.rope_theta == 1000000.0


def test_a_sentence_transformers_repo_is_a_pooling_server(cache):
    """vLLM checks for a Pooling module before it looks at the architecture, so
    a repo whose architectures say ForCausalLM still comes up serving
    /v1/embeddings. Discovering that after a four-minute load is the whole
    complaint this page exists to answer."""
    cache("Qwen/Embedder", {
        "config.json": LLAMA,
        "modules.json": [
            {"idx": 0, "name": "0", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "type": "sentence_transformers.models.Pooling"},
        ],
    })
    profile = model_profile.read("Qwen/Embedder")
    assert profile.runner == "pooling"
    assert "Pooling" in profile.runner_reason


def test_a_config_with_no_architecture_resolves_to_pooling(cache):
    """No architectures key means vLLM synthesises a bare `...Model` name from
    model_type, and a bare Model suffix is itself a pooling default — so a
    GPT-2 text model comes up as an embedding server."""
    cache("org/gpt2ish", {"config.json": {"model_type": "gpt2", "n_layer": 5, "n_embd": 32,
                                          "n_head": 4, "n_positions": 512}})
    profile = model_profile.read("org/gpt2ish")
    assert profile.runner == "pooling"
    assert "no architecture" in profile.runner_reason


def test_an_ordinary_causal_model_is_a_generator(cache):
    cache("org/plain", {"config.json": LLAMA})
    profile = model_profile.read("org/plain")
    assert profile.runner == "generate" and profile.runner_reason == ""


def test_an_architecture_this_build_cannot_load_is_named(cache, monkeypatch):
    """The registry is generated from the image, so the answer is that image's.
    Finding out four minutes into a load, from a traceback, is the alternative."""
    monkeypatch.setattr(model_profile, "supported_architectures",
                        lambda: frozenset({"LlamaForCausalLM", "Qwen3ForCausalLM"}))

    cache("org/known", {"config.json": LLAMA})
    assert model_profile.read("org/known").supported is True

    cache("org/exotic", {"config.json": {**LLAMA, "architectures": ["MadeUpForCausalLM"]}})
    exotic = model_profile.read("org/exotic")
    assert exotic.supported is False
    assert any("does not register MadeUpForCausalLM" in note for note in exotic.notes)


def test_an_ungenerated_registry_claims_nothing(cache, monkeypatch):
    """No generated list is an unknown answer, and an unknown answer must not
    read as a refusal."""
    monkeypatch.setattr(model_profile, "supported_architectures", frozenset)
    cache("org/model", {"config.json": LLAMA})
    profile = model_profile.read("org/model")
    assert profile.supported is None
    assert not any("does not register" in note for note in profile.notes)


def test_the_shard_index_is_authoritative_for_weight_bytes(cache, tmp_path):
    """A repo can ship tensors the loader skips — speculative-decoding heads,
    for one — which count on disk and never reach memory."""
    snapshot = cache("org/model", {
        "config.json": LLAMA,
        "model.safetensors.index.json": {
            "metadata": {"total_size": 7000, "total_parameters": 1234},
            "weight_map": {"a": "model.safetensors"},
        },
    })
    blob = tmp_path / "w.bin"
    blob.write_bytes(b"x" * 9000)
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "model_extra.safetensors").symlink_to(blob)

    profile = model_profile.read("org/model")
    assert profile.disk_bytes == 18000
    assert profile.weight_bytes == 7000
    assert profile.parameters == 1234


def test_without_an_index_the_files_are_the_answer(cache, tmp_path):
    snapshot = cache("org/model", {"config.json": LLAMA})
    blob = tmp_path / "w.bin"
    blob.write_bytes(b"x" * 4096)
    (snapshot / "model.safetensors").symlink_to(blob)

    profile = model_profile.read("org/model")
    assert profile.weight_bytes == 4096 == profile.disk_bytes
    assert profile.parameters is None


def test_a_custom_sampler_architecture_is_flagged(cache, monkeypatch):
    """vLLM's pipeline-parallel broadcast assumes the standard sampler's output.
    A model that supplies its own is not bound by that, and DiffusionGemma's
    returns int32 where the receiving rank allocates int64 — an assertion in the
    far rank's warmup, after every shard has been read."""
    monkeypatch.setattr(model_profile, "custom_sampler_architectures",
                        lambda: frozenset({"DiffusionGemmaForBlockDiffusion"}))
    monkeypatch.setattr(model_profile, "supported_architectures",
                        lambda: frozenset({"DiffusionGemmaForBlockDiffusion",
                                           "LlamaForCausalLM"}))

    cache("google/diffusion", {"config.json": {
        **LLAMA, "architectures": ["DiffusionGemmaForBlockDiffusion"]}})
    assert model_profile.read("google/diffusion").custom_sampler is True

    cache("org/plain", {"config.json": LLAMA})
    assert model_profile.read("org/plain").custom_sampler is False


def test_block_scaled_fp8_is_read_under_both_spellings(cache):
    """The same scale layout, and the same DeepGEMM path, written two ways: a
    plain fp8 checkpoint names weight_block_size, compressed-tensors names a
    per-group strategy. Qwen3.8-27B-FP8 is the first spelling and was missed by
    a rule that only knew the second."""
    cache("qwen/blocked", {"config.json": {**LLAMA, "quantization_config": {
        "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}}})
    assert model_profile.read("qwen/blocked").block_scaled_fp8 is True

    cache("org/ct-blocked", {"config.json": {**LLAMA, "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"g": {"weights": {"strategy": "block",
                                            "block_structure": [128, 128]}}}}}})
    assert model_profile.read("org/ct-blocked").block_scaled_fp8 is True


def test_scales_that_are_not_per_block_fp8_are_not_claimed(cache):
    """Turning DeepGEMM off costs throughput on the checkpoints it serves
    correctly, so this has to be the narrow answer rather than the safe one."""
    cache("org/per-tensor", {"config.json": {**LLAMA, "quantization_config": {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}}})
    assert model_profile.read("org/per-tensor").block_scaled_fp8 is False

    cache("org/channel", {"config.json": {**LLAMA, "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"g": {"weights": {"strategy": "channel", "num_bits": 8}}}}}})
    assert model_profile.read("org/channel").block_scaled_fp8 is False

    # Block is not exclusive to FP8, and a block-scaled int4 checkpoint fails in
    # its own way rather than this one.
    cache("org/int4-block", {"config.json": {**LLAMA, "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"g": {"weights": {"strategy": "block", "num_bits": 4}}}}}})
    assert model_profile.read("org/int4-block").block_scaled_fp8 is False

    cache("org/unquantised", {"config.json": LLAMA})
    assert model_profile.read("org/unquantised").block_scaled_fp8 is False
