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
