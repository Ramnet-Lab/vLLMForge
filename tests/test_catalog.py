"""What the model pickers are allowed to offer.

Offering something that cannot be loaded — a dataset, a bare LoRA adapter, a
run that failed before it wrote weights — costs the user a container start and
several minutes before the failure surfaces, so the filtering is worth pinning.
"""

from __future__ import annotations

import json

import pytest

from app import catalog


def _write_model(path, *, weights="model.safetensors"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"architectures": ["X"]}))
    (path / weights).write_bytes(b"\0" * 16)
    return path


def test_a_directory_is_a_model_only_with_config_and_weights(tmp_path):
    assert not catalog._is_model_dir(tmp_path / "missing")

    bare = tmp_path / "bare"
    bare.mkdir()
    assert not catalog._is_model_dir(bare), "an empty directory is not a model"

    config_only = tmp_path / "config-only"
    config_only.mkdir()
    (config_only / "config.json").write_text("{}")
    assert not catalog._is_model_dir(config_only), "config.json alone is not loadable"

    weights_only = tmp_path / "weights-only"
    weights_only.mkdir()
    (weights_only / "model.safetensors").write_bytes(b"\0")
    assert not catalog._is_model_dir(weights_only), "weights with no config are not loadable"

    assert catalog._is_model_dir(_write_model(tmp_path / "real"))
    assert catalog._is_model_dir(_write_model(tmp_path / "gguf", weights="m.gguf"))


def test_datasets_and_adapters_are_not_offered_as_models():
    local = {
        "ok": True,
        "repos": [
            {"repo_id": "org/model", "repo_type": "model", "size_on_disk": 2 << 30,
             "revisions": [{"kind": "model"}]},
            {"repo_id": "org/data", "repo_type": "dataset", "size_on_disk": 1 << 20,
             "revisions": [{"kind": "model"}]},
            {"repo_id": "org/lora", "repo_type": "model", "size_on_disk": 1 << 20,
             "revisions": [{"kind": "adapter"}]},
        ],
    }
    offered = [entry["value"] for entry in catalog._cached_models(local)]
    assert offered == ["org/model"]


def test_a_broken_cache_yields_nothing_rather_than_raising():
    assert catalog._cached_models({"ok": False, "error": "permission denied"}) == []


@pytest.mark.asyncio
async def test_the_catalogue_survives_a_cache_that_cannot_be_read(monkeypatch):
    async def broken():
        raise OSError("cache unreadable")

    monkeypatch.setattr(catalog.hf, "local_models", broken)
    payload = await catalog.loadable_models()
    assert payload["cache_ok"] is False
    assert "unreadable" in payload["cache_error"]
    # Job outputs are still offered — a bad cache must not empty the whole picker.
    assert "groups" in payload
