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


def test_path_options_are_container_paths_not_host_paths(tmp_path, monkeypatch):
    """vLLM runs with the cache at /hf and outputs at /outputs. A host path here
    starts an engine that then cannot find its file, minutes later, in a log."""
    from app import servers
    from app.config import settings

    produced = settings.output_dir / "run-xyz" / "chat_template.jinja"
    assert servers.container_path(produced).startswith("/outputs/")
    assert not servers.container_path(produced).startswith(str(settings.output_dir))


def test_the_walk_is_bounded(tmp_path):
    # The model cache holds tens of thousands of blobs; an unbounded walk would
    # enumerate every one of them for no benefit.
    deep = tmp_path
    for level in range(8):
        deep = deep / f"level{level}"
    deep.mkdir(parents=True)
    (deep / "deep.jinja").write_text("x")
    (tmp_path / "shallow.jinja").write_text("x")

    found = catalog._walk(tmp_path, lambda p: p.suffix == ".jinja")
    names = {p.name for p in found}
    assert "shallow.jinja" in names
    assert "deep.jinja" not in names, "the walk should stop before MAX_DEPTH"


def test_an_adapter_is_offered_in_the_form_vllm_wants():
    # --lora-modules takes name=path, so the option has to arrive that way or the
    # user has to hand-assemble it, which is the typing this exists to avoid.
    entry = catalog._entry("myrun=/outputs/x/adapter", "myrun", kind="path")
    assert "=" in entry["value"]
    name, _, path = entry["value"].partition("=")
    assert name and path.startswith("/outputs/")
