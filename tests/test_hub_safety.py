"""The cache-delete path runs `rm -rf` as root inside a container.

It is the only place in the dashboard where a request can destroy data outside
its own state directory, so the guard in front of it gets its own test file.
"""

from __future__ import annotations

import types

import pytest

from app import hf

TRAVERSALS = [
    "../../etc",
    "../etc",
    "a/../../b",
    "/etc/passwd",
    "..",
    ".",
    "",
    "models--x/../..",
    "org/../../../root",
    "org/model/../..",
    "..%2F..%2Fetc",
    "org//model",
    "-rf",
    "org/model;rm -rf /",
    "org/model\n../../x",
    ".hidden/model",
    "org/.hidden",
    "org/model/extra",
]


@pytest.mark.parametrize("repo_id", TRAVERSALS)
def test_a_repo_id_that_is_not_one_never_becomes_a_path(repo_id):
    with pytest.raises(hf.HubError) as caught:
        hf.cache_dir_for(repo_id)
    assert caught.value.status == 400


def test_a_real_repo_resolves_inside_the_hub(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    (hub / "models--org--model").mkdir(parents=True)
    monkeypatch.setattr(hf, "settings", types.SimpleNamespace(hf_cache=tmp_path))
    assert hf.cache_dir_for("org/model") == (hub / "models--org--model").resolve()


def test_a_symlinked_cache_entry_is_refused(tmp_path, monkeypatch):
    # A directory inside the hub whose name is valid but which points somewhere
    # else entirely would otherwise hand `rm -rf` a target outside the cache.
    hub = tmp_path / "hub"
    hub.mkdir(parents=True)
    elsewhere = tmp_path / "precious"
    elsewhere.mkdir()
    (hub / "models--org--model").symlink_to(elsewhere, target_is_directory=True)

    monkeypatch.setattr(hf, "settings", types.SimpleNamespace(hf_cache=tmp_path))
    with pytest.raises(hf.HubError) as caught:
        hf.cache_dir_for("org/model")
    assert caught.value.status == 400
    assert elsewhere.is_dir()


def test_a_repo_that_is_not_cached_is_a_404(tmp_path, monkeypatch):
    (tmp_path / "hub").mkdir(parents=True)
    monkeypatch.setattr(hf, "settings", types.SimpleNamespace(hf_cache=tmp_path))
    with pytest.raises(hf.HubError) as caught:
        hf.cache_dir_for("org/absent")
    assert caught.value.status == 404
