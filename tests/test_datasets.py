"""Dataset support reuses the model plumbing, so the parts that differ — the
API segment, the cache prefix, and what makes a dataset trainable — are what
need pinning."""

from __future__ import annotations

import types

import pytest

from app import hf


def test_the_two_repo_types_map_to_their_hub_segments():
    assert hf._repo_type("model") == ("models", "models--")
    assert hf._repo_type("dataset") == ("datasets", "datasets--")
    with pytest.raises(hf.HubError):
        hf._repo_type("space")


def test_a_dataset_resolves_to_its_own_cache_directory(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    (hub / "datasets--org--set").mkdir(parents=True)
    (hub / "models--org--set").mkdir(parents=True)
    monkeypatch.setattr(hf, "settings", types.SimpleNamespace(hf_cache=tmp_path))

    assert hf.cache_dir_for("org/set", "dataset").name == "datasets--org--set"
    assert hf.cache_dir_for("org/set", "model").name == "models--org--set"


def test_deleting_a_dataset_cannot_escape_the_cache_either(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setattr(hf, "settings", types.SimpleNamespace(hf_cache=tmp_path))
    # The same rm -rf as a root container runs for models.
    for escape in ("../../etc", "/etc/passwd", "..", "org/set/extra"):
        with pytest.raises(hf.HubError) as caught:
            hf.cache_dir_for(escape, "dataset")
        assert caught.value.status == 400


def test_a_model_id_is_not_found_under_the_dataset_prefix(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    (hub / "models--org--only").mkdir(parents=True)
    monkeypatch.setattr(hf, "settings", types.SimpleNamespace(hf_cache=tmp_path))
    with pytest.raises(hf.HubError) as caught:
        hf.cache_dir_for("org/only", "dataset")
    assert caught.value.status == 404


def columns(*names):
    return [{"name": name, "type": "string"} for name in names]


def test_conversational_columns_win():
    shape = hf._training_shape(columns("messages", "text"))
    assert shape["format"] == "messages" and shape["field"] == "messages"


def test_a_prerendered_text_column_beats_the_instruction_pair():
    # alpaca ships instruction/input/output *and* a rendered `text`; training on
    # `text` is right, training on `output` would throw the questions away.
    shape = hf._training_shape(columns("instruction", "input", "output", "text"))
    assert shape["level"] == "ok" and shape["field"] == "text"


def test_without_a_text_column_the_pair_is_flagged_rather_than_guessed():
    shape = hf._training_shape(columns("instruction", "input", "output"))
    assert shape["level"] == "warn"
    assert shape["format"] == "instruction"
    assert "only the answers are learned" in shape["note"]


def test_an_unrecognisable_dataset_says_so():
    shape = hf._training_shape(columns("audio", "label"))
    assert shape["level"] == "warn" and shape["field"] == ""


@pytest.mark.asyncio
async def test_a_missing_datasets_server_entry_is_not_an_error(monkeypatch):
    async def unavailable(url, params=None, *, label=""):
        raise hf.HubError("not found", 404)

    monkeypatch.setattr(hf, "_get", unavailable)
    # Not every dataset is auto-converted to parquet, and a dataset that is not
    # is still perfectly downloadable — it must not fail the detail call.
    assert await hf._datasets_server("/splits", {"dataset": "x"}) == {}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"dtype": "string", "_type": "Value"}, "string"),
        ({"_type": "Image"}, "Image"),
        # A conversational column is a *list* of struct definitions, not a dict.
        # Assuming a dict here broke every chat dataset — which is precisely the
        # kind a fine-tune is most likely to want.
        ([{"content": {"dtype": "string"}, "role": {"dtype": "string"}}],
         "list<struct{content, role}>"),
        ([], "list"),
        (None, ""),
    ],
)
def test_feature_types_survive_both_shapes(raw, expected):
    assert hf._feature_type(raw) == expected


def test_a_conversational_column_is_recognised_through_its_list_type():
    columns = [
        {"name": "messages", "type": hf._feature_type(
            [{"content": {"dtype": "string"}, "role": {"dtype": "string"}}])},
        {"name": "source", "type": "string"},
    ]
    shape = hf._training_shape(columns)
    assert shape["level"] == "ok"
    assert shape["format"] == "messages" and shape["field"] == "messages"
