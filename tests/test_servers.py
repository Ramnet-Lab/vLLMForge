"""Server definitions, and reading a foreign container's intent off its argv."""

from __future__ import annotations

import pytest

from app import db, servers, vllm_spec


@pytest.fixture
def a_server():
    row = servers.create_server(
        {
            "name": f"test-{db.now():.6f}",
            "model": "org/model",
            "port": 8123,
            "args": {"gpu_memory_utilization": 0.3, "max_num_seqs": 4},
        }
    )
    yield row
    servers.delete_server(int(row["id"]))


def test_round_trip(a_server):
    fetched = servers.get_server(int(a_server["id"]))
    assert fetched["model"] == "org/model"
    assert fetched["args"]["gpu_memory_utilization"] == 0.3


def test_served_name_falls_through_to_the_flag(a_server):
    servers.update_server(int(a_server["id"]), {"served_name": "shortname"})
    command = servers.build_command(servers.get_server(int(a_server["id"])))
    assert "--served-model-name" in command
    assert command[command.index("--served-model-name") + 1] == "shortname"


def test_enabling_lora_opts_into_runtime_updates(a_server):
    servers.update_server(int(a_server["id"]), {"args": {"enable_lora": True}})
    env = servers.build_env(servers.get_server(int(a_server["id"])))
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "1"
    assert env["HF_HOME"] == "/hf"


def test_suggested_port_avoids_what_is_already_spoken_for(a_server):
    assert servers.suggest_port() not in (8000, 8001, 8080, 8265, 8123)


@pytest.mark.parametrize(
    "command,port,model",
    [
        (["vllm", "serve", "org/m", "--port", "8000"], 8000, "org/m"),
        (["vllm", "serve", "org/m", "--port=8002"], 8002, "org/m"),
        (["vllm", "serve", "--port", "8003"], 8003, ""),
        (["vllm", "serve", "org/m"], None, "org/m"),
    ],
)
def test_reading_a_foreign_container_s_argv(command, port, model):
    assert servers._port_from_command(command) == port
    assert servers._model_from_command(command) == model


def test_metrics_parsing_keeps_the_labelled_series_apart():
    text = "\n".join(
        [
            "# HELP vllm:num_requests_running Number of requests currently running.",
            "# TYPE vllm:num_requests_running gauge",
            'vllm:num_requests_running{engine="0",model_name="m"} 3.0',
            'vllm:prompt_tokens_total{engine="0",model_name="m"} 100.0',
            'vllm:prompt_tokens_total{engine="1",model_name="m"} 50.0',
            "garbage line",
        ]
    )
    parsed = servers.parse_metrics(text)
    assert parsed["selected"]["vllm:num_requests_running"] == 3.0
    # Counters from several engines describe one server, so they sum.
    assert parsed["selected"]["vllm:prompt_tokens_total"] == 150.0


def test_the_gpu_util_helper_tolerates_junk():
    assert vllm_spec.gpu_memory_utilization({"gpu_memory_utilization": "0.5"}) == 0.5
    assert vllm_spec.gpu_memory_utilization({"gpu_memory_utilization": "auto"}) is None
    assert vllm_spec.gpu_memory_utilization({}) is None


def test_a_job_s_output_path_is_translated_for_the_container():
    from pathlib import Path

    from app.config import settings

    produced = settings.output_dir / "heretic-qwen-abc123" / "out"
    assert servers.container_path(produced) == "/outputs/heretic-qwen-abc123/out"
    assert servers.container_path(settings.output_dir) == "/outputs"
    # A path the server container does not mount is left alone rather than
    # silently rewritten into something that does not exist.
    assert servers.container_path(Path("/home/user/models/hf-cache")) == "/home/user/models/hf-cache"


def test_a_token_stored_from_the_ui_reaches_server_containers():
    # A stored token that authenticated a download but not a server launched
    # from the same UI is a difference nobody could predict.
    from app import db, hf

    previous = db.get_setting("hf_token", "")
    try:
        db.set_setting("hf_token", "hf_probe")
        assert hf.token() == "hf_probe"
        assert servers.build_env({"args": {}, "env": {}})["HF_TOKEN"] == "hf_probe"
    finally:
        db.set_setting("hf_token", previous)


def test_renaming_onto_a_taken_name_is_a_conflict_not_a_crash():
    """Names are unique in the table. Create answered 409 and rename did not,
    so it reached sqlite as an IntegrityError and came back a 500 — and a name
    derived from the model is exactly the name a second server wants."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    first = client.post("/api/servers", json={
        "name": "qwen3-chat", "model": "org/qwen3", "port": 8401}).json()
    second = client.post("/api/servers", json={
        "name": "qwen3-chat-2", "model": "org/qwen3", "port": 8402}).json()

    clash = client.patch(f"/api/servers/{second['id']}", json={"name": "qwen3-chat"})
    assert clash.status_code == 409
    assert "already exists" in clash.json()["detail"]

    # Renaming a server to the name it already has is not a clash with itself.
    same = client.patch(f"/api/servers/{first['id']}", json={"name": "qwen3-chat"})
    assert same.status_code == 200

    for row in (first, second):
        servers.delete_server(int(row["id"]))


def test_a_pooled_refusal_is_not_reported_as_a_missing_server(monkeypatch):
    """The bug this pins: start_pooled returned a failure with no `safety` key,
    the route did `**result["safety"]`, and the route's own `except KeyError`
    turned that into 404 "no such server" — for a server that plainly existed.
    A launch refused for running out of memory came back as not found."""
    from fastapi.testclient import TestClient

    from app import servers as svc
    from app.main import app

    row = servers.create_server({
        "name": f"pooled-{db.now():.6f}", "model": "org/model", "port": 8409,
        "pool_nodes": ["local", "node2"],
    })

    async def refused(pool, model="", args=None):
        return {
            "ok": False,
            "reason": "local: 0.92 would reserve more than is free",
            "free_util": 0.73,
            "nodes": [{"name": "local", "verdict": {"ok": False, "level": "block",
                                                    "message": "too much", "requested_util": 0.92,
                                                    "requested_bytes": 1, "budget": {}}},
                      {"name": "node2", "verdict": {"ok": False, "level": "block",
                                                   "message": "too much", "requested_util": 0.92,
                                                   "requested_bytes": 1, "budget": {}}}],
        }

    from app import cluster

    monkeypatch.setattr(cluster, "plan", refused)
    client = TestClient(app)
    try:
        response = client.post(f"/api/servers/{row['id']}/start")
        # Not 404, and not a bare 502: the machines cannot take it.
        assert response.status_code == 409, response.text
        detail = response.json()["detail"]
        assert detail["level"] == "block"
        assert "would reserve more than is free" in detail["message"]
        assert detail["suggested_util"] == 0.73
        assert [n["name"] for n in detail["nodes"]] == ["local", "node2"]

        # A server id that really is missing still answers 404.
        assert client.post("/api/servers/999999/start").status_code == 404
    finally:
        servers.delete_server(int(row["id"]))


def test_an_environment_failure_stays_a_502(monkeypatch):
    """A missing image is not the host declining; it is something to go fix."""
    from fastapi.testclient import TestClient

    from app import cluster
    from app.main import app

    row = servers.create_server({
        "name": f"pooled-{db.now():.6f}", "model": "org/model", "port": 8410,
        "pool_nodes": ["local", "node2"],
    })

    async def no_image(pool, model="", args=None):
        return {"ok": False, "reason": "ray image is not built on: node2", "nodes": []}

    monkeypatch.setattr(cluster, "plan", no_image)
    client = TestClient(app)
    try:
        response = client.post(f"/api/servers/{row['id']}/start")
        assert response.status_code == 502
        assert "not built on" in response.json()["detail"]["message"]
    finally:
        servers.delete_server(int(row["id"]))
