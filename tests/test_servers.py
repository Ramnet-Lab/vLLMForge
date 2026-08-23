"""Server definitions, and reading a foreign container's intent off its argv."""

from __future__ import annotations

import pytest

from app import db, servers, vllm_spec
from app.config import settings


async def _noop_publish(*a, **k):
    return None


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
    assert servers.container_path(Path("/srv/elsewhere/hf-cache")) == "/srv/elsewhere/hf-cache"


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

    from app.main import app

    row = servers.create_server({
        "name": f"pooled-{db.now():.6f}", "model": "org/model", "port": 8409,
        "pool_nodes": ["local", "node2"],
    })

    async def refused(pool, model="", args=None, replacing=None, image=""):
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

    async def no_image(pool, model="", args=None, replacing=None, image=""):
        return {"ok": False, "reason": "ray image is not built on: node2", "nodes": []}

    monkeypatch.setattr(cluster, "plan", no_image)
    client = TestClient(app)
    try:
        response = client.post(f"/api/servers/{row['id']}/start")
        assert response.status_code == 502
        assert "not built on" in response.json()["detail"]["message"]
    finally:
        servers.delete_server(int(row["id"]))


@pytest.mark.anyio
async def test_a_pool_that_really_lost_a_rank_stops_the_ranks_still_holding_memory(monkeypatch):
    """A pooled engine has a fixed world size: one rank down and the others do
    not carry on shorthanded — they abort or hang, and keep a full utilisation
    share of their machine each, for an engine that can never re-form.

    "Really" is the whole difficulty. One observation cannot tell a dead rank
    from a peer that did not answer, and start_pooled makes a pool legitimately
    partial for the seconds between creating its ranks, so acting on a single
    look means causing the outage this exists to prevent."""
    from app import docker_ctl, servers
    from app import nodes as node_registry

    stopped = []
    alive = {"llmd-vllm-77": True, "llmd-vllm-77-r1": True}
    unreachable = set()

    local = node_registry.local_node()
    node2 = node_registry.Node(name="node2", address="10.0.0.2", docker_host="ssh://node2")

    async def ps(prefix=None, *, all_containers=True, host=None):
        if host in unreachable:
            raise RuntimeError("ssh: connect to host node2 port 22: Connection refused")
        here = None if host is None else host
        return [{"Names": n} for n, h in
                (("llmd-vllm-77", None), ("llmd-vllm-77-r1", "ssh://node2")) if h == here]

    async def state(name, host=None):
        running = alive.get(name, False)
        return docker_ctl.ContainerState(name=name, exists=True, running=running,
                                         status="running" if running else "exited")

    async def stop(name, host=None):
        stopped.append(name)

    monkeypatch.setattr(servers, "list_servers",
                        lambda: [{"id": 77, "pool_nodes": ["local", "node2"], "node": "local"}])
    monkeypatch.setattr(servers.nodes, "registered", lambda: [local, node2])
    monkeypatch.setattr(servers.cluster.docker_ctl, "ps", ps)
    monkeypatch.setattr(servers.docker_ctl, "ps", ps)
    monkeypatch.setattr(servers.docker_ctl, "state", state)
    monkeypatch.setattr(servers.docker_ctl, "stop", stop)
    monkeypatch.setattr(servers.events.broker, "publish", _noop_publish)
    servers._partial_seen.clear()

    # A healthy pool is left alone however many times it is looked at.
    assert await servers.reap_partial_pools_once() == []
    assert await servers.reap_partial_pools_once() == []

    # A rank goes down. The FIRST pass must not act — this is the shape a launch
    # in progress has, and the shape an ssh blip has.
    alive["llmd-vllm-77-r1"] = False
    assert await servers.reap_partial_pools_once() == [], "one sighting is not a loss"
    assert stopped == []

    # Still gone on the next pass: now it is real.
    assert await servers.reap_partial_pools_once() == ["llmd-vllm-77"]
    assert sorted(stopped) == ["llmd-vllm-77", "llmd-vllm-77-r1"]

    # An unreachable peer is not a dead rank. `docker inspect` cannot tell them
    # apart, so the node is asked whether it is answering at all first.
    stopped.clear()
    servers._partial_seen.clear()
    alive.update({"llmd-vllm-77": True, "llmd-vllm-77-r1": True})
    unreachable.add("ssh://node2")
    assert await servers.reap_partial_pools_once() == []
    assert await servers.reap_partial_pools_once() == []
    assert stopped == [], "a peer that did not answer must never cost a healthy engine"

    # A pool that is entirely down was stopped on purpose.
    unreachable.clear()
    servers._partial_seen.clear()
    alive.update({"llmd-vllm-77": False, "llmd-vllm-77-r1": False})
    assert await servers.reap_partial_pools_once() == []
    assert await servers.reap_partial_pools_once() == []
    assert stopped == []


# --- two engines ---------------------------------------------------------

def test_an_existing_row_reads_back_as_vllm_and_keeps_its_container_name():
    """The compatibility guarantee for every row already in an installed
    database. The engine name IS the container kind, so a row that predates the
    column has to yield the llmd-vllm-<id> name its running container already
    has — otherwise the upgrade renames a live process out from under stop()."""
    row = servers.create_server({"name": f"legacy-{db.now():.6f}", "model": "org/m",
                                 "port": 8421})
    try:
        fetched = servers.get_server(int(row["id"]))
        assert fetched["engine"] == "vllm"
        assert servers.container_name(fetched) == f"llmd-vllm-{row['id']}"
        assert servers.engine_for(fetched).name == "vllm"
    finally:
        servers.delete_server(int(row["id"]))


def test_a_llamacpp_row_builds_a_llama_server_command():
    row = servers.create_server({
        "name": f"gguf-{db.now():.6f}", "engine": "llamacpp", "port": 8422,
        "model": "/hf/hub/models--x/snapshots/abc/m-Q4_K_M.gguf",
        "served_name": "my-model", "args": {"n_gpu_layers": 40, "ctx_size": 8192},
    })
    try:
        fetched = servers.get_server(int(row["id"]))
        assert servers.container_name(fetched) == f"llmd-llamacpp-{row['id']}"
        command = servers.build_command(fetched)
        assert command[0] == "llama-server"
        # served_name becomes each engine's own flag for it.
        assert command[command.index("--alias") + 1] == "my-model"
        assert command[command.index("--n-gpu-layers") + 1] == "40"
        assert "--served-model-name" not in command
        # No vLLM-only environment leaks in.
        env = servers.build_env(fetched)
        assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in env
        assert env["HF_HOME"] == "/hf" and env["LLAMA_CACHE"] == "/hf/llamacpp"
        # And the image follows the engine rather than defaulting to vLLM's.
        assert servers.image_of(fetched) == settings.llamacpp_image
    finally:
        servers.delete_server(int(row["id"]))


def test_switching_engine_keeps_the_other_engines_arguments():
    """`args` stays authoritative for the active engine; the stash is the
    editor's memory of the rest. Switching and switching back must not throw
    away what was typed either time."""
    row = servers.create_server({
        "name": f"switch-{db.now():.6f}", "model": "org/m", "port": 8423,
        "args": {"gpu_memory_utilization": 0.4},
    })
    try:
        sid = int(row["id"])
        assert servers.get_server(sid)["args_by_engine"] == {
            "vllm": {"gpu_memory_utilization": 0.4}}

        servers.update_server(sid, {"engine": "llamacpp", "model": "/hf/m.gguf",
                                    "args": {"n_gpu_layers": 40}})
        after = servers.get_server(sid)
        assert after["engine"] == "llamacpp"
        assert after["args"] == {"n_gpu_layers": 40}
        assert after["args_by_engine"]["vllm"] == {"gpu_memory_utilization": 0.4}

        # Back again with no args named: the stash answers, not the other
        # engine's flags carried forward.
        servers.update_server(sid, {"engine": "vllm", "model": "org/m"})
        back = servers.get_server(sid)
        assert back["args"] == {"gpu_memory_utilization": 0.4}
        assert back["args_by_engine"]["llamacpp"] == {"n_gpu_layers": 40}
    finally:
        servers.delete_server(int(row["id"]))


def test_a_pooled_llamacpp_definition_is_refused_at_the_api():
    """Pooling is vLLM's pipeline-parallel path. app/cluster.py has no llama.cpp
    analogue, so the refusal is at the boundary rather than a half-port."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.post("/api/servers", json={
        "name": f"pooled-gguf-{db.now():.6f}", "engine": "llamacpp",
        "model": "/hf/m.gguf", "port": 8424, "pool_nodes": ["local", "node2"]})
    assert response.status_code == 422
    assert "cannot be pooled" in response.text


def test_a_llamacpp_args_dict_is_validated_against_llama_cpp():
    """Both the create and the patch path. The patch one has to load the row
    first — without it, every edit to a llama.cpp server came back as 'unknown
    parameter for this vLLM build'."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    created = client.post("/api/servers", json={
        "name": f"validated-{db.now():.6f}", "engine": "llamacpp",
        "model": "/hf/m.gguf", "port": 8425, "args": {"n_gpu_layers": 40}})
    assert created.status_code == 201, created.text
    row = created.json()
    try:
        patched = client.patch(f"/api/servers/{row['id']}", json={"args": {"ctx_size": 4096}})
        assert patched.status_code == 200, patched.text
        # A vLLM flag on a llama.cpp row is refused, not silently accepted.
        rejected = client.patch(f"/api/servers/{row['id']}",
                                json={"args": {"gpu_memory_utilization": 0.5}})
        assert rejected.status_code == 422
        assert "llama.cpp build" in rejected.text
    finally:
        servers.delete_server(int(row["id"]))


def test_the_schema_endpoint_answers_per_engine():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    # No parameter returns exactly what it always returned.
    assert client.get("/api/servers/schema").json()["engine"] == "vllm"
    llama = client.get("/api/servers/schema?engine=llamacpp").json()
    assert llama["engine"] == "llamacpp" and llama["featured"]
    assert client.get("/api/servers/suggest?engine=llamacpp").json()["image"] \
        == settings.llamacpp_image
    names = [e["name"] for e in client.get("/api/servers/engines").json()["engines"]]
    assert names == ["vllm", "llamacpp"]
