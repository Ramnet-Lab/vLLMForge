"""Placing work on a peer.

A cluster here is a few identical boxes on one subnet, reached with
`docker -H ssh://peer`. The parts worth pinning are the ones where a local
assumption would leak into a remote decision.
"""

from __future__ import annotations

import pytest

from app import docker_ctl, nodes, safety

GIB = 1024 ** 3
TOTAL = 130_663_006_208


def test_docker_argv_targets_the_right_daemon():
    assert docker_ctl.docker_argv(None, "ps") == ["docker", "ps"]
    assert docker_ctl.docker_argv("ssh://node2", "ps") == ["docker", "-H", "ssh://node2", "ps"]


def test_a_remote_run_keeps_the_image_before_the_command():
    argv = docker_ctl.build_run_argv(
        name="llmd-vllm-1", image="img", command=["vllm", "serve", "m"], host="ssh://node2"
    )
    assert argv[:4] == ["docker", "-H", "ssh://node2", "run"]
    assert argv.index("img") < argv.index("vllm")


def test_remove_still_forces_on_the_right_host():
    argv = docker_ctl.docker_argv("ssh://node2", "rm", "c")
    argv.insert(-1, "-f")
    assert argv == ["docker", "-H", "ssh://node2", "rm", "-f", "c"]


def test_an_unknown_node_falls_back_to_this_machine():
    # A server saved before the cluster existed has no node recorded, and must
    # keep working rather than failing to resolve.
    assert nodes.by_name(None).is_local
    assert nodes.by_name("").is_local
    assert nodes.by_name("a-node-that-was-removed").is_local


def test_the_local_node_has_no_docker_host():
    local = nodes.local_node()
    assert local.is_local and local.docker_host is None


def test_registering_a_peer_is_idempotent():
    from app import db

    previous = db.get_setting("nodes", [])
    try:
        nodes.add("probe-peer", address="10.0.0.9")
        nodes.add("probe-peer", address="10.0.0.9")
        names = [n.name for n in nodes.registered()]
        assert names.count("probe-peer") == 1
        assert nodes.by_name("probe-peer").docker_host == "ssh://probe-peer"
        nodes.remove("probe-peer")
        assert "probe-peer" not in [n.name for n in nodes.registered()]
    finally:
        db.set_setting("nodes", previous)


@pytest.mark.asyncio
async def test_a_peers_budget_never_counts_this_machines_gpu(monkeypatch):
    """nvidia-smi reports on THIS box. Counting it against a peer made an idle
    peer look half full and refused launches it had ample room for."""
    called = []

    async def local_processes():
        called.append(True)
        return [type("P", (), {"used_bytes": 60 * GIB})()]

    async def no_containers(prefix=None, *, all_containers=True, host=None):
        return []

    async def remote_memory(node):
        return TOTAL, 118 * GIB, 100 * GIB

    monkeypatch.setattr(safety, "read_gpu_processes", local_processes)
    monkeypatch.setattr(safety.docker_ctl, "ps", no_containers)
    monkeypatch.setattr(nodes, "_remote_memory", remote_memory)

    peer = nodes.Node(name="peer", address="10.0.0.9", docker_host="ssh://peer")
    budget = await safety.current_budget(node=peer)
    assert budget.measured_gpu_bytes == 0
    assert budget.occupied_bytes == 0
    assert not called, "nvidia-smi must not be consulted for a peer"

    local = await safety.current_budget(node=nodes.local_node())
    assert local.measured_gpu_bytes == 60 * GIB


# --- model sync ----------------------------------------------------------

def test_rsync_progress_is_parsed():
    from app import sync

    line = "  1,234,567,890  42%  512.34MB/s    0:01:23 (xfr#12, to-chk=30/94)"
    update = sync.parse_sync(line, {})
    assert update["transferred_bytes"] == 1_234_567_890
    assert update["percent"] == 42.0
    assert update["speed_bps"] == 512.34 * 10 ** 6
    assert update["files_total"] == 94 and update["files_done"] == 64


def test_a_line_without_progress_is_left_as_log():
    from app import sync

    assert sync.parse_sync("sending incremental file list", {}) is None
    assert sync.parse_sync("total size is 12,512,651  speedup is 1.00", {})["percent"] == 100.0


def test_the_cache_directory_name_matches_huggingface_layout():
    from app import sync

    assert sync.repo_dir_name("org/model") == "models--org--model"
    assert sync.repo_dir_name("org/set", "dataset") == "datasets--org--set"


def test_xet_bookkeeping_is_excluded_from_the_copy():
    """trees/ and .locks/ are written 0600 by the root container that downloaded
    the model, so a non-root rsync cannot read them — and neither is needed to
    load a model. Including them is the difference between exit 0 and exit 23."""
    from pathlib import Path

    source = Path("app/sync.py").read_text()
    assert "--exclude=trees/" in source
    assert "--exclude=.locks/" in source


# --- pooled engines ------------------------------------------------------

def test_the_engine_runs_inside_the_ray_head():
    """A Ray driver needs the raylet session directory, which lives inside
    whichever container ran `ray start`. Two containers on one host do not share
    /tmp/ray, so a separate engine container fails with 'No node info found
    matching attributes' — which is exactly what happened before this shape."""
    from app import cluster

    head = cluster.NodeWiring(node=nodes.local_node(), interface="eth0", address="10.0.0.1")
    command = cluster.head_command(head, ["vllm", "serve", "m", "--port", "8000"])
    assert command[0] == "-lc"
    script = command[1]
    assert script.startswith("ray start --head --node-ip-address=10.0.0.1")
    # exec, so the engine becomes PID 1's child and the container dies with it.
    assert " && exec vllm serve m --port 8000" in script


def test_a_pool_needs_more_than_one_node():
    import asyncio

    from app import cluster

    result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        cluster.plan(["local"])
    )
    assert not result["ok"] and "two nodes" in result["reason"]


def test_pooled_is_only_true_for_more_than_one_node():
    from app import servers

    assert not servers.is_pooled({"pool_nodes": []})
    assert not servers.is_pooled({"pool_nodes": ["local"]})
    assert servers.is_pooled({"pool_nodes": ["local", "node2"]})


def test_a_pooled_servers_containers_live_on_its_head():
    from app import db, servers

    previous = db.get_setting("nodes", [])
    try:
        nodes.add("peer-x", address="10.0.0.9")
        # An ordinary server runs on its own node...
        assert servers.host_of({"node": "peer-x", "pool_nodes": []}) == "ssh://peer-x"
        # ...but a pooled one runs where its Ray head is: the first pool entry.
        # The other nodes hold shards, not the HTTP frontend.
        assert servers.host_of({"node": "peer-x", "pool_nodes": ["local", "peer-x"]}) is None
        assert servers.host_of(
            {"node": "local", "pool_nodes": ["peer-x", "local"]}
        ) == "ssh://peer-x"
    finally:
        db.set_setting("nodes", previous)


@pytest.mark.asyncio
async def test_the_plan_reports_a_model_missing_from_a_node(monkeypatch):
    """Each node loads its own shard from its own disk, so a model present only
    on the head fails minutes into a launch. The form needs to know first."""
    from app import cluster

    async def only_here(model, node_names):
        return [n for n in node_names if n != "local"]

    async def fine(*a, **k):
        return True

    async def budget(node=None, exclude=None):
        return safety.Budget(total_bytes=TOTAL, available_bytes=TOTAL,
                             reserve_bytes=32 * GIB, warn_reserve_bytes=38 * GIB)

    monkeypatch.setattr(cluster, "missing_model", only_here)
    monkeypatch.setattr(cluster, "_subnet_prefix", lambda: "10.0.0")
    monkeypatch.setattr(cluster.docker_ctl, "image_exists", fine)
    monkeypatch.setattr(cluster.safety, "current_budget", budget)

    async def wired(node, prefix):
        return cluster.NodeWiring(node=node, interface="eth0", address="10.0.0.1")

    monkeypatch.setattr(cluster, "wiring", wired)

    from app import db

    previous = db.get_setting("nodes", [])
    try:
        nodes.add("peer-y", address="10.0.0.2")
        result = await cluster.plan(["local", "peer-y"], "org/model")
        # A model missing from a node is work, not a refusal: the launch copies
        # it over the cluster link before starting the engine. The plan has to
        # say which nodes need it and that it will be handled.
        assert result["ok"], "a missing model must not block the plan"
        assert result["missing_model_on"] == ["peer-y"]
        assert "will be copied to peer-y" in result["will_sync"]

        clean = await cluster.plan(["local", "peer-y"], "")
        assert clean["ok"] and not clean["will_sync"]
    finally:
        db.set_setting("nodes", previous)


@pytest.mark.asyncio
async def test_pool_status_asks_the_engine_because_it_is_the_head(monkeypatch):
    from app import cluster, docker_ctl

    seen = {}

    async def state(name, host=None):
        return docker_ctl.ContainerState(name=name, exists=True, status="running", running=True)

    async def run(argv, check=True):
        seen["argv"] = argv
        return 0, "node_aaa\nnode_bbb\n", ""

    monkeypatch.setattr(cluster.docker_ctl, "state", state)
    monkeypatch.setattr(cluster.docker_ctl, "_run", run)

    result = await cluster.status(["local", "node2"], container="llmd-vllm-9")
    # There is no standalone head container any more; the engine is the head.
    assert "llmd-vllm-9" in seen["argv"] and "llmd-ray-head" not in seen["argv"]
    assert result["nodes"] == 2 and result["expected"] == 2
