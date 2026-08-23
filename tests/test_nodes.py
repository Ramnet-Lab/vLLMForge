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

def test_every_rank_gets_the_same_engine_and_a_different_rank_number():
    """Nothing verifies that the ranks agree. A model or a parallel size that
    differs between them surfaces as a shape mismatch deep in the rendezvous, if
    it surfaces at all — so they are built from one argv and differ only in
    which rank they are, and whether they serve HTTP."""
    from app import cluster

    head = cluster.NodeWiring(node=nodes.local_node(), interface="eth0", address="10.0.0.1")
    base = ["vllm", "serve", "org/m", "--port", "8010", "--gpu-memory-utilization", "0.3"]

    rank0 = cluster.rank_argv(base, 0, 2, head, 29500)
    rank1 = cluster.rank_argv(base, 1, 2, head, 29500)

    assert rank0[:len(base)] == base and rank1[:len(base)] == base
    assert "--headless" not in rank0, "rank 0 is the one that serves"
    assert "--headless" in rank1

    for argv in (rank0, rank1):
        assert argv[argv.index("--nnodes") + 1] == "2"
        assert argv[argv.index("--pipeline-parallel-size") + 1] == "2"
        assert argv[argv.index("--master-addr") + 1] == "10.0.0.1"
        assert argv[argv.index("--master-port") + 1] == "29500"
    assert rank0[rank0.index("--node-rank") + 1] == "0"
    assert rank1[rank1.index("--node-rank") + 1] == "1"


def test_rank_zero_keeps_the_plain_container_name():
    """Everything that looks a server up by container — logs, health, stop, the
    budget's self-exclusion, the foreign-container scan — knows only that name,
    and none of it should have to learn about pooling."""
    from app import cluster

    assert cluster.rank_container("llmd-vllm-7", 0) == "llmd-vllm-7"
    assert cluster.rank_container("llmd-vllm-7", 1) == "llmd-vllm-7-r1"
    assert cluster.rank_names("llmd-vllm-7", 3) == [
        "llmd-vllm-7", "llmd-vllm-7-r1", "llmd-vllm-7-r2"]


def test_a_rank_is_told_its_own_nodes_interface():
    """The interface carrying the fabric has a different name on each box, and
    settings.nccl_env() carries this machine's. Handing that to a peer points
    its NCCL at a device that is not there."""
    from app import cluster

    head = cluster.NodeWiring(node=nodes.local_node(), interface="enp1s0f0np0",
                              address="10.0.0.1")
    peer = cluster.NodeWiring(node=nodes.Node(name="node2", address="10.0.0.2"),
                              interface="enp1s0f1np1", address="10.0.0.2")

    base = {"NCCL_SOCKET_IFNAME": "enp1s0f0np0", "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
            "NCCL_IB_DISABLE": "0", "NCCL_IB_HCA": "rocep1s0f0", "HF_TOKEN": "keep-me"}

    env = cluster.rank_env(peer, head, base)
    assert env["NCCL_SOCKET_IFNAME"] == "enp1s0f1np1"
    assert env["GLOO_SOCKET_IFNAME"] == "enp1s0f1np1"
    assert env["NCCL_IB_DISABLE"] == "1", "the proven recipe disables IB"
    assert "NCCL_IB_HCA" not in env, "this machine's HCA name means nothing on a peer"
    assert env["HF_TOKEN"] == "keep-me", "everything not node-specific survives"


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
    # Otherwise this reaches a real `docker ps`, and the answer depends on what
    # happens to be running on the machine the suite is on.

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
async def test_pool_status_is_the_rank_containers_and_nothing_else(monkeypatch):
    """There is no cluster daemon to interrogate any more. An engine is exactly
    its ranks: all up means up, and one missing means dead however healthy rank
    0 looks, because the world size is fixed."""
    from app import cluster, docker_ctl

    asked = []
    down = {"llmd-vllm-9-r1"}
    # by_name falls back to this machine for an unknown name, which would
    # quietly report both ranks as living on the same box.
    known = {"local": nodes.local_node(),
             "node2": nodes.Node(name="node2", address="10.0.0.2", docker_host="ssh://node2")}
    monkeypatch.setattr(cluster.nodes, "by_name", lambda name: known[name])

    async def state(name, host=None):
        asked.append(name)
        running = name not in down
        return docker_ctl.ContainerState(
            name=name, exists=True, running=running,
            status="running" if running else "exited")

    monkeypatch.setattr(cluster.docker_ctl, "state", state)

    result = await cluster.status(["local", "node2"], container="llmd-vllm-9")
    assert asked == ["llmd-vllm-9", "llmd-vllm-9-r1"], "one container per rank, in rank order"
    assert result["running"] is False, "a missing far rank is a dead engine"
    assert result["nodes"] == 1 and result["expected"] == 2
    # The rank that failed is named, because the failures unique to multi-node
    # land there while rank 0 shows only a stall.
    assert "rank 1 on node2" in result["raw"]

    down.clear()
    healthy = await cluster.status(["local", "node2"], container="llmd-vllm-9")
    assert healthy["running"] is True and healthy["nodes"] == 2


# --- a peer's model cache -------------------------------------------------


def _fake_ssh(code: int, out: str):
    """Stand in for the one round trip node_cache makes."""
    async def run(host, script):
        return code, out
    return run


@pytest.fixture
def peer(monkeypatch):
    """A registered node2. by_name falls back to this machine for an unknown
    name, which would quietly turn every peer test into a local one."""
    node = nodes.Node(name="node2", address="10.0.0.2", docker_host="ssh://node2")
    monkeypatch.setattr(nodes, "by_name", lambda name: node)
    return node


@pytest.mark.anyio
async def test_a_peers_cache_scan_matches_the_local_shape(monkeypatch, peer):
    """The Models page renders one table for either node, so a peer's scan has
    to answer the same field names the local cache does."""
    from app import hf

    out = "\n".join([
        "R|models--google--gemma-4-31B-it|62578686995|21|1787369116",
        "R|datasets--tatsu-lab--alpaca|4096|3|1787000000",
        "R|garbage-without-enough-fields",
        "I|2 8192",
        "D|3648547196928 4031871553536",
    ])
    monkeypatch.setattr(nodes, "_ssh", _fake_ssh(0, out))
    payload = await hf.node_cache("node2")

    assert payload["ok"] and payload["node"] == "node2"
    assert payload["incomplete_files"] == 2 and payload["incomplete_bytes"] == 8192
    assert payload["disk"]["free_bytes"] == 3648547196928

    # Sorted by size, and the malformed line is dropped rather than guessed at.
    assert [r["repo_id"] for r in payload["repos"]] == [
        "google/gemma-4-31B-it", "tatsu-lab/alpaca"]
    model = payload["repos"][0]
    assert model["size_on_disk"] == 62578686995
    assert model["nb_files"] == 21
    assert model["kind"] == "model" and model["repo_type"] == "model"
    assert payload["repos"][1]["kind"] == "dataset"
    # What ssh cannot see is empty, never invented.
    assert model["revisions"] == [] and model["refs"] == {}
    assert payload["size_on_disk"] == 62578686995 + 4096


@pytest.mark.anyio
async def test_a_node_without_a_cache_directory_is_empty_not_broken(monkeypatch, peer):
    from app import hf

    monkeypatch.setattr(nodes, "_ssh", _fake_ssh(3, ""))
    payload = await hf.node_cache("node2")
    assert payload["ok"] and payload["repos"] == [] and payload["size_on_disk"] == 0
    assert "error" in payload  # says why, without claiming the node is broken


@pytest.mark.anyio
async def test_an_unreachable_node_reports_the_failure(monkeypatch, peer):
    from app import hf

    monkeypatch.setattr(nodes, "_ssh", _fake_ssh(255, "ssh: connect to host node2: no route"))
    payload = await hf.node_cache("node2")
    assert payload["ok"] is False and "no route" in payload["error"]


def test_one_repo_can_download_onto_two_nodes_at_once(monkeypatch):
    """The clash being guarded against is two writers on one cache, and a cache
    belongs to a node — so the same repo landing on two machines is fine."""
    from app import hf, jobs

    running = [{
        "id": "job-local", "status": "running",
        "spec": {"meta": {"repo_id": "org/model", "node": "local"}},
    }]
    monkeypatch.setattr(jobs.manager, "list", lambda *a, **k: running)

    assert hf._running_download("org/model") == "job-local"
    assert hf._running_download("org/model", "local") == "job-local"
    assert hf._running_download("org/model", "node2") is None
    assert hf._running_download("org/other", "local") is None


# --- the guard on a pooled launch -----------------------------------------


@pytest.mark.anyio
async def test_a_pooled_plan_judges_every_node_not_just_the_head(monkeypatch):
    """A pooled engine declares the same utilisation fraction on each machine it
    spans. Asking only the head is how a config that cannot start anywhere gets
    accepted — which is exactly the launch that failed with 'free memory on
    device cuda:0 (107.81/121.69 GiB) is less than desired (0.95, 115.6 GiB)'."""
    from app import cluster, safety

    wired = [nodes.Node(name="node1", address="10.0.0.1", docker_host=""),
             nodes.Node(name="node2", address="10.0.0.2", docker_host="ssh://node2")]
    monkeypatch.setattr(cluster, "_subnet_prefix", lambda: "10.0.0")
    monkeypatch.setattr(nodes, "by_name", lambda name: next(
        (n for n in wired if n.name == name), wired[0]))

    async def fake_wiring(node, prefix):
        return cluster.NodeWiring(node=node, interface="enp1s0", address=f"{prefix}.1")

    async def has_image(*a, **k):
        return True


    async def budget(node=None, exclude=None):
        # max_util and free_util are derived; 12% held back leaves 0.88 free.
        return safety.Budget(total_bytes=TOTAL, available_bytes=TOTAL,
                             free_bytes=TOTAL, reserve_bytes=int(TOTAL * 0.12))

    seen: list[str] = []

    async def check(util, *, replacing=None, params=None, node=None):
        seen.append(node.name)
        fits = (util or 0) <= 0.88
        return safety.Verdict(
            ok=fits, level="ok" if fits else "block",
            message="fits" if fits else f"0.95 of {node.name} needs more than is free",
            budget={}, requested_util=util or 0.0)

    monkeypatch.setattr(cluster, "wiring", fake_wiring)
    monkeypatch.setattr(cluster.docker_ctl, "image_exists", has_image)
    monkeypatch.setattr(cluster.safety, "current_budget", budget)
    monkeypatch.setattr(cluster.safety, "check_launch", check)

    greedy = await cluster.plan(["node1", "node2"], "", {"gpu_memory_utilization": 0.95})
    assert seen == ["node1", "node2"], "every node in the pool is asked"
    assert greedy["ok"] is False
    assert "node1" in greedy["reason"] and "node2" in greedy["reason"]
    assert greedy["free_util"] == pytest.approx(0.88, abs=0.01)

    seen.clear()
    modest = await cluster.plan(["node1", "node2"], "", {"gpu_memory_utilization": 0.80})
    assert modest["ok"] is True
    assert [n["verdict"]["ok"] for n in modest["nodes"]] == [True, True]

    # Without arguments there is nothing to judge, and the plan stays a plan.
    seen.clear()
    bare = await cluster.plan(["node1", "node2"], "")
    assert bare["ok"] is True and seen == []
    assert all(n["verdict"] is None for n in bare["nodes"])


@pytest.mark.anyio
async def test_a_pooled_plan_refuses_a_model_that_cannot_be_split(monkeypatch, peer):
    """The alternative is discovering it from an assertion in the far rank's
    warmup, four minutes in, with every shard already read."""
    from app import cluster, model_profile, safety

    wired = [nodes.Node(name="node1", address="10.0.0.1"),
             nodes.Node(name="node2", address="10.0.0.2", docker_host="ssh://node2")]
    monkeypatch.setattr(cluster, "_subnet_prefix", lambda: "10.0.0")
    monkeypatch.setattr(nodes, "by_name", lambda name: next(
        (n for n in wired if n.name == name), wired[0]))

    async def fake_wiring(node, prefix):
        return cluster.NodeWiring(node=node, interface="enp1s0", address=f"{prefix}.1")

    async def has_image(*a, **k):
        return True


    async def budget(node=None, exclude=None):
        return safety.Budget(total_bytes=TOTAL, available_bytes=TOTAL, free_bytes=TOTAL,
                             reserve_bytes=int(TOTAL * 0.12))

    async def no_missing(model, names):
        return []

    monkeypatch.setattr(cluster, "wiring", fake_wiring)
    monkeypatch.setattr(cluster.docker_ctl, "image_exists", has_image)
    monkeypatch.setattr(cluster.safety, "current_budget", budget)
    monkeypatch.setattr(cluster, "missing_model", no_missing)

    def profile(model):
        return model_profile.Profile(
            reference=model, found=True, custom_sampler=model == "org/diffusion",
            architectures=["DiffusionGemmaForBlockDiffusion"] if model == "org/diffusion"
            else ["LlamaForCausalLM"])

    monkeypatch.setattr(model_profile, "read", profile)

    # No args, so the memory guard is not consulted and the only thing that can
    # refuse is the model itself.
    refused = await cluster.plan(["node1", "node2"], "org/diffusion")
    assert refused["ok"] is False
    assert "supplies its own sampler" in refused["reason"]

    fine = await cluster.plan(["node1", "node2"], "org/ordinary")
    assert fine["ok"] is True


@pytest.mark.asyncio
async def test_a_pool_that_lands_two_ranks_on_one_box_is_refused(monkeypatch):
    """nodes.by_name answers "this machine" for a name it does not know, so a
    pool naming a peer that has since been removed silently resolves to two
    ranks on one box. They then rendezvous with themselves and both claim the
    full utilisation, and it surfaces as a memory problem rather than as the
    configuration error it is."""
    from app import cluster

    monkeypatch.setattr(cluster, "_subnet_prefix", lambda: "10.0.0")

    refused = await cluster.plan(["local", "a-peer-that-was-removed"])
    assert not refused["ok"]
    assert "more than one rank on local" in refused["reason"]
    assert "a-peer-that-was-removed is not a registered node" in refused["reason"]

    # The same name twice is the same mistake without the removed peer.
    twice = await cluster.plan(["local", "local"])
    assert not twice["ok"] and "more than one rank on local" in twice["reason"]


@pytest.mark.asyncio
async def test_a_pooled_restart_excludes_its_own_rank_on_every_node(monkeypatch):
    """The name to exclude is different on every machine: rank r runs as
    <base>-r<r> there. Under Ray the peer container ran `ray start`, which the
    guard does not recognise as an engine, so passing one name for the whole
    pool was harmless. A far rank runs `vllm serve --headless` and is a tenant
    like any other, so rank 0's name on every node would count each peer's own
    container against its own restart — commit f3622ff's bug, once per peer."""
    from app import cluster, safety

    excluded = {}

    async def check_launch(util, *, replacing=None, params=None, node=None):
        excluded[node.name] = replacing
        return safety.Verdict(ok=True, level="ok", message="", budget={})

    async def budget(node=None, exclude=None):
        return safety.Budget(total_bytes=TOTAL, available_bytes=TOTAL, free_bytes=TOTAL,
                             reserve_bytes=int(TOTAL * 0.12))

    async def fake_wiring(node, prefix):
        return cluster.NodeWiring(node=node, interface="enp1s0", address=f"{prefix}.1")

    async def has_image(*a, **k):
        return True

    async def none_missing(model, node_names):
        return []

    known = {"local": nodes.local_node(),
             "node2": nodes.Node(name="node2", address="10.0.0.2", docker_host="ssh://node2")}
    monkeypatch.setattr(cluster.nodes, "by_name", lambda name: known[name])
    monkeypatch.setattr(cluster, "_subnet_prefix", lambda: "10.0.0")
    monkeypatch.setattr(cluster, "wiring", fake_wiring)
    monkeypatch.setattr(cluster, "missing_model", none_missing)
    monkeypatch.setattr(cluster.docker_ctl, "image_exists", has_image)
    monkeypatch.setattr(cluster.safety, "current_budget", budget)
    monkeypatch.setattr(cluster.safety, "check_launch", check_launch)

    await cluster.plan(["local", "node2"], "org/m", {"gpu_memory_utilization": 0.3},
                       replacing="llmd-vllm-12")
    assert excluded == {"local": "llmd-vllm-12", "node2": "llmd-vllm-12-r1"}

    # A fresh launch excludes nothing anywhere.
    excluded.clear()
    await cluster.plan(["local", "node2"], "org/m", {"gpu_memory_utilization": 0.3})
    assert excluded == {"local": None, "node2": None}
