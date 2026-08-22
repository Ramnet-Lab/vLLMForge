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
