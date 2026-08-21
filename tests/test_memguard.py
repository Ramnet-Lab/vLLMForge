"""The watchdog kills containers. What it will and will not touch is worth
pinning down, because the blast radius of a mistake here is the user's work."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from app import docker_ctl, memguard, safety, telemetry


class FakeDocker:
    """Stands in for the running-container view the watchdog surveys."""

    def __init__(self, containers: dict[str, list[str] | None]):
        self.containers = containers
        self.killed: list[str] = []
        self.restart_policies: dict[str, str] = {}

    async def ps(self, prefix=None, all_containers=True):
        return [{"Names": name} for name in self.containers]

    async def state(self, name):
        return docker_ctl.ContainerState(
            name=name,
            exists=True,
            status="running",
            running=True,
            command=self.containers.get(name),
        )

    async def kill(self, name):
        self.killed.append(name)

    async def set_restart_policy(self, name, policy):
        self.restart_policies[name] = policy

    async def inspect(self, name):
        return {"HostConfig": {"RestartPolicy": {"Name": "unless-stopped"}}}


@pytest.fixture
def fake(monkeypatch):
    def install(containers):
        stub = FakeDocker(containers)
        for method in ("ps", "state", "kill", "set_restart_policy", "inspect"):
            monkeypatch.setattr(memguard.docker_ctl, method, getattr(stub, method))
        return stub

    return install


@pytest.mark.asyncio
async def test_candidates_are_vllm_only_and_biggest_first(fake):
    fake(
        {
            "vllm-small": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.16"],
            "vllm-big": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"],
            "llmd-finetune-abc": ["python", "/worker/finetune_train.py", "/job/config.json"],
            "postgres": ["postgres"],
            "ray-head": ["ray", "start", "--head"],
        }
    )
    victims = await memguard._candidates()
    assert [name for name, _ in victims] == ["vllm-big", "vllm-small"]


@pytest.mark.asyncio
async def test_a_serve_container_with_no_util_flag_is_still_a_candidate(fake):
    fake({"vllm-plain": ["vllm", "serve", "m"]})
    victims = await memguard._candidates()
    assert [name for name, _ in victims] == ["vllm-plain"]


@pytest.mark.asyncio
async def test_nothing_is_killed_when_no_vllm_is_running(fake):
    stub = fake({"llmd-heretic-xyz": ["heretic", "--model", "m"], "nginx": ["nginx"]})
    assert await memguard._candidates() == []
    assert stub.killed == []


@pytest.mark.asyncio
async def test_a_starved_host_loses_its_largest_engine_and_keeps_it_off(fake, monkeypatch):
    stub = fake(
        {
            "vllm-big": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"],
            "vllm-small": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.16"],
        }
    )
    monkeypatch.setattr(
        memguard, "read_meminfo", lambda: telemetry.HostMemory(
            total_bytes=130_663_006_208, available_bytes=1 * 1024 ** 3
        )
    )
    monkeypatch.setattr(memguard, "POLL_SECONDS", 0.01)

    task = asyncio.create_task(memguard.watch())
    await asyncio.sleep(0.15)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert stub.killed[:1] == ["vllm-big"]
    # An `unless-stopped` engine would otherwise relaunch and immediately
    # re-reserve the memory that was just freed.
    assert stub.restart_policies["vllm-big"] == "no"
    assert memguard.history()[-1]["container"] == "vllm-big"
    # The operator's own policy is recorded, not silently discarded.
    assert memguard.history()[-1]["previous_restart_policy"] == "unless-stopped"


@pytest.mark.asyncio
async def test_a_healthy_host_is_left_alone(fake, monkeypatch):
    stub = fake({"vllm-big": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"]})
    monkeypatch.setattr(
        memguard, "read_meminfo", lambda: telemetry.HostMemory(
            total_bytes=130_663_006_208, available_bytes=60 * 1024 ** 3
        )
    )
    monkeypatch.setattr(memguard, "POLL_SECONDS", 0.01)

    task = asyncio.create_task(memguard.watch())
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert stub.killed == []


def test_the_watchdog_recognises_exactly_what_the_launch_guard_does():
    # Both sides must agree, or the guard could budget for an engine the
    # watchdog would refuse to kill.
    command = ["vllm", "serve", "m", "--gpu-memory-utilization", "0.5"]
    assert safety.is_vllm_command(command)
    assert safety.parse_util(command) == 0.5


@pytest.mark.asyncio
async def test_the_implicit_default_holder_is_killed_first(fake):
    # A container with no --gpu-memory-utilization is holding vLLM's own
    # default — about 112 GiB here — and ranking it by a fraction of 0.0 put
    # the biggest offender at the back of the queue.
    fake({
        "vllm-declared": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"],
        "vllm-implicit": ["vllm", "serve", "m"],
    })
    assert [name for name, _ in await memguard._candidates()] == ["vllm-implicit", "vllm-declared"]


@pytest.mark.asyncio
async def test_an_explicit_kv_cache_outranks_a_small_fraction(fake):
    fake({
        "vllm-small-frac-big-kv": [
            "vllm", "serve", "m", "--gpu-memory-utilization", "0.02",
            "--kv-cache-memory", "90G",
        ],
        "vllm-mid": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.30"],
    })
    ranked = [name for name, _ in await memguard._candidates()]
    assert ranked[0] == "vllm-small-frac-big-kv"
