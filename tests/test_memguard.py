"""The watchdog kills containers. What it will and will not touch is worth
pinning down, because the blast radius of a mistake here is the user's work."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from app import docker_ctl, engines, memguard, safety, telemetry


class FakeDocker:
    """Stands in for the running-container view the watchdog surveys."""

    def __init__(self, containers: dict[str, list[str] | None],
                 entrypoints: dict[str, list[str]] | None = None):
        self.containers = containers
        self.entrypoints = entrypoints or {}
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
            entrypoint=self.entrypoints.get(name),
        )

    async def kill(self, name, host=None):
        self.killed.append(name)

    async def set_restart_policy(self, name, policy, host=None):
        self.restart_policies[name] = policy

    async def inspect(self, name):
        return {"HostConfig": {"RestartPolicy": {"Name": "unless-stopped"}}}


async def until(predicate, timeout: float = 10.0) -> bool:
    """Wait for the watchdog to act, rather than for a fixed number of seconds.

    A tick now costs an nvidia-smi round trip, and how long that takes is a
    property of the machine: ~0.05s on the single unified part this was written
    on, ~0.45s on a loaded two-card box. A sleep(0.15) encoded the first as if
    it were a law, so the suite failed on the second while the watchdog was
    working correctly and merely had not been given time to finish a tick.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@pytest.fixture
def fake(monkeypatch):
    def install(containers, entrypoints=None):
        stub = FakeDocker(containers, entrypoints)
        for method in ("ps", "state", "kill", "set_restart_policy", "inspect"):
            monkeypatch.setattr(memguard.docker_ctl, method, getattr(stub, method))
        return stub

    return install


def ranked(victims):
    return [name for name, _util, _bytes in victims]


@pytest.mark.asyncio
async def test_candidates_are_engines_only_and_biggest_first(fake):
    fake(
        {
            "vllm-small": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.16"],
            "vllm-big": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"],
            "llmd-finetune-abc": ["python", "/worker/finetune_train.py", "/job/config.json"],
            "postgres": ["postgres"],
            "ray-head": ["ray", "start", "--head"],
        }
    )
    assert ranked(await memguard._candidates()) == ["vllm-big", "vllm-small"]


@pytest.mark.asyncio
async def test_a_serve_container_with_no_util_flag_is_still_a_candidate(fake):
    fake({"vllm-plain": ["vllm", "serve", "m"]})
    assert ranked(await memguard._candidates()) == ["vllm-plain"]


@pytest.mark.asyncio
async def test_nothing_is_killed_when_no_engine_is_running(fake):
    stub = fake({"llmd-heretic-xyz": ["heretic", "--model", "m"], "nginx": ["nginx"]})
    assert await memguard._candidates() == []
    assert stub.killed == []


@pytest.mark.asyncio
async def test_a_hand_launched_llama_server_is_a_candidate(fake):
    """The failure this fixes is inverted, not merely incomplete.

    Before the watchdog recognised a second engine, a llama-server holding tens
    of gigabytes was not a candidate — so when the threshold was crossed the
    watchdog killed the largest *vLLM* engine instead, which freed memory the
    machine was not short of and left the actual offender untouched.
    """
    fake({
        "llama-hand": ["llama-server", "-m", "/hf/big.gguf", "-ngl", "99"],
        "vllm-small": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.16"],
    })
    assert "llama-hand" in ranked(await memguard._candidates())


@pytest.mark.asyncio
async def test_an_engine_hiding_in_its_images_entrypoint_is_seen(fake):
    """The upstream llama.cpp server images put the binary in ENTRYPOINT and
    leave bare flags in Cmd, so the command alone reads as no engine at all."""
    stub = fake(
        {"llama-upstream": ["-m", "/models/x.gguf", "-ngl", "99"]},
        {"llama-upstream": ["/app/llama-server"]},
    )
    assert ranked(await memguard._candidates()) == ["llama-upstream"]
    # And the fallback only ever ADDS: what docker calls the command is
    # untouched, because every other parser in the codebase reads it directly.
    assert (await stub.state("llama-upstream")).command == ["-m", "/models/x.gguf",
                                                            "-ngl", "99"]


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
    await until(lambda: bool(stub.killed))
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
    # Long enough for several ticks on a slow nvidia-smi: an assertion that
    # nothing was killed is worthless if nothing had time to look.
    await asyncio.sleep(1.5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert stub.killed == []


@pytest.mark.parametrize(
    "command,engine,util",
    [
        (["vllm", "serve", "m", "--gpu-memory-utilization", "0.5"], "vllm", 0.5),
        (["llama-server", "-m", "/hf/x.gguf", "-ngl", "99"], "llamacpp", None),
    ],
)
def test_the_watchdog_recognises_exactly_what_the_launch_guard_does(command, engine, util):
    # Both sides must agree, or the guard budgets for an engine the watchdog
    # would refuse to kill — which is not a gap but an inversion: the threshold
    # is crossed, the offender survives, and something healthy dies instead.
    # They agree by construction now, because both ask engines.recognise().
    found = engines.recognise_argv(command)
    assert found is not None and found.name == engine
    assert found.declared_util(found.command_params(command)) == util


def test_the_vllm_gate_was_not_widened():
    """The predicate the budget and the watchdog shared for one engine is still
    about that one engine. Widening it would have admitted a llama.cpp container
    to every caller at once while `footprint` still priced it vLLM's way."""
    assert safety.is_vllm_command(["vllm", "serve", "m"])
    assert not safety.is_vllm_command(["llama-server", "-m", "x.gguf"])
    assert safety.parse_util(["llama-server", "-m", "x.gguf", "-ngl", "99"]) is None


def test_a_llamacpp_config_is_never_priced_at_vllms_default():
    """The one mistake that would take the whole machine's launching down.

    vLLM's pricer charges its default fraction to any argv with no utilisation
    flag — which is every llama.cpp argv there is. On this box that is over
    100 GiB, so a single mis-routed llama.cpp server would refuse every launch
    after it.
    """
    total = 130_663_006_208
    llama = engines.get("llamacpp")
    priced = llama.footprint_bytes({"n_gpu_layers": "99"}, total)
    assert priced != int(safety.default_util() * total)
    assert priced < total


@pytest.mark.asyncio
async def test_the_implicit_default_holder_is_killed_first(fake):
    # A container with no --gpu-memory-utilization is holding vLLM's own
    # default — about 112 GiB here — and ranking it by a fraction of 0.0 put
    # the biggest offender at the back of the queue.
    fake({
        "vllm-declared": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"],
        "vllm-implicit": ["vllm", "serve", "m"],
    })
    assert ranked(await memguard._candidates()) == ["vllm-implicit", "vllm-declared"]


@pytest.mark.asyncio
async def test_an_explicit_kv_cache_outranks_a_small_fraction(fake):
    fake({
        "vllm-small-frac-big-kv": [
            "vllm", "serve", "m", "--gpu-memory-utilization", "0.02",
            "--kv-cache-memory", "90G",
        ],
        "vllm-mid": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.30"],
    })
    assert ranked(await memguard._candidates())[0] == "vllm-small-frac-big-kv"


def test_a_victim_with_no_fraction_is_named_by_what_it_holds():
    """The kill notice reaches the operator through the history and the live
    telemetry feed. Printing "util 0.92" for an engine that has no such flag
    would put a fabricated number in both."""
    starved = memguard.Starvation(kind="device", available_bytes=1024 ** 3,
                                  threshold_bytes=2 * 1024 ** 3, pool_kind="discrete")
    assert "util 0.52" in starved.reason("llmd-vllm-1", 0.52)
    assert "holding 40960 MiB" in starved.reason("llmd-llamacpp-2", None, 40 * 1024 ** 3)


def discrete_pool(free_bytes: int, total_bytes: int = 48 * 1024 ** 3):
    """A machine whose GPU has its own memory, at a stated level of free."""
    from app import accel

    return accel.Pool(
        kind=accel.DISCRETE, confidence="measured", can_size=True,
        total_bytes=total_bytes, available_bytes=free_bytes, free_bytes=free_bytes,
        has_accelerator=True,
    )


@pytest.mark.asyncio
async def test_a_full_framebuffer_kills_even_while_host_ram_is_plentiful(fake, monkeypatch):
    """The failure this whole split exists for. Host MemAvailable reads 90 GiB
    on a discrete box whose every card is full, so the old trigger never fired
    and the watchdog watched the machine wedge."""
    stub = fake({
        "vllm-big": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"],
        "vllm-small": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.16"],
    })

    async def pool(node=None):
        return discrete_pool(free_bytes=256 * 1024 ** 2)

    monkeypatch.setattr(memguard.accel, "pool_for", pool)
    monkeypatch.setattr(memguard, "read_meminfo", lambda: telemetry.HostMemory(
        total_bytes=100 * 1024 ** 3, available_bytes=90 * 1024 ** 3))
    monkeypatch.setattr(memguard, "POLL_SECONDS", 0.01)

    task = asyncio.create_task(memguard.watch())
    await until(lambda: bool(stub.killed))
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert stub.killed[:1] == ["vllm-big"]
    assert "device free" in memguard.history()[-1]["reason"]


@pytest.mark.asyncio
async def test_a_roomy_framebuffer_is_left_alone_however_starved_the_host_is(fake, monkeypatch):
    """The other half: host memory is not the engines' pool on a discrete box,
    so its exhaustion is not theirs to answer for. Killing a serving engine
    because something else on the machine ate the RAM is an outage we caused."""
    stub = fake({"vllm-big": ["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"]})

    async def pool(node=None):
        return discrete_pool(free_bytes=30 * 1024 ** 3)

    monkeypatch.setattr(memguard.accel, "pool_for", pool)
    monkeypatch.setattr(memguard, "read_meminfo", lambda: telemetry.HostMemory(
        total_bytes=100 * 1024 ** 3, available_bytes=1 * 1024 ** 3))
    monkeypatch.setattr(memguard, "POLL_SECONDS", 0.01)

    task = asyncio.create_task(memguard.watch())
    await asyncio.sleep(1.5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert stub.killed == []
