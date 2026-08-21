"""The memory guard is the one component whose failure mode is a frozen host."""

from __future__ import annotations

import pytest

from app import safety

GIB = 1024 ** 3
TOTAL = 130_663_006_208  # this box: torch's total_memory and MemTotal agree exactly


def budget(committed: list[tuple[str, float]], available_gib: float = 60.0) -> safety.Budget:
    return safety.Budget(
        total_bytes=TOTAL,
        available_bytes=int(available_gib * GIB),
        reserve_bytes=int(32 * GIB),
        warn_reserve_bytes=int(38 * GIB),
        tenants=[
            safety.Tenant(name=name, util=util, managed=False, bytes_committed=int(util * TOTAL))
            for name, util in committed
        ],
    )


@pytest.mark.parametrize(
    "command,expected",
    [
        (["vllm", "serve", "m", "--gpu-memory-utilization", "0.52"], 0.52),
        (["vllm", "serve", "m", "--gpu-memory-utilization=0.16"], 0.16),
        (["vllm", "serve", "m", "--gpu_memory_utilization", "0.4"], 0.4),
        (["vllm", "serve", "m"], None),
        (["vllm", "serve", "m", "--gpu-memory-utilization"], None),
        (None, None),
    ],
)
def test_parse_util(command, expected):
    assert safety.parse_util(command) == expected


def test_is_vllm_command():
    assert safety.is_vllm_command(["vllm", "serve", "m"])
    assert not safety.is_vllm_command(["ray", "start", "--head"])
    assert not safety.is_vllm_command(["python", "-c", "serve"])


def test_the_utils_of_co_resident_engines_add_up():
    b = budget([("vllm-qwen", 0.52), ("vllm-embed", 0.16)])
    assert b.committed_util == pytest.approx(0.68)
    # 121.7 GiB total minus a 32 GiB reserve leaves room for a 0.737 total.
    assert b.max_util == pytest.approx(0.737, abs=0.002)
    assert b.free_util == pytest.approx(0.057, abs=0.002)


def test_the_value_that_locked_this_machine_is_refused():
    b = budget([])
    assert 0.80 > b.max_util, "0.80 alone must not fit; it hard-locked this host once"


def test_the_validated_configuration_fits():
    b = budget([])
    assert 0.52 + 0.16 <= b.max_util
    assert 0.57 <= b.max_util


@pytest.mark.asyncio
async def test_check_launch_blocks_an_overcommit(monkeypatch):
    async def fake(exclude=None):
        return budget([("vllm-qwen", 0.52), ("vllm-embed", 0.16)])

    monkeypatch.setattr(safety, "current_budget", fake)
    verdict = await safety.check_launch(0.30)
    assert not verdict.ok and verdict.level == "block"
    assert "0.52" in verdict.message and "vllm-qwen" in verdict.message
    assert verdict.suggested_util is not None and verdict.suggested_util < 0.30


@pytest.mark.asyncio
async def test_check_launch_warns_about_an_unset_util(monkeypatch):
    async def fake(exclude=None):
        return budget([])

    monkeypatch.setattr(safety, "current_budget", fake)
    verdict = await safety.check_launch(None)
    assert verdict.level == "warn" and "0.92" in verdict.message


@pytest.mark.asyncio
async def test_live_headroom_blocks_even_a_budget_legal_launch(monkeypatch):
    async def fake(exclude=None):
        return budget([], available_gib=4.0)

    monkeypatch.setattr(safety, "current_budget", fake)
    verdict = await safety.check_launch(0.5)
    assert not verdict.ok and "available on the host right now" in verdict.message
