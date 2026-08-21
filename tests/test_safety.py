"""The memory guard is the one component whose failure mode is a frozen host."""

from __future__ import annotations

import pytest

from app import safety

GIB = 1024 ** 3
TOTAL = 130_663_006_208  # this box: torch's total_memory and MemTotal agree exactly


def budget(
    committed: list[tuple[str, float]],
    available_gib: float = 60.0,
    free_gib: float | None = None,
    measured_gib: float = 0.0,
) -> safety.Budget:
    return safety.Budget(
        total_bytes=TOTAL,
        available_bytes=int(available_gib * GIB),
        free_bytes=int((available_gib if free_gib is None else free_gib) * GIB),
        measured_gpu_bytes=int(measured_gib * GIB),
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
    assert b.max_util < 0.80, "0.80 alone must not fit; it hard-locked this host once"


def test_the_validated_configuration_fits():
    b = budget([])
    assert b.max_util >= 0.52 + 0.16
    assert b.max_util >= 0.57


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
async def test_an_unset_util_is_evaluated_as_vllm_s_default(monkeypatch):
    # Leaving the flag off is not "unspecified", it is 0.92 — 112 GiB here, which
    # cannot fit beside the OS reserve on an otherwise idle host.
    async def fake(exclude=None):
        return budget([])

    monkeypatch.setattr(safety, "current_budget", fake)
    verdict = await safety.check_launch(None)
    assert not verdict.ok and verdict.level == "block"
    assert "0.92" in verdict.message


@pytest.mark.asyncio
async def test_live_headroom_blocks_even_a_budget_legal_launch(monkeypatch):
    async def fake(exclude=None):
        return budget([], available_gib=4.0)

    monkeypatch.setattr(safety, "current_budget", fake)
    verdict = await safety.check_launch(0.5)
    assert not verdict.ok and "available on the host right now" in verdict.message


def test_measured_usage_wins_when_it_exceeds_the_declared_utils():
    # A fine-tuning job or a Heretic run holds GPU memory without any util flag
    # to sum, so the measured figure has to be able to dominate.
    b = budget([("vllm-embed", 0.16)], measured_gib=90.0)
    assert b.committed_bytes < b.measured_gpu_bytes
    assert b.occupied_bytes == b.measured_gpu_bytes
    assert b.free_util < 0.05


def test_the_declared_utils_win_when_engines_have_not_finished_allocating():
    b = budget([("vllm-qwen", 0.52)], measured_gib=5.0)
    assert b.occupied_bytes == b.committed_bytes


@pytest.mark.asyncio
async def test_a_serve_command_without_a_util_flag_is_not_free(monkeypatch):
    # vLLM applies its own 0.92 default, which is over 100 GiB on this host.
    # Treating such a container as contributing zero was the dangerous bug here.
    async def fake(exclude=None):
        return budget([("vllm-nodefault", safety.default_util())])

    monkeypatch.setattr(safety, "current_budget", fake)
    verdict = await safety.check_launch(0.2)
    assert not verdict.ok and verdict.level == "block"


@pytest.mark.asyncio
async def test_free_versus_available_is_surfaced_not_ignored(monkeypatch):
    # vLLM's own guard compares against free memory; page cache does not count
    # for it even though MemAvailable includes it.
    async def fake(exclude=None):
        return budget([], available_gib=80.0, free_gib=5.0)

    monkeypatch.setattr(safety, "current_budget", fake)
    verdict = await safety.check_launch(0.3)
    assert verdict.ok and verdict.level == "warn"
    assert "free memory" in verdict.message


@pytest.mark.asyncio
async def test_check_job_blocks_when_a_model_will_not_fit(monkeypatch):
    async def fake(exclude=None):
        return budget([("vllm-qwen", 0.52), ("vllm-embed", 0.16)])

    monkeypatch.setattr(safety, "current_budget", fake)
    blocked = await safety.check_job(int(30 * GIB), label="A 12B Heretic run")
    assert not blocked.ok and "12B Heretic run" in blocked.message

    fits = await safety.check_job(int(2 * GIB), label="A 0.5B Heretic run")
    assert fits.ok


def test_kv_cache_memory_override_is_recognised():
    assert safety.KV_BYTES_FLAG.match("--kv-cache-memory")
    assert safety.KV_BYTES_FLAG.match("--kv-cache-memory-bytes=123")
    assert not safety.KV_BYTES_FLAG.match("--kv-cache-dtype")
