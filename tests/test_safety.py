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


def test_kv_cache_memory_override_is_recognised():
    assert safety.KV_BYTES_FLAG.match("--kv-cache-memory")
    assert safety.KV_BYTES_FLAG.match("--kv-cache-memory-bytes=123")
    assert not safety.KV_BYTES_FLAG.match("--kv-cache-dtype")


@pytest.mark.parametrize(
    "raw,expected",
    [("25.6k", 25600), ("100G", 100_000_000_000), ("8Gi", 8589934592),
     (12345, 12345), ("auto", None), (None, None), ("nonsense", None)],
)
def test_human_readable_sizes(raw, expected):
    from app import vllm_spec

    assert vllm_spec.parse_size(raw) == expected


def test_flags_that_bypass_the_fraction_are_counted():
    from app import vllm_spec

    plain = vllm_spec.footprint_bytes({"gpu_memory_utilization": 0.02}, TOTAL, default_util=0.92)
    assert plain == pytest.approx(0.02 * TOTAL, rel=0.01)

    # --kv-cache-memory sizes the cache explicitly and overrides the fraction,
    # so pairing it with a tiny util used to look almost free.
    with_kv = vllm_spec.footprint_bytes(
        {"gpu_memory_utilization": 0.02, "kv_cache_memory_bytes": "90G"}, TOTAL, default_util=0.92
    )
    assert with_kv >= 90_000_000_000

    # --cpu-offload-gb lands in the same unified pool on this part.
    with_offload = vllm_spec.footprint_bytes(
        {"gpu_memory_utilization": 0.1, "cpu_offload_gb": 40}, TOTAL, default_util=0.92
    )
    assert with_offload == pytest.approx(0.1 * TOTAL + 40 * GIB, rel=0.01)


@pytest.mark.parametrize(
    "command,key,value",
    [
        (["vllm", "serve", "m", "--kv-cache-memory", "90G"], "kv_cache_memory_bytes", "90G"),
        (["vllm", "serve", "m", "--kv-cache-memory-bytes=8Gi"], "kv_cache_memory_bytes", "8Gi"),
        (["vllm", "serve", "m", "--cpu-offload-gb", "40"], "cpu_offload_gb", "40"),
        (["vllm", "serve", "m", "--gpu-memory-utilization=0.4"], "gpu_memory_utilization", 0.4),
    ],
)
def test_a_running_container_is_read_back_the_way_a_stored_config_is(command, key, value):
    assert safety.command_params(command)[key] == value


@pytest.mark.asyncio
async def test_a_huge_explicit_kv_cache_cannot_walk_past_the_guard(monkeypatch):
    async def fake(exclude=None):
        return budget([])

    monkeypatch.setattr(safety, "current_budget", fake)

    sneaky = {"gpu_memory_utilization": 0.02, "kv_cache_memory_bytes": "90G"}
    verdict = await safety.check_launch(0.02, params=sneaky)
    assert not verdict.ok, "0.02 with a 90G explicit cache is not a 2.4 GiB launch"
    assert "--kv-cache-memory" in verdict.message

    honest = await safety.check_launch(0.02, params={"gpu_memory_utilization": 0.02})
    assert honest.ok


def test_tenants_are_summed_in_bytes_not_fractions():
    b = budget([])
    b.tenants = [
        safety.Tenant(name="a", util=0.02, managed=False, bytes_committed=90 * GIB),
        safety.Tenant(name="b", util=0.10, managed=False, bytes_committed=12 * GIB),
    ]
    # Summing the declared fractions would report 0.12 of the machine committed
    # while 102 GiB of it was actually spoken for.
    assert b.committed_bytes == 102 * GIB
    assert b.committed_util > 0.8
