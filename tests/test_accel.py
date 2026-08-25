"""Which memory pool a launch competes for, on whatever machine this is.

The rule these all serve: only positive, corroborated proof of a separate
framebuffer yields DISCRETE. Guessing discrete on a unified box lets a launch
claim memory the OS is living in and freezes the machine; guessing unified on a
discrete box only refuses launches that would have fit. The asymmetry is the
whole design.
"""

from __future__ import annotations

import pytest

from app import accel
from app.telemetry import HostMemory

GIB = 1024 ** 3
MIB = 1024 ** 2

# The shape a GB10 actually returns: every memory field declined.
GB10_ROW = "0, GPU-00000000-0000-0000-0000-000000000000, NVIDIA GB10, [N/A], [N/A], [N/A], [N/A]"
GB10_HOST = HostMemory(total_bytes=130_663_006_208,      # MemTotal 127600592 kB
                       available_bytes=23_591_030_784,
                       free_bytes=2_000_000_000)


def _pool(rows: str, host: HostMemory, addr: str = "", *, smi_ok: bool = True,
          override: str = "") -> accel.Pool:
    devices = accel._apply_addressing(accel.parse_accel_csv(rows), accel.parse_addressing(addr))
    return accel.build(devices, host, smi_ok=smi_ok, override=override,
                       saw_output=bool(rows.strip()))


def test_this_boxs_pool_is_host_memory_byte_for_byte():
    """The validated behaviour. Every number the guard uses on a GB10 must come
    out of the new probe identical to what /proc/meminfo said before it existed,
    because two engines were admitted against those numbers."""
    pool = _pool(GB10_ROW, GB10_HOST, "0, ATS")
    assert pool.kind == accel.UNIFIED
    assert pool.total_bytes == GB10_HOST.total_bytes == 130_663_006_208
    assert pool.available_bytes == GB10_HOST.available_bytes
    assert pool.free_bytes == GB10_HOST.free_bytes
    assert pool.reserve_bytes == 32 * GIB
    assert pool.warn_reserve_bytes == 38 * GIB
    assert pool.can_size and pool.has_accelerator
    assert pool.describes_host, "--cpu-offload-gb frees nothing on this machine"


def test_a_discrete_card_is_sized_against_its_framebuffer():
    """24 GiB of VRAM in a box with 128 GiB of RAM: the fraction multiplies the
    card, not the host, and the reserve stops being room for a desktop."""
    row = "0, GPU-aaaa, NVIDIA GeForce RTX 4090, 24564, 512, 24052, [N/A]"
    host = HostMemory(total_bytes=128 * GIB, available_bytes=100 * GIB, free_bytes=90 * GIB)
    pool = _pool(row, host, "0, None")

    assert pool.kind == accel.DISCRETE
    assert pool.total_bytes == 24564 * MIB
    assert pool.available_bytes == 24052 * MIB
    assert pool.measured_bytes == 512 * MIB
    assert pool.reserve_bytes == 0, "the OS does not live in a framebuffer, so nothing is held back"
    assert pool.warn_reserve_bytes == 0
    assert pool.host_reserve_bytes > 0, "host RAM is still guarded — CPU layers land there"
    assert not pool.describes_host, "offloading to host RAM really does free VRAM here"
    assert pool.host.total_bytes == 128 * GIB, "host memory is still reported, separately"


def test_the_pool_is_every_card_and_the_ceiling_is_the_smallest():
    """Two numbers, because two different questions are asked of them.

    What the box has to spend is every framebuffer added up: an engine given
    `--gpus all` splits across all of them. This used to report the smallest
    card as the whole pool while still counting what every card held, which is
    not conservative but incoherent — it refused a 14 GiB launch into 19 GiB of
    free video memory and printed a negative headroom doing it.

    The smallest card survives as `device_total_bytes`, because a per-device
    fraction like --gpu-memory-utilization multiplies one card and a machine can
    only promise what its smallest can hold.
    """
    rows = ("0, GPU-a, NVIDIA A, 24564, 100, 24000, [N/A]\n"
            "1, GPU-b, NVIDIA B, 49152, 200, 48000, [N/A]")
    host = HostMemory(total_bytes=256 * GIB, available_bytes=200 * GIB)
    pool = _pool(rows, host, "0, None\n1, None")
    assert pool.kind == accel.DISCRETE
    assert pool.device_count == 2
    assert pool.total_bytes == (24564 + 49152) * MIB
    assert pool.available_bytes == (24000 + 48000) * MIB
    assert pool.measured_bytes == (100 + 200) * MIB, "occupancy adds up too"
    assert pool.device_total_bytes == 24564 * MIB


def test_a_unified_pool_is_one_card_by_definition():
    """device_total_bytes is not a discrete-only concept with a hole in it: on a
    unified part there is one pool and the fraction multiplies all of it."""
    pool = _pool(GB10_ROW, GB10_HOST, "0, ATS")
    assert pool.kind == accel.UNIFIED
    assert pool.device_total_bytes == pool.total_bytes == GB10_HOST.total_bytes


def test_a_part_that_addresses_host_memory_is_never_called_discrete():
    """Guards this box against a future driver that starts answering
    memory.total: ATS means the device addresses host memory, so a number there
    is the same pool counted twice."""
    row = "0, GPU-x, NVIDIA GB10, 65536, 1024, 64000, [N/A]"
    pool = _pool(row, GB10_HOST, "0, ATS")
    assert pool.kind == accel.UNIFIED
    assert pool.reserve_bytes == 32 * GIB


def test_a_total_that_is_nearly_all_of_host_memory_is_the_same_memory():
    """A "framebuffer" the size of the box is host memory being reported twice,
    whatever the driver calls it."""
    row = "0, GPU-x, Some Part, 124000, 0, 120000, [N/A]"
    host = HostMemory(total_bytes=128 * GIB, available_bytes=100 * GIB)
    assert _pool(row, host, "0, None").kind == accel.UNIFIED


@pytest.mark.parametrize("rows,smi_ok,addr", [
    ("", False, ""),                                        # no nvidia-smi at all
    ("", False, "0, None"),                                 # ran, said nothing, failed
    (GB10_ROW, True, ""),                                   # every field [N/A]
    ("0, GPU-x, Card, [Insufficient Permissions], [N/A], [N/A], [N/A]", True, "0, None"),
    ("garbage not a csv row", True, ""),                    # unparseable
    ("0, GPU-x, Card, 24564", True, ""),                    # truncated row
])
def test_everything_that_is_not_proof_falls_back_to_unified(rows, smi_ok, addr):
    """Absence of evidence never relaxes a reserve."""
    pool = _pool(rows, GB10_HOST, addr, smi_ok=smi_ok)
    assert pool.kind == accel.UNIFIED
    assert pool.total_bytes == GB10_HOST.total_bytes
    assert pool.reserve_bytes == 32 * GIB


def test_mig_refuses_to_size_rather_than_guessing():
    """A partitioned device's usable memory is not what memory.total reports,
    so there is no honest fraction to recommend."""
    row = "0, GPU-x, NVIDIA A100, 81920, 0, 81920, Enabled"
    host = HostMemory(total_bytes=256 * GIB, available_bytes=200 * GIB)
    pool = _pool(row, host, "0, None")
    assert pool.kind == accel.UNKNOWN
    assert pool.can_size is False

    # And GB10's [N/A] in that column must not read as MIG.
    assert _pool(GB10_ROW, GB10_HOST, "0, ATS").can_size is True


def test_a_machine_with_no_gpu_is_blocked_rather_than_sized():
    pool = _pool("", GB10_HOST, smi_ok=True)
    assert pool.kind == accel.NONE
    assert pool.has_accelerator is False
    assert pool.can_size is False


def test_some_devices_answering_and_others_not_is_not_an_average():
    rows = ("0, GPU-a, Card A, 24564, 100, 24000, [N/A]\n"
            "1, GPU-b, Card B, [N/A], [N/A], [N/A], [N/A]")
    host = HostMemory(total_bytes=128 * GIB, available_bytes=100 * GIB)
    pool = _pool(rows, host, "0, None\n1, None")
    assert pool.kind == accel.UNKNOWN and pool.can_size is False


def test_an_override_wins_and_says_so():
    pool = _pool(GB10_ROW, GB10_HOST, "0, ATS", override=accel.DISCRETE)
    assert pool.kind == accel.DISCRETE and pool.confidence == "configured"


def test_a_small_card_keeps_all_of_itself():
    """An 8 GiB card must not be handed a reserve at all, let alone a 32 GiB one.

    This used to assert a small-but-nonzero figure and a cap that kept it below
    the pool. Both are gone for the same reason: the reserve exists to stop an
    overcommit freezing a machine whose OS is in the pool being claimed, and on
    a discrete card it is not. Holding any of a small card back only makes it
    smaller.
    """
    row = "0, GPU-x, Small Card, 8192, 0, 8000, [N/A]"
    host = HostMemory(total_bytes=64 * GIB, available_bytes=50 * GIB)
    pool = _pool(row, host, "0, None")
    assert pool.kind == accel.DISCRETE
    assert pool.reserve_bytes == 0
    assert pool.warn_reserve_bytes == 0


def test_a_unified_pool_still_caps_its_reserve_below_itself():
    """The cap that discrete no longer needs still matters where the reserve is
    real: a 24 GiB unified part must not be handed the configured 32 GiB and
    left with a negative ceiling that refuses every launch."""
    small = HostMemory(total_bytes=24 * GIB, available_bytes=20 * GIB, free_bytes=18 * GIB)
    hard, warn, _host = accel.reserves_for(accel.UNIFIED, small.total_bytes, small.total_bytes)
    assert 0 < hard < small.total_bytes
    assert hard <= int(0.30 * small.total_bytes)
    assert warn <= int(0.35 * small.total_bytes)
