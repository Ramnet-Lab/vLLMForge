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
    assert pool.reserve_bytes == 2 * GIB, "the OS does not live in a framebuffer"
    assert not pool.describes_host, "offloading to host RAM really does free VRAM here"
    assert pool.host.total_bytes == 128 * GIB, "host memory is still reported, separately"


def test_the_smallest_card_sets_the_ceiling():
    """The fraction applies per device, so a machine can only promise what its
    smallest one can hold."""
    rows = ("0, GPU-a, NVIDIA A, 24564, 100, 24000, [N/A]\n"
            "1, GPU-b, NVIDIA B, 49152, 200, 48000, [N/A]")
    host = HostMemory(total_bytes=256 * GIB, available_bytes=200 * GIB)
    pool = _pool(rows, host, "0, None\n1, None")
    assert pool.kind == accel.DISCRETE
    assert pool.device_count == 2
    assert pool.total_bytes == 24564 * MIB
    assert pool.available_bytes == 24000 * MIB


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


def test_a_reserve_is_never_larger_than_the_pool_it_guards():
    """A 8 GiB card must not be handed a 32 GiB reserve, which would leave a
    negative ceiling and refuse every launch with an unreadable number."""
    row = "0, GPU-x, Small Card, 8192, 0, 8000, [N/A]"
    host = HostMemory(total_bytes=64 * GIB, available_bytes=50 * GIB)
    pool = _pool(row, host, "0, None")
    assert pool.reserve_bytes < pool.total_bytes
    assert 0 < pool.reserve_bytes <= 2 * GIB
