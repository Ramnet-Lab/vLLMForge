"""Which pool `--gpu-memory-utilization` is a fraction of, on this machine.

Every memory decision in this dashboard used to read /proc/meminfo and treat
host RAM as the accelerator's memory. On the box it was built on — a GB10 with
unified memory — that is not an approximation, it is the same bytes. On a
machine with a discrete GPU it is simply false, and everything downstream (the
budget, the recommended utilisation, the launch guard) is then wrong.

So the pool is *detected*, and the two numbers that matter keep their meaning
whichever kind of machine this is:

    total_bytes      what vLLM multiplies --gpu-memory-utilization by
    available_bytes  what vLLM's own startup gate compares the result against

On unified those are MemTotal and MemAvailable. On discrete they are one
device's framebuffer total and its free memory. Nothing else in the codebase has
to know which it got.

**Fail unified.** Only positive, corroborated proof of a separate framebuffer
yields DISCRETE; everything else — no nvidia-smi, a timeout, a driver that
answers [N/A], a half-parsed row — falls back to unified with its full reserve.
The asymmetry is deliberate and is not a matter of taste: guessing "discrete" on
a unified box lets a launch claim memory the OS is living in, and that freezes
the machine with no OOM killer fast enough to save it. Guessing "unified" on a
discrete box merely refuses launches that would have fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.telemetry import HostMemory, read_meminfo

GIB = 1024 ** 3
MIB = 1024 ** 2

UNIFIED = "unified"
DISCRETE = "discrete"
NONE = "none"
UNKNOWN = "unknown"

# Asked separately from the telemetry tile's query on purpose: nvidia-smi fails
# the WHOLE query with a non-zero exit when it does not recognise one field, so
# a newer field name here would blind the memory probe on an older driver rather
# than just omitting a column.
ACCEL_FIELDS = ["index", "uuid", "name", "memory.total", "memory.used", "memory.free",
                "mig.mode.current"]
ACCEL_QUERY = ["--query-gpu=" + ",".join(ACCEL_FIELDS), "--format=csv,noheader,nounits"]
# Separate again, and for the same reason: addressing_mode is recent.
ADDR_QUERY = ["--query-gpu=index,addressing_mode", "--format=csv,noheader,nounits"]

# A device total this close to host memory is not a framebuffer, it is host
# memory being reported by a driver that shares it.
UNIFIED_RATIO = 0.90


@dataclass(frozen=True)
class Device:
    index: int
    uuid: str = ""
    name: str = ""
    total_bytes: int | None = None      # None means the driver would not say
    used_bytes: int | None = None
    free_bytes: int | None = None
    mig: bool = False
    addressing_mode: str = ""           # ATS | HMM | None | ""


@dataclass(frozen=True)
class Pool:
    """The memory a launch on one machine competes for."""

    kind: str = UNIFIED
    confidence: str = "assumed"         # measured | inferred | assumed | configured

    # The engine pool: what the utilisation fraction multiplies.
    total_bytes: int = 0
    available_bytes: int = 0
    free_bytes: int = 0
    measured_bytes: int = 0
    reserve_bytes: int = 0
    warn_reserve_bytes: int = 0

    device_total_bytes: int = 0
    """One card's framebuffer — the smallest, when they differ. `total_bytes` is
    every card added up, which is what the box has to spend; this is what a
    per-device fraction like --gpu-memory-utilization multiplies. On a unified
    pool the two are the same number, because there is only one pool."""

    # The host, which is a separate thing on a discrete box and the same thing
    # on a unified one. Always truthful either way.
    host: HostMemory = field(default_factory=HostMemory)
    host_reserve_bytes: int = 0

    devices: tuple[Device, ...] = ()
    has_accelerator: bool = True
    can_size: bool = False
    """Whether a utilisation may be recommended at all. False when the machine
    could be read but not understood — MIG, or some devices answering and others
    not — because a number produced from half an answer is worse than none."""
    evidence: tuple[str, ...] = ()

    @property
    def device_count(self) -> int:
        return len(self.devices)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "reserve_bytes": self.reserve_bytes,
            "device_count": self.device_count,
            "device_total_bytes": self.device_total_bytes,
            "devices": [
                {"index": d.index, "name": d.name, "total_bytes": d.total_bytes}
                for d in self.devices
            ],
            "has_accelerator": self.has_accelerator,
            "can_size": self.can_size,
            "evidence": list(self.evidence),
            "host_total_bytes": self.host.total_bytes,
            "host_available_bytes": self.host.available_bytes,
        }

    @property
    def describes_host(self) -> bool:
        """Whether this pool and host RAM are the same bytes.

        The thing that makes --cpu-offload-gb free nothing, and the thing that
        makes the OS a tenant of the model's memory.
        """
        return self.kind in (UNIFIED, NONE, UNKNOWN)


def _int(raw: str) -> int | None:
    """A field nvidia-smi answered with a number, or None for anything else.

    Anything bracketed — [N/A], [Insufficient Permissions], [Unknown Error] — is
    the driver declining to answer, which must not survive as a truthy string.
    """
    raw = (raw or "").strip()
    if not raw or raw.startswith("["):
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def parse_accel_csv(out: str) -> list[Device]:
    """Devices from `--query-gpu=index,uuid,name,memory.*,mig.mode.current`."""
    devices: list[Device] = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(ACCEL_FIELDS):
            continue
        index = _int(parts[0])
        if index is None:
            continue
        total, used, free = (_int(parts[3]), _int(parts[4]), _int(parts[5]))
        devices.append(Device(
            index=index,
            uuid=parts[1],
            name=parts[2],
            total_bytes=None if total is None else total * MIB,
            used_bytes=None if used is None else used * MIB,
            free_bytes=None if free is None else free * MIB,
            # Only the literal word counts. GB10 answers [N/A] here, and reading
            # that as "MIG" would make this box unsizeable.
            mig=parts[6].strip().lower() == "enabled",
        ))
    return devices


def parse_addressing(out: str) -> dict[int, str]:
    modes: dict[int, str] = {}
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        index = _int(parts[0])
        if index is not None:
            modes[index] = parts[1]
    return modes


def classify(devices: list[Device], host: HostMemory, smi_ok: bool,
             override: str = "", saw_output: bool = False
             ) -> tuple[str, str, bool, tuple[str, ...]]:
    """(kind, confidence, can_size, evidence). Pure, total, never raises."""
    if override in (UNIFIED, DISCRETE, NONE):
        return override, "configured", override != NONE, (f"LLMD_ACCEL_MODE={override}",)

    if not smi_ok:
        return UNIFIED, "assumed", True, (
            "nvidia-smi did not answer, so the pool is assumed to be host memory",)

    if not devices:
        if saw_output:
            # It said something and none of it parsed. That is not proof of an
            # absent GPU, it is proof that this probe does not understand this
            # driver — which is exactly when to fall back rather than conclude.
            return UNIFIED, "assumed", True, (
                "nvidia-smi answered in a form this probe could not read, so the pool is "
                "assumed to be host memory",)
        return NONE, "measured", False, ("nvidia-smi reports no devices",)

    if any(d.mig for d in devices):
        return UNKNOWN, "measured", False, (
            "MIG is enabled, and a partitioned device's usable memory is not what "
            "memory.total reports",)

    totals = [d.total_bytes for d in devices]
    if any(t is None for t in totals):
        if all(t is None for t in totals):
            return UNIFIED, "inferred", True, (
                "the driver reports no per-device memory, which is what a unified part does",)
        return UNKNOWN, "measured", False, (
            "some devices report their memory and others do not, so there is no "
            "figure that describes the machine",)

    smallest = min(t for t in totals if t is not None)
    if any(d.free_bytes is None for d in devices):
        return UNKNOWN, "measured", False, (
            "devices report a total but not their free memory, so a launch cannot be judged",)

    ats = [d.index for d in devices if d.addressing_mode.upper() == "ATS"]
    if ats:
        return UNIFIED, "measured", True, (
            f"device {ats[0]} addresses memory through ATS, so its memory is the host's",)

    if host.total_bytes and smallest > UNIFIED_RATIO * host.total_bytes:
        return UNIFIED, "inferred", True, (
            f"the device reports {smallest // GIB} GiB against {host.total_bytes // GIB} GiB "
            "of host memory, which is the same pool counted twice",)

    return DISCRETE, "measured", True, (
        f"{len(devices)} device(s), smallest framebuffer {smallest // GIB} GiB, "
        f"separate from {host.total_bytes // GIB} GiB of host memory",)


def reserves_for(kind: str, total_bytes: int, host_total: int) -> tuple[int, int, int]:
    """(engine reserve, engine warn reserve, host reserve) for this machine.

    On unified the reserve is the operator's configured figure, because the OS
    is a tenant of the very pool the model is claiming — 32 GiB here, measured.

    **On discrete the reserve is nothing, and that is deliberate.** The reserve
    exists for exactly one failure: a pool the OS is living in, overcommitted,
    freezing a machine that no OOM killer reacts to in time. A framebuffer is
    not that pool. Nothing the operator sees — the desktop, the ssh session,
    every other engine on the box — is inside it, and the way a discrete GPU
    runs out is a CUDA OOM inside the process that asked for too much. That
    process dies; the machine does not notice. So holding gigabytes of VRAM back
    against a freeze that cannot happen just makes a card smaller than it is,
    and on this host it refused a 14 GiB launch into 19 GiB of free framebuffer.

    The HOST reserve is untouched and still real: llama.cpp puts every layer it
    could not offload into host RAM, and that pool is the one the OS lives in.

    The unified figures are capped by a fraction of the pool so a small part is
    never handed a reserve larger than itself.
    """
    if kind == DISCRETE:
        hard = warn = 0
    else:
        hard = min(int(settings.mem_reserve_gib * GIB), int(0.30 * total_bytes)) \
            if total_bytes else int(settings.mem_reserve_gib * GIB)
        warn = min(int(settings.mem_warn_reserve_gib * GIB), int(0.35 * total_bytes)) \
            if total_bytes else int(settings.mem_warn_reserve_gib * GIB)
    host_reserve = int(settings.host_reserve_gib * GIB) if kind == DISCRETE else hard
    return hard, warn, host_reserve


def build(devices: list[Device], host: HostMemory, *, smi_ok: bool,
          measured_bytes: int = 0, override: str = "", saw_output: bool = False) -> Pool:
    """Assemble the pool from what was measured. Pure."""
    kind, confidence, can_size, evidence = classify(
        devices, host, smi_ok, override, saw_output)

    if kind == DISCRETE:
        totals = [d.total_bytes or 0 for d in devices]
        frees = [d.free_bytes or 0 for d in devices]
        # The pool is every card, not the smallest one. A box with two 44 GiB
        # cards has 88 GiB of video memory, and an engine given `--gpus all` —
        # llama.cpp splitting layers, vLLM sharding a tensor-parallel rank per
        # device — spends across all of them. Sizing that box at 44 GiB while
        # ALSO counting what every card already holds is not conservative, it is
        # incoherent: it refused a 14 GiB launch into 19 GiB of free framebuffer
        # and printed "-9.8 GiB" doing it.
        #
        # `device_total_bytes` keeps the per-device figure, because vLLM's
        # --gpu-memory-utilization is a fraction of ONE card and its footprint
        # is that fraction times however many cards the rank spans. That is what
        # `devices=` on the pricers is for.
        total = sum(totals)
        available = sum(frees)
        # memory.used, not the compute-apps sum: that sum misses graphics
        # contexts and other users' processes, and under-reporting occupancy is
        # the permissive direction.
        measured = sum(d.used_bytes or 0 for d in devices)
        free = available
        device_total = min(totals) if totals else 0
    else:
        total = host.total_bytes
        available = host.available_bytes
        free = host.free_bytes
        measured = measured_bytes
        # One pool, and every engine on it spends the same bytes. A fraction
        # here multiplies the whole machine, which is what it always did.
        device_total = total

    hard, warn, host_reserve = reserves_for(kind, total, host.total_bytes)
    return Pool(
        kind=kind, confidence=confidence,
        total_bytes=total, available_bytes=available, free_bytes=free,
        measured_bytes=measured, device_total_bytes=device_total,
        reserve_bytes=hard, warn_reserve_bytes=warn,
        host=host, host_reserve_bytes=host_reserve,
        devices=tuple(devices),
        has_accelerator=kind != NONE,
        can_size=can_size,
        evidence=evidence,
    )


def _apply_addressing(devices: list[Device], modes: dict[int, str]) -> list[Device]:
    from dataclasses import replace

    return [replace(d, addressing_mode=modes.get(d.index, "")) for d in devices]


async def local_pool() -> Pool:
    """This machine's pool, measured now."""
    from app import telemetry

    rc, accel_out = await telemetry.nvidia_smi_result(ACCEL_QUERY)
    _addr_rc, addr_out = await telemetry.nvidia_smi_result(ADDR_QUERY)
    devices = _apply_addressing(parse_accel_csv(accel_out), parse_addressing(addr_out))
    measured = sum(p.used_bytes for p in await telemetry.read_gpu_processes())
    return build(devices, read_meminfo(), smi_ok=rc == 0, measured_bytes=measured,
                 override=settings.accel_mode, saw_output=bool((accel_out or "").strip()))


def from_sections(*, accel: str, addr: str, meminfo: str, accel_ok: bool,
                  override: str = "") -> Pool:
    """A peer's pool, from the sections its telemetry script returned."""
    values: dict[str, int] = {}
    for line in (meminfo or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                values[key] = int(parts[0]) * 1024
            except ValueError:
                continue
    host = HostMemory(
        total_bytes=values.get("MemTotal", 0),
        available_bytes=values.get("MemAvailable", 0),
        free_bytes=values.get("MemFree", 0),
    )
    devices = _apply_addressing(parse_accel_csv(accel), parse_addressing(addr))
    # measured stays 0 for a peer, exactly as the budget already assumes:
    # per-process figures are read from THIS machine's nvidia-smi.
    return build(devices, host, smi_ok=accel_ok, measured_bytes=0, override=override,
                 saw_output=bool((accel or "").strip()))


async def pool_for(node: Any = None) -> Pool:
    """The pool of the machine this node names. The only entry point."""
    if node is None or getattr(node, "docker_host", None) is None:
        return await local_pool()
    from app import nodes as node_registry

    return await node_registry.remote_pool(node)


def cheap_script() -> str:
    """The nvidia-smi half of a peer's telemetry script.

    `timeout` on every call: a wedged driver on a peer must not hold up a node
    sweep, and an empty answer already means "assume unified".
    """
    fields = ",".join(ACCEL_FIELDS)
    return (
        f"timeout 5 nvidia-smi --query-gpu={fields} --format=csv,noheader,nounits "
        "2>/dev/null || true"
    ), (
        "timeout 5 nvidia-smi --query-gpu=index,addressing_mode "
        "--format=csv,noheader,nounits 2>/dev/null || true"
    )


__all__ = [
    "ACCEL_QUERY",
    "ADDR_QUERY",
    "DISCRETE",
    "NONE",
    "UNIFIED",
    "UNKNOWN",
    "Device",
    "Pool",
    "build",
    "cheap_script",
    "classify",
    "from_sections",
    "local_pool",
    "parse_accel_csv",
    "parse_addressing",
    "pool_for",
    "reserves_for",
]
