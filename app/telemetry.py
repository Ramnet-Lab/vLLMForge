"""Host and accelerator telemetry.

Two memories are reported, and which one matters depends on the hardware.

On a discrete GPU the framebuffer is a separate pool: nvidia-smi answers
memory.total/used/free per device, and that — not host RAM — is what an engine
spends. On a unified part such as GB10 those fields come back [N/A], there is no
separate pool to report, and host memory *is* the accelerator's memory.

So `vram` is filled in only when the driver gives real per-device figures, and
is empty otherwise. A consumer that finds it empty should fall back to `memory`,
which is host RAM and always truthful. Anything reporting [N/A] is filtered out
rather than shown as a broken tile.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import asdict, dataclass, field

from app.config import settings

KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3

# The per-device row every part answers. memory.* is queried separately (see
# DEVICE_FIELDS) because a unified part reports [N/A] for it while still
# answering everything here — mixing the two would throw away a good row for a
# field that was never going to arrive. power.limit, clocks.mem and fan.speed
# are [N/A] on GB10 and deliberately absent — see docs/MEMORY.md.
GPU_FIELDS = [
    "name",
    "driver_version",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "pstate",
]

_NUMERIC = {"utilization.gpu", "utilization.memory", "temperature.gpu", "power.draw", "clocks.sm"}


@dataclass
class HostMemory:
    total_bytes: int = 0
    available_bytes: int = 0
    free_bytes: int = 0
    swap_total_bytes: int = 0
    swap_free_bytes: int = 0

    @property
    def used_bytes(self) -> int:
        return max(0, self.total_bytes - self.available_bytes)

    @property
    def used_fraction(self) -> float:
        return self.used_bytes / self.total_bytes if self.total_bytes else 0.0


@dataclass
class GpuProcess:
    pid: int
    used_bytes: int
    name: str = ""


@dataclass
class Snapshot:
    ts: float = field(default_factory=time.time)
    memory: dict = field(default_factory=dict)
    # Per-device framebuffer. Empty on a unified part, where `memory` is the
    # accelerator's memory and there is nothing separate to report.
    vram: dict = field(default_factory=dict)
    gpu: dict = field(default_factory=dict)
    # One entry per device, for anything that must not average two cards
    # together or silently describe only the first.
    gpus: list = field(default_factory=list)
    gpu_processes: list = field(default_factory=list)
    load: list = field(default_factory=list)
    cpu_count: int = 0
    disk: dict = field(default_factory=dict)
    containers: list = field(default_factory=list)


def read_meminfo() -> HostMemory:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        values[key] = int(parts[0]) * KIB
                    except ValueError:
                        continue
    except OSError:
        return HostMemory()
    return HostMemory(
        total_bytes=values.get("MemTotal", 0),
        available_bytes=values.get("MemAvailable", 0),
        free_bytes=values.get("MemFree", 0),
        swap_total_bytes=values.get("SwapTotal", 0),
        swap_free_bytes=values.get("SwapFree", 0),
    )


def read_load() -> tuple[list[float], int]:
    try:
        with open("/proc/loadavg", encoding="utf-8") as handle:
            parts = handle.read().split()
        return [float(p) for p in parts[:3]], _cpu_count()
    except (OSError, ValueError):
        return [0.0, 0.0, 0.0], _cpu_count()


def _cpu_count() -> int:
    try:
        return len(__import__("os").sched_getaffinity(0))
    except (AttributeError, OSError):
        return __import__("os").cpu_count() or 1


async def nvidia_smi_result(args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """(returncode, stdout). 127 when there is no nvidia-smi to run.

    Detection turns on the difference between "the driver answered [N/A]" and
    "there was nothing to ask", and the old helper flattened both to "". Callers
    that only want the text keep using _nvidia_smi.
    """
    if not shutil.which("nvidia-smi"):
        return 127, ""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (TimeoutError, OSError):
        return 124, ""
    return proc.returncode or 0, out.decode(errors="replace")


async def _nvidia_smi(args: list[str], timeout: float = 5.0) -> str:
    _rc, out = await nvidia_smi_result(args, timeout)
    return out


def _coerce(field_name: str, raw: str):
    raw = raw.strip()
    if not raw or raw.startswith("[N/A]") or raw == "[Not Supported]":
        return None
    if field_name in _NUMERIC:
        try:
            return float(raw)
        except ValueError:
            return None
    return raw


GPU_QUERY = ["--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"]
APPS_QUERY = ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]


def parse_gpu_csv(out: str) -> dict:
    """One row of `nvidia-smi --query-gpu`, wherever it came from.

    Split out from the collector so a peer's output, fetched over ssh, goes
    through exactly the same parsing — including dropping the [N/A] fields this
    part reports for anything memory-related.
    """
    line = next((ln for ln in out.splitlines() if ln.strip()), "")
    if not line:
        return {}
    parts = [p.strip() for p in line.split(",")]
    gpu = {}
    # A short row means nvidia-smi omitted trailing fields; take what came back.
    for name, raw in zip(GPU_FIELDS, parts, strict=False):
        value = _coerce(name, raw)
        if value is not None:
            gpu[name.replace(".", "_")] = value
    return gpu


DEVICE_FIELDS = ["index", "name", "memory.total", "memory.used", "memory.free"]
DEVICE_QUERY = ["--query-gpu=" + ",".join(DEVICE_FIELDS), "--format=csv,noheader,nounits"]


def parse_device_memory(out: str) -> list[dict]:
    """Per-device framebuffer figures, or [] when the driver will not say.

    A unified part answers [N/A] for every memory field, which is not a parse
    failure to be logged but the honest answer that no separate pool exists —
    so a row that will not yield numbers is dropped and the caller sees the
    empty list it uses to fall back to host memory.
    """
    devices: list[dict] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(DEVICE_FIELDS):
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[2])) * MIB
            used = int(float(parts[3])) * MIB
            free = int(float(parts[4])) * MIB
        except ValueError:
            continue
        if total <= 0:
            continue
        devices.append({
            "index": index,
            "name": parts[1],
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
        })
    return devices


def summarise_device_memory(devices: list[dict]) -> dict:
    """The two totals a framebuffer has, which are not the same number.

    `total_bytes` sums every device, because that is how much VRAM the machine
    has. `per_device_total_bytes` is the smallest single framebuffer, because
    that is what --gpu-memory-utilization multiplies: the fraction applies to
    one device, and on a mixed set the smallest is the one that runs out first.
    Reporting only the sum is what makes a 2x45 GiB box look like it can take a
    90 GiB model.
    """
    if not devices:
        return {}
    totals = [d["total_bytes"] for d in devices]
    return {
        "total_bytes": sum(totals),
        "used_bytes": sum(d["used_bytes"] for d in devices),
        "free_bytes": sum(d["free_bytes"] for d in devices),
        "per_device_total_bytes": min(totals),
        "device_count": len(devices),
        "devices": devices,
    }


async def read_device_memory() -> dict:
    return summarise_device_memory(parse_device_memory(await _nvidia_smi(DEVICE_QUERY)))


GPUS_FIELDS = ["index"] + GPU_FIELDS
GPUS_QUERY = ["--query-gpu=" + ",".join(GPUS_FIELDS), "--format=csv,noheader,nounits"]


def parse_gpu_rows(out: str) -> list[dict]:
    """Every device nvidia-smi listed, not just the first.

    parse_gpu_csv takes one row because the tiles above show one device. The
    charts must not: on a two-card box the utilisation, temperature and power
    lines were GPU 0's, drawn unlabelled, while the second card could sit at
    100% and 80C without appearing anywhere.
    """
    rows: list[dict] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        row: dict = {}
        for name, raw in zip(GPUS_FIELDS, parts, strict=False):
            value = _coerce(name, raw)
            if value is not None:
                row[name.replace(".", "_")] = value
        try:
            row["index"] = int(str(row.get("index", "")).strip())
        except (TypeError, ValueError):
            continue
        rows.append(row)
    return rows


async def read_gpus() -> list[dict]:
    """Per-device stats with each device's framebuffer merged in by index."""
    raw, memory = await asyncio.gather(_nvidia_smi(GPUS_QUERY), read_device_memory())
    rows = parse_gpu_rows(raw)
    by_index = {d["index"]: d for d in (memory.get("devices") or [])}
    for row in rows:
        found = by_index.get(row["index"])
        if not found:
            continue
        row["memory_total_bytes"] = found["total_bytes"]
        row["memory_used_bytes"] = found["used_bytes"]
        row["memory_free_bytes"] = found["free_bytes"]
        if found["total_bytes"]:
            row["memory_used_pct"] = round(100 * found["used_bytes"] / found["total_bytes"], 1)
    return rows


def parse_compute_apps(out: str) -> list[dict]:
    procs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            procs.append({
                "pid": int(parts[0]),
                "used_bytes": int(float(parts[1])) * 1024 * 1024,
                "name": "",
            })
        except ValueError:
            continue
    return procs


def parse_meminfo(out: str) -> dict:
    values: dict[str, int] = {}
    for line in out.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                values[key] = int(parts[0]) * KIB
            except ValueError:
                continue
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    used = max(0, total - available)
    return {
        "total_bytes": total,
        "available_bytes": available,
        "free_bytes": values.get("MemFree", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
        "used_bytes": used,
        "used_fraction": (used / total) if total else 0.0,
    }


async def read_gpu() -> dict:
    return parse_gpu_csv(await _nvidia_smi(GPU_QUERY))


async def read_gpu_processes() -> list[GpuProcess]:
    """Per-process GPU memory. This DOES work on GB10 even though totals do not."""
    out = await _nvidia_smi(APPS_QUERY)
    return [GpuProcess(pid=p["pid"], used_bytes=p["used_bytes"]) for p in parse_compute_apps(out)]


def read_disk(path=None) -> dict:
    target = path or settings.hf_cache
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return {}
    return {
        "path": str(target),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


async def snapshot(containers: list | None = None) -> dict:
    memory, vram, gpu, gpus, procs = await asyncio.gather(
        asyncio.to_thread(read_meminfo), read_device_memory(), read_gpu(), read_gpus(),
        read_gpu_processes()
    )
    load, cpus = read_load()
    snap = Snapshot(
        memory={
            **asdict(memory),
            "used_bytes": memory.used_bytes,
            "used_fraction": memory.used_fraction,
        },
        vram=vram,
        gpu=gpu,
        gpus=gpus,
        gpu_processes=[asdict(p) for p in procs],
        load=load,
        cpu_count=cpus,
        disk=read_disk(),
        containers=containers or [],
    )
    return asdict(snap)
