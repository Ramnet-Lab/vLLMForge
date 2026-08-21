"""Host and accelerator telemetry.

GB10 is a unified-memory part: nvidia-smi reports no memory.total/used/free and
no power limit, so "GPU memory" here means host memory plus the per-process
figures that --query-compute-apps does return. Anything that reports [N/A] on
this hardware is filtered out rather than shown as a broken tile.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import asdict, dataclass, field

from app.config import settings

KIB = 1024
GIB = 1024 ** 3

# Fields verified to return real values on GB10. memory.*, power.limit,
# clocks.mem and fan.speed all report [N/A] on this part and are deliberately
# absent — see docs/MEMORY.md.
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
    gpu: dict = field(default_factory=dict)
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


async def _nvidia_smi(args: list[str], timeout: float = 5.0) -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (TimeoutError, OSError):
        return ""
    return out.decode(errors="replace")


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


async def read_gpu() -> dict:
    out = await _nvidia_smi(
        ["--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"]
    )
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


async def read_gpu_processes() -> list[GpuProcess]:
    """Per-process GPU memory. This DOES work on GB10 even though totals do not."""
    out = await _nvidia_smi(
        ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    )
    procs: list[GpuProcess] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            used_mib = int(float(parts[1]))
            procs.append(GpuProcess(pid=int(parts[0]), used_bytes=used_mib * 1024 * 1024))
        except ValueError:
            continue
    return procs


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
    memory, gpu, procs = await asyncio.gather(
        asyncio.to_thread(read_meminfo), read_gpu(), read_gpu_processes()
    )
    load, cpus = read_load()
    snap = Snapshot(
        memory={
            **asdict(memory),
            "used_bytes": memory.used_bytes,
            "used_fraction": memory.used_fraction,
        },
        gpu=gpu,
        gpu_processes=[asdict(p) for p in procs],
        load=load,
        cpu_count=cpus,
        disk=read_disk(),
        containers=containers or [],
    )
    return asdict(snap)
