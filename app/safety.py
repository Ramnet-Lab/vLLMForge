"""Launch-time memory guard.

The single most destructive mistake available on this box is starting a vLLM
server whose --gpu-memory-utilization does not fit. GPU memory is host memory:
overcommit does not degrade performance, it freezes the machine during CUDA
graph capture, and the kernel OOM killer reacts far too slowly to save an
interactive desktop.

Two independent checks run before any launch:

  1. Static budget. vLLM sizes its allocation as `util * torch_total`, and on a
     unified-memory part torch_total equals MemTotal. So the utils of every
     resident engine sum directly, and the sum must leave a reserve for the OS,
     torch.compile and transient activation spikes.

  2. Live headroom. Even a budget-legal launch is refused if MemAvailable right
     now cannot cover it, because something outside the dashboard may be holding
     memory.

Containers the dashboard did not start still count: their utils are parsed out
of their recorded command line, so a hand-launched `vllm serve` is included in
the budget exactly like a managed one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app import docker_ctl, vllm_spec
from app.config import settings
from app.telemetry import read_gpu_processes, read_meminfo

GIB = 1024 ** 3

UTIL_FLAG = re.compile(r"--gpu[-_]memory[-_]utilization(?:=(\S+))?")
KV_BYTES_FLAG = re.compile(r"--kv[-_]cache[-_]memory(?:[-_]bytes)?(?:=|$)")


def default_util() -> float:
    """vLLM's own default, read from the image's schema so it tracks upgrades."""
    arg = vllm_spec.by_dest().get("gpu_memory_utilization") or {}
    try:
        return float(arg.get("default"))
    except (TypeError, ValueError):
        return 0.92


def parse_util(command: list[str] | None) -> float | None:
    """Pull --gpu-memory-utilization out of a container's argv."""
    if not command:
        return None
    for index, token in enumerate(command):
        match = UTIL_FLAG.fullmatch(token) or UTIL_FLAG.match(token)
        if not match:
            continue
        raw = match.group(1)
        if raw is None and index + 1 < len(command):
            raw = command[index + 1]
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def is_vllm_command(command: list[str] | None) -> bool:
    return bool(command) and any(token == "serve" for token in command) and any(
        "vllm" in token for token in command[:2]
    )


@dataclass
class Tenant:
    name: str
    util: float
    managed: bool
    bytes_committed: int = 0
    implicit: bool = False        # util came from vLLM's default, not the command line
    note: str = ""


@dataclass
class Budget:
    total_bytes: int = 0
    available_bytes: int = 0
    free_bytes: int = 0
    measured_gpu_bytes: int = 0
    reserve_bytes: int = 0
    warn_reserve_bytes: int = 0
    tenants: list[Tenant] = field(default_factory=list)

    @property
    def committed_util(self) -> float:
        return sum(t.util for t in self.tenants)

    @property
    def committed_bytes(self) -> int:
        return int(self.committed_util * self.total_bytes)

    @property
    def occupied_bytes(self) -> int:
        """What is really spoken for.

        The util sum misses anything that is not a vLLM engine — a fine-tuning
        job, a Heretic run, even a browser with a few GiB of GPU surfaces — so
        the measured per-process figure is used whenever it is larger.
        """
        return max(self.committed_bytes, self.measured_gpu_bytes)

    @property
    def occupied_util(self) -> float:
        return self.occupied_bytes / self.total_bytes if self.total_bytes else 0.0

    @property
    def max_util(self) -> float:
        """Largest total utilisation that still leaves the hard reserve."""
        if not self.total_bytes:
            return 0.0
        return max(0.0, 1.0 - self.reserve_bytes / self.total_bytes)

    @property
    def warn_util(self) -> float:
        if not self.total_bytes:
            return 0.0
        return max(0.0, 1.0 - self.warn_reserve_bytes / self.total_bytes)

    @property
    def free_util(self) -> float:
        return max(0.0, self.max_util - self.occupied_util)

    def as_dict(self) -> dict:
        return {
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "free_bytes": self.free_bytes,
            "measured_gpu_bytes": self.measured_gpu_bytes,
            "reserve_bytes": self.reserve_bytes,
            "warn_reserve_bytes": self.warn_reserve_bytes,
            "committed_util": round(self.committed_util, 4),
            "committed_bytes": self.committed_bytes,
            "occupied_bytes": self.occupied_bytes,
            "occupied_util": round(self.occupied_util, 4),
            "max_util": round(self.max_util, 4),
            "warn_util": round(self.warn_util, 4),
            "free_util": round(self.free_util, 4),
            "default_util": default_util(),
            "tenants": [
                {
                    "name": t.name,
                    "util": t.util,
                    "managed": t.managed,
                    "bytes_committed": t.bytes_committed,
                    "implicit": t.implicit,
                    "note": t.note,
                }
                for t in self.tenants
            ],
        }


@dataclass
class Verdict:
    ok: bool
    level: str            # ok | warn | block
    message: str
    budget: dict
    requested_util: float = 0.0
    requested_bytes: int = 0
    suggested_util: float | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "level": self.level,
            "message": self.message,
            "requested_util": round(self.requested_util, 4),
            "requested_bytes": self.requested_bytes,
            "suggested_util": (
                round(self.suggested_util, 3) if self.suggested_util is not None else None
            ),
            "budget": self.budget,
        }


def _gib(value: float) -> str:
    return f"{value / GIB:.1f} GiB"


async def current_budget(exclude: str | None = None) -> Budget:
    """Survey every running container and process that has taken GPU memory."""
    memory = read_meminfo()
    budget = Budget(
        total_bytes=memory.total_bytes,
        available_bytes=memory.available_bytes,
        free_bytes=memory.free_bytes,
        reserve_bytes=int(settings.mem_reserve_gib * GIB),
        warn_reserve_bytes=int(settings.mem_warn_reserve_gib * GIB),
    )

    processes = await read_gpu_processes()
    budget.measured_gpu_bytes = sum(p.used_bytes for p in processes)

    fallback = default_util()
    for row in await docker_ctl.ps(all_containers=False):
        name = str(row.get("Names", ""))
        if not name or (exclude and name == exclude):
            continue
        info = await docker_ctl.state(name)
        if not is_vllm_command(info.command):
            continue

        util = parse_util(info.command)
        implicit = util is None
        note = ""
        if implicit:
            # A serve command with no --gpu-memory-utilization is not free: vLLM
            # applies its own default, which on this host is over 100 GiB.
            util = fallback
            note = f"no --gpu-memory-utilization set, so vLLM uses its default of {fallback:g}"
        if any(KV_BYTES_FLAG.match(token) for token in info.command or ()):
            note = (
                "uses --kv-cache-memory, which overrides the utilisation fraction; "
                "the measured figure is the one to trust"
            )

        budget.tenants.append(
            Tenant(
                name=name,
                util=util,
                managed=name.startswith(settings.container_prefix),
                bytes_committed=int(util * budget.total_bytes),
                implicit=implicit,
                note=note,
            )
        )
    return budget


async def check_launch(util: float | None, *, replacing: str | None = None) -> Verdict:
    """Decide whether a launch at `util` is safe right now."""
    budget = await current_budget(exclude=replacing)
    payload = budget.as_dict()
    fallback = default_util()

    if util is None:
        util = fallback
        preface = (
            f"No --gpu-memory-utilization set, so vLLM will apply its default of {fallback:g}. "
        )
    else:
        preface = ""

    requested_bytes = int(util * budget.total_bytes)
    projected_bytes = budget.occupied_bytes + requested_bytes
    projected = projected_bytes / budget.total_bytes if budget.total_bytes else 0.0
    headroom_after = budget.total_bytes - projected_bytes
    suggested = round(max(0.0, budget.free_util - 0.01), 2) or None

    if budget.tenants:
        tenants = ", ".join(
            f"{t.name}={t.util:g}{' (implied)' if t.implicit else ''}" for t in budget.tenants
        )
    else:
        tenants = "none"

    if projected_bytes > budget.total_bytes - budget.reserve_bytes:
        return Verdict(
            ok=False,
            level="block",
            message=(
                f"{preface}Refusing to launch: {util:g} on top of {_gib(budget.occupied_bytes)} "
                f"already in use ({tenants}) would reserve {_gib(projected_bytes)} of "
                f"{_gib(budget.total_bytes)} host memory, leaving only {_gib(headroom_after)} for "
                f"the OS and torch.compile. This host needs at least {_gib(budget.reserve_bytes)} "
                f"free. Largest safe value right now is {budget.free_util:.2f}."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
            suggested_util=suggested,
        )

    if requested_bytes > budget.available_bytes:
        return Verdict(
            ok=False,
            level="block",
            message=(
                f"{preface}Refusing to launch: {util:g} needs {_gib(requested_bytes)} but only "
                f"{_gib(budget.available_bytes)} is available on the host right now. "
                "Stop something first."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
            suggested_util=suggested,
        )

    if requested_bytes > budget.free_bytes:
        # vLLM's own pre-flight compares against free memory, not available, so
        # a launch can be budget-legal here and still be refused by the engine.
        return Verdict(
            ok=True,
            level="warn",
            message=(
                f"{preface}{util:g} needs {_gib(requested_bytes)} and the host has "
                f"{_gib(budget.available_bytes)} available, but only {_gib(budget.free_bytes)} is "
                "actually free — the rest is reclaimable page cache. vLLM's own guard compares "
                "against free memory and may refuse to start with 'Free memory on device is less "
                "than desired GPU memory utilization'. Dropping caches or stopping something else "
                "resolves it."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
        )

    if projected_bytes > budget.total_bytes - budget.warn_reserve_bytes:
        return Verdict(
            ok=True,
            level="warn",
            message=(
                f"{preface}Tight: total utilisation would be {projected:.2f} "
                f"({_gib(projected_bytes)}), leaving {_gib(headroom_after)}. "
                "Keep --max-num-seqs low; concurrency spikes are what actually kill this box."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
        )

    return Verdict(
        ok=True,
        level="ok",
        message=(
            f"{preface}Fits: {util:g} reserves {_gib(requested_bytes)}; total {projected:.2f} "
            f"leaves {_gib(headroom_after)} for the host."
        ),
        budget=payload,
        requested_util=util,
        requested_bytes=requested_bytes,
    )


async def check_job(estimate_bytes: int, *, label: str) -> Verdict:
    """Pre-flight for work that loads a model outside vLLM.

    Heretic and fine-tuning jobs take no utilisation fraction — they simply
    allocate what the model needs — so the question is whether the estimate fits
    beside whatever is already resident.
    """
    budget = await current_budget()
    payload = budget.as_dict()
    headroom = budget.total_bytes - budget.occupied_bytes - budget.reserve_bytes
    tenants = ", ".join(f"{t.name}={t.util:g}" for t in budget.tenants) or "nothing"

    if estimate_bytes > headroom:
        return Verdict(
            ok=False,
            level="block",
            message=(
                f"{label} needs roughly {_gib(estimate_bytes)}, but only {_gib(headroom)} is "
                f"unspoken-for ({tenants} already resident, {_gib(budget.reserve_bytes)} reserved "
                "for the OS). Stop a server first, or use a smaller model or 4-bit quantisation."
            ),
            budget=payload,
            requested_bytes=estimate_bytes,
        )

    if estimate_bytes > budget.available_bytes * 0.8:
        return Verdict(
            ok=True,
            level="warn",
            message=(
                f"{label} needs roughly {_gib(estimate_bytes)} against "
                f"{_gib(budget.available_bytes)} "
                "available. The estimate is approximate and peak usage happens at save time, when "
                "the base model is reloaded to merge — leave room."
            ),
            budget=payload,
            requested_bytes=estimate_bytes,
        )

    return Verdict(
        ok=True,
        level="ok",
        message=(
            f"{label} needs roughly {_gib(estimate_bytes)}; {_gib(headroom)} is unspoken-for."
        ),
        budget=payload,
        requested_bytes=estimate_bytes,
    )
