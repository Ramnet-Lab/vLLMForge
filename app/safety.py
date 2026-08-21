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

from app import docker_ctl
from app.config import settings
from app.telemetry import read_meminfo

GIB = 1024 ** 3

UTIL_FLAG = re.compile(r"--gpu[-_]memory[-_]utilization(?:=(\S+))?")


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


@dataclass
class Budget:
    total_bytes: int = 0
    available_bytes: int = 0
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
        return max(0.0, self.max_util - self.committed_util)

    def as_dict(self) -> dict:
        return {
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "reserve_bytes": self.reserve_bytes,
            "warn_reserve_bytes": self.warn_reserve_bytes,
            "committed_util": round(self.committed_util, 4),
            "committed_bytes": self.committed_bytes,
            "max_util": round(self.max_util, 4),
            "warn_util": round(self.warn_util, 4),
            "free_util": round(self.free_util, 4),
            "tenants": [
                {
                    "name": t.name,
                    "util": t.util,
                    "managed": t.managed,
                    "bytes_committed": t.bytes_committed,
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
    """Survey every running container that has committed GPU memory."""
    memory = read_meminfo()
    budget = Budget(
        total_bytes=memory.total_bytes,
        available_bytes=memory.available_bytes,
        reserve_bytes=int(settings.mem_reserve_gib * GIB),
        warn_reserve_bytes=int(settings.mem_warn_reserve_gib * GIB),
    )

    running = [row for row in await docker_ctl.ps(all_containers=False)]
    names = [str(row.get("Names", "")) for row in running if row.get("Names")]
    for name in names:
        if exclude and name == exclude:
            continue
        info = await docker_ctl.state(name)
        util = parse_util(info.command)
        if util is None:
            continue
        budget.tenants.append(
            Tenant(
                name=name,
                util=util,
                managed=name.startswith(settings.container_prefix),
                bytes_committed=int(util * budget.total_bytes),
            )
        )
    return budget


async def check_launch(util: float | None, *, replacing: str | None = None) -> Verdict:
    """Decide whether a launch at `util` is safe right now."""
    budget = await current_budget(exclude=replacing)
    payload = budget.as_dict()

    if util is None:
        return Verdict(
            ok=True,
            level="warn",
            message=(
                "No --gpu-memory-utilization set, so vLLM will use its own default of 0.92. "
                f"On this unified-memory host that reserves {_gib(0.92 * budget.total_bytes)} "
                "and will very likely lock the machine. Set an explicit value."
            ),
            budget=payload,
        )

    requested_bytes = int(util * budget.total_bytes)
    projected = budget.committed_util + util
    headroom_after = budget.total_bytes - int(projected * budget.total_bytes)
    suggested = round(max(0.0, budget.free_util - 0.01), 2) or None

    tenants = ", ".join(f"{t.name}={t.util:g}" for t in budget.tenants) or "none"

    if projected > budget.max_util:
        return Verdict(
            ok=False,
            level="block",
            message=(
                f"Refusing to launch: {util:g} on top of already-committed {budget.committed_util:g} "
                f"({tenants}) would reserve {_gib(projected * budget.total_bytes)} of "
                f"{_gib(budget.total_bytes)} host memory, leaving only {_gib(headroom_after)} for the OS "
                f"and torch.compile. This host needs at least {_gib(budget.reserve_bytes)} free. "
                f"Largest safe value right now is {budget.free_util:.2f}."
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
                f"Refusing to launch: {util:g} needs {_gib(requested_bytes)} but only "
                f"{_gib(budget.available_bytes)} is available on the host right now. "
                "Stop something first."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
            suggested_util=suggested,
        )

    if projected > budget.warn_util:
        return Verdict(
            ok=True,
            level="warn",
            message=(
                f"Tight: total utilisation would be {projected:.2f} "
                f"({_gib(projected * budget.total_bytes)}), leaving {_gib(headroom_after)}. "
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
            f"Fits: {util:g} reserves {_gib(requested_bytes)}; total {projected:.2f} "
            f"leaves {_gib(headroom_after)} for the host."
        ),
        budget=payload,
        requested_util=util,
        requested_bytes=requested_bytes,
    )
