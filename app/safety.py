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

import math
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from app import docker_ctl, vllm_spec
from app.config import settings

GIB = 1024 ** 3

UTIL_FLAG = re.compile(r"--gpu[-_]memory[-_]utilization(?:=(\S+))?")
KV_BYTES_FLAG = re.compile(r"--kv[-_]cache[-_]memory(?:[-_]bytes)?(?:=(\S+))?$")
OFFLOAD_FLAG = re.compile(r"--cpu[-_]offload[-_]gb(?:=(\S+))?$")


def flag_value(command: list[str] | None, pattern: re.Pattern[str]) -> str | None:
    """The value of a flag in a container's argv, whether joined by = or spaced."""
    command = argv_of(command)
    if not command:
        return None
    for index, token in enumerate(command):
        match = pattern.fullmatch(token)
        if not match:
            continue
        if match.group(1) is not None:
            return match.group(1)
        if index + 1 < len(command):
            return command[index + 1]
        return None
    return None


# How close to MemAvailable a request may sit before it is worth a warning.
# Recommendations keep the same headroom; see app/recommend.py UTIL_MARGIN.
DRIFT_MARGIN = 0.03


def default_util() -> float:
    """vLLM's own default, read from the image's schema so it tracks upgrades."""
    arg = vllm_spec.by_dest().get("gpu_memory_utilization") or {}
    try:
        return float(arg.get("default"))
    except (TypeError, ValueError):
        return 0.92


def parse_util(command: list[str] | None) -> float | None:
    """Pull --gpu-memory-utilization out of a container's argv."""
    try:
        return float(flag_value(command, UTIL_FLAG))
    except (TypeError, ValueError):
        return None


def command_params(command: list[str] | None) -> dict[str, Any]:
    """The memory-relevant flags of a running container, shaped like stored args.

    Reading them back out of the argv means a hand-launched container is
    accounted exactly the way a managed one is.
    """
    params: dict[str, Any] = {}
    util = parse_util(command)
    if util is not None:
        params["gpu_memory_utilization"] = util
    kv = flag_value(command, KV_BYTES_FLAG)
    if kv is not None:
        params["kv_cache_memory_bytes"] = kv
    offload = flag_value(command, OFFLOAD_FLAG)
    if offload is not None:
        params["cpu_offload_gb"] = offload
    return params


def footprint(params: dict[str, Any], total_bytes: int) -> int:
    return vllm_spec.footprint_bytes(params, total_bytes, default_util=default_util())


# `vllm serve` as a word, anywhere in a string that may be a whole shell line.
_VLLM_SERVE = re.compile(r"(?:^|[\s;&|])vllm\s+serve(?:\s|$)")


def argv_of(command: list[str] | None) -> list[str]:
    """A container's command as tokens, whether or not a shell is in the way.

    A pooled engine is launched as `bash -lc "ray start ... && exec vllm serve
    ..."` — one element, because the Ray head and the engine have to share a
    container. Treating that as opaque made the dashboard's own pooled servers
    weigh nothing in the budget, so a second launch could be admitted on top of
    memory that was already spoken for.
    """
    if not command:
        return []
    for token in command:
        if not _VLLM_SERVE.search(token):
            continue
        if len(token.split()) == 1:
            break
        try:
            parts = shlex.split(token)
        except ValueError:
            continue
        for index, part in enumerate(parts):
            if part.endswith("vllm") and parts[index + 1:index + 2] == ["serve"]:
                return parts[index:]
    return list(command)


def is_vllm_command(command: list[str] | None) -> bool:
    argv = argv_of(command)
    return bool(argv) and any(token == "serve" for token in argv) and any(
        "vllm" in token for token in argv[:2]
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
    excluded_bytes: int = 0
    """What the excluded container is measured to be holding. A restart gets its
    own memory back, and nvidia-smi cannot know that — it reports processes, not
    intentions — so it is subtracted here rather than left to inflate a budget
    that is about to change."""
    reserve_bytes: int = 0
    warn_reserve_bytes: int = 0
    tenants: list[Tenant] = field(default_factory=list)
    pool: Any = None
    """The accel.Pool these figures came from: which memory they describe, how
    that was decided, and how sure it is. Additive — every field above keeps
    the meaning it had when they were always host memory."""

    @property
    def committed_bytes(self) -> int:
        # Summed in bytes, not fractions: a tenant using --kv-cache-memory or
        # --cpu-offload-gb takes memory its utilisation fraction never mentions.
        return sum(t.bytes_committed for t in self.tenants)

    @property
    def committed_util(self) -> float:
        return self.committed_bytes / self.total_bytes if self.total_bytes else 0.0

    @property
    def occupied_bytes(self) -> int:
        """What is really spoken for.

        The util sum misses anything that is not a vLLM engine — a fine-tuning
        job, a Heretic run, even a browser with a few GiB of GPU surfaces — so
        the measured per-process figure is used whenever it is larger.

        The container being replaced comes off the measured side only. The
        committed side never had it: current_budget drops it before it reaches
        the tenant list. Taking it off both was invisible while only the local
        node had tenants — measured is read from nvidia-smi and is 0 for a peer,
        so the two figures agreed — but a pooled engine puts a real tenant on
        every node it spans, and there the double subtraction cancelled a peer's
        genuine occupancy and reported a full machine as empty.
        """
        return max(0, max(self.committed_bytes, self.measured_gpu_bytes - self.excluded_bytes))

    @property
    def available_after_replacement(self) -> int:
        """MemAvailable as it will be once the container being replaced has gone.

        vLLM compares its request against free memory at the moment the engine
        starts, and by then the old container has been removed — so judging a
        restart against memory that container is still holding refuses launches
        that would have been fine. Same reasoning as excluding it from the
        budget; this is the live half of it, which was missed.
        """
        return min(self.total_bytes, self.available_bytes + self.excluded_bytes)

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
            # Which memory every figure above describes, so a reader never has
            # to assume it is host RAM. Additive: no key above changed.
            "pool_kind": getattr(self.pool, "kind", "unified"),
            "pool_confidence": getattr(self.pool, "confidence", "assumed"),
            "pool_devices": getattr(self.pool, "device_count", 0),
            "pool_evidence": list(getattr(self.pool, "evidence", ()) or ()),
            "host_available_bytes": getattr(
                getattr(self.pool, "host", None), "available_bytes", self.available_bytes),
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


async def current_budget(exclude: str | None = None, node: Any = None) -> Budget:
    """Survey every running container and process that has taken GPU memory.

    `node` is an app.nodes.Node; omitted means this machine. A launch aimed at a
    peer has to be measured against that peer's memory, not against ours.
    """
    from app import accel

    host = getattr(node, "docker_host", None)
    # One probe decides what these numbers describe. On a unified machine it
    # answers with MemTotal and MemAvailable, byte for byte, and this is the
    # same budget it always was; on a machine whose GPU has its own memory it
    # answers with the framebuffer, which is what the utilisation fraction
    # actually multiplies there. Nothing below needs to know which it got.
    pool = await accel.pool_for(node)

    budget = Budget(
        total_bytes=pool.total_bytes,
        available_bytes=pool.available_bytes,
        free_bytes=pool.free_bytes,
        reserve_bytes=pool.reserve_bytes,
        warn_reserve_bytes=pool.warn_reserve_bytes,
        pool=pool,
    )

    # Per-process accounting comes from nvidia-smi, which reports on THIS
    # machine. Counting it against a peer made an idle peer look half full and
    # refused launches it had ample room for.
    if host is None:
        budget.measured_gpu_bytes = pool.measured_bytes

    fallback = default_util()
    for row in await docker_ctl.ps(all_containers=False, host=host):
        name = str(row.get("Names", ""))
        if not name:
            continue
        info = await docker_ctl.state(name, host)
        if not is_vllm_command(info.command):
            continue

        if exclude and name == exclude:
            # The container this survey is about to replace. It is not a tenant,
            # and what it holds is memory the replacement can have back — so it
            # comes off the measured total too, which otherwise counts the very
            # engine being restarted against its own restart.
            budget.excluded_bytes = footprint(command_params(info.command), pool.total_bytes)
            continue

        params = command_params(info.command)
        util = params.get("gpu_memory_utilization")
        implicit = util is None
        notes = []
        if implicit:
            # A serve command with no --gpu-memory-utilization is not free: vLLM
            # applies its own default, which on this host is over 100 GiB.
            util = fallback
            notes.append(
                f"no --gpu-memory-utilization set, so vLLM uses its default of {fallback:g}"
            )
        if "kv_cache_memory_bytes" in params:
            notes.append("--kv-cache-memory overrides the utilisation fraction")
        if "cpu_offload_gb" in params:
            notes.append("--cpu-offload-gb lands in the same unified pool")

        budget.tenants.append(
            Tenant(
                name=name,
                util=util,
                managed=name.startswith(settings.container_prefix),
                bytes_committed=footprint(params, budget.total_bytes),
                implicit=implicit,
                note="; ".join(notes),
            )
        )
    return budget


async def check_launch(
    util: float | None,
    *,
    replacing: str | None = None,
    params: dict[str, Any] | None = None,
    node: Any = None,
) -> Verdict:
    """Decide whether a launch is safe right now.

    Pass `params` — the whole stored argument dict — wherever it is available:
    the utilisation fraction alone does not describe what a config will take.
    """
    budget = await current_budget(exclude=replacing, node=node)
    payload = budget.as_dict()
    fallback = default_util()

    if util is None and params:
        util = vllm_spec.gpu_memory_utilization(params)
    if util is None:
        util = fallback
        preface = (
            f"No --gpu-memory-utilization set, so vLLM will apply its default of {fallback:g}. "
        )
    else:
        preface = ""

    requested_bytes = footprint({**(params or {}), "gpu_memory_utilization": util},
                                budget.total_bytes)
    if params and requested_bytes > int(util * budget.total_bytes):
        preface += (
            "This config also sets "
            + " and ".join(
                flag for flag, key in (
                    ("--kv-cache-memory", "kv_cache_memory_bytes"),
                    ("--cpu-offload-gb", "cpu_offload_gb"),
                ) if key in params
            )
            + ", so its real footprint is larger than the fraction implies. "
        )
    projected_bytes = budget.occupied_bytes + requested_bytes
    projected = projected_bytes / budget.total_bytes if budget.total_bytes else 0.0
    headroom_after = budget.total_bytes - projected_bytes
    # Floor, never round: suggesting 0.06 when 0.057 is the ceiling hands the
    # user a value the very next check refuses.
    safe_now = math.floor(max(0.0, budget.free_util) * 100) / 100
    suggested = safe_now or None

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

    live_available = budget.available_after_replacement
    if requested_bytes > live_available:
        # This is vLLM's own gate, not a policy of ours. On a unified-memory box
        # the engine reads psutil.virtual_memory().available — MemAvailable — and
        # refuses when ceil(total * util) exceeds it.
        return Verdict(
            ok=False,
            level="block",
            message=(
                f"{preface}Refusing to launch: {util:g} needs {_gib(requested_bytes)} but only "
                f"{_gib(live_available)} is available on the host right now — which is "
                "the number vLLM itself compares against, so it would refuse to start. "
                "Stop something first."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
            suggested_util=suggested,
        )

    # MemAvailable moves while an image is pulled and weights are read, so a
    # request that only just fits now can be refused by the time the engine
    # starts. That is worth a word; page cache is not — vLLM counts reclaimable
    # memory as free on this hardware, and telling the operator to drop caches
    # for a launch that would have succeeded is how an afternoon disappears.
    #
    #   vllm/utils/mem_utils.py: if current_platform.is_integrated_gpu(...):
    #       self.free_memory = psutil.virtual_memory().available
    #
    if requested_bytes > live_available * (1 - DRIFT_MARGIN):
        return Verdict(
            ok=True,
            level="warn",
            message=(
                f"{preface}{util:g} needs {_gib(requested_bytes)} of the "
                f"{_gib(live_available)} available — close enough that a little more "
                "activity before the engine starts would push it over vLLM's own check. "
                f"{max(0.0, suggested or 0):.2f} leaves room for that."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
            suggested_util=suggested,
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
