"""Host-memory watchdog.

On a unified-memory host the kernel OOM killer reacts too slowly to save an
interactive session: by the time it fires, the desktop has already been
swap-thrashing for tens of seconds. This watches MemAvailable and kills vLLM
containers first, newest and largest first, so the machine stays usable.

It only ever kills containers that are running a vLLM engine — never the
dashboard, never a fine-tuning job it did not start.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app import accel, docker_ctl, events, safety
from app.config import settings
from app.telemetry import read_meminfo

log = logging.getLogger("llmd.memguard")

MIB = 1024 * 1024
POLL_SECONDS = 2.0
# After a kill, memory takes a moment to come back; don't kill the next victim
# on the very next tick.
COOLDOWN_SECONDS = 15.0

_history: list[dict] = []


def history() -> list[dict]:
    return list(_history)


async def _candidates() -> list[tuple[str, float]]:
    """Running vLLM containers, biggest real footprint first.

    Ranking on the utilisation fraction alone put the worst offender last: a
    container with no --gpu-memory-utilization collapsed to 0.0, when in fact
    vLLM had applied its own default and it was holding more than any of them.
    """
    total = (await accel.pool_for(None)).total_bytes or 1
    out: list[tuple[str, float, int]] = []
    for row in await docker_ctl.ps(all_containers=False):
        name = str(row.get("Names", ""))
        if not name:
            continue
        state = await docker_ctl.state(name)
        if not state.running or not safety.is_vllm_command(state.command):
            continue
        params = safety.command_params(state.command)
        util = params.get("gpu_memory_utilization")
        out.append((
            name,
            safety.default_util() if util is None else float(util),
            safety.footprint(params, total),
        ))
    ranked = sorted(out, key=lambda item: item[2], reverse=True)
    return [(name, util) for name, util, _bytes in ranked]


@dataclass
class Starvation:
    """Which memory is running out, and how close to the floor it is.

    The watchdog exists to get in front of a freeze, so it has to watch the pool
    an engine actually spends. On a unified part that is host RAM. On a discrete
    one it is the framebuffer, and host MemAvailable is the wrong instrument
    entirely: it can read 90 GiB while every device is full, so the old trigger
    either never fired or fired for a reason that had nothing to do with the
    engines.
    """

    kind: str                 # "device" or "host"
    available_bytes: int
    threshold_bytes: int
    pool_kind: str

    @property
    def starved(self) -> bool:
        return bool(self.available_bytes) and self.available_bytes < self.threshold_bytes

    def reason(self, name: str, util: float) -> str:
        label = "device free" if self.kind == "device" else "MemAvailable"
        return (
            f"{label} {self.available_bytes // MIB} MiB below "
            f"{self.threshold_bytes // MIB} MiB — killing {name} (util {util:g})"
        )


async def signal() -> Starvation:
    """Read whichever memory this machine can actually exhaust.

    Falls back to host on anything less than a measured framebuffer, matching
    accel.py's rule: only positive proof of a separate pool is treated as one.
    """
    pool = await accel.pool_for(None)
    if pool.kind == accel.DISCRETE and pool.can_size and pool.available_bytes:
        return Starvation(
            kind="device",
            available_bytes=pool.available_bytes,
            threshold_bytes=settings.memguard_device_threshold_mib * MIB,
            pool_kind=pool.kind,
        )
    memory = read_meminfo()
    return Starvation(
        kind="host",
        available_bytes=memory.available_bytes,
        threshold_bytes=settings.memguard_threshold_mib * MIB,
        pool_kind=pool.kind,
    )


async def _restart_policy(name: str) -> str:
    info = await docker_ctl.inspect(name)
    policy = ((info or {}).get("HostConfig") or {}).get("RestartPolicy") or {}
    return str(policy.get("Name") or "unknown")


async def watch() -> None:
    last_kill = 0.0
    while True:
        try:
            starvation = await signal()
            if starvation.starved and time.monotonic() - last_kill > COOLDOWN_SECONDS:
                victims = await _candidates()
                if victims:
                    name, util = victims[0]
                    reason = starvation.reason(name, util)
                    log.warning(reason)
                    # A container set to `unless-stopped` would come back and
                    # re-reserve the memory just freed, so the policy has to go
                    # — but it is the operator's setting, not ours, so record
                    # what it was for whoever restarts the container.
                    if starvation.kind == "device":
                        # The framebuffer is exactly what these engines spend, so
                        # this reading is about them by construction and there is
                        # no host-side OOM killer that will ever see it.
                        action = "kill"
                    else:
                        action = settings.memguard_host_action.strip().lower()
                        if action not in ("kill", "warn"):
                            # On a discrete GPU this trigger — host MemAvailable —
                            # says nothing about what the engines are holding, and
                            # the kernel OOM killer is a working backstop there
                            # because the desktop is not in the framebuffer. Killing
                            # a serving engine on that signal is a self-inflicted
                            # outage.
                            action = "warn" if starvation.pool_kind == accel.DISCRETE else "kill"
                    if action == "warn":
                        entry = {
                            "ts": time.time(), "container": name, "util": util,
                            "action": "warn", "reason": reason + " (host memory, not "
                            f"{starvation.pool_kind} device memory — nothing killed)",
                        }
                        _history.append(entry)
                        del _history[:-50]
                        await events.broker.publish(
                            events.TELEMETRY, {"type": "memguard", "event": entry})
                        last_kill = time.monotonic()
                        continue

                    previous = await _restart_policy(name)
                    # A pooled engine dies as a unit or not at all. Killing one
                    # rank aborts it on its fixed world size while every other
                    # rank stays resident holding a full share of its machine —
                    # so the watchdog would free one node's worth of memory and
                    # strand the rest, having achieved the outage anyway.
                    from app import servers as server_service

                    casualties = await server_service.engine_containers(name)
                    for victim, host in casualties:
                        await docker_ctl.kill(victim, host=host)
                        await docker_ctl.set_restart_policy(victim, "no", host=host)
                    entry = {
                        "ts": time.time(),
                        "container": name,
                        "containers": [victim for victim, _host in casualties],
                        "util": util,
                        "previous_restart_policy": previous,
                        "reason": reason,
                    }
                    _history.append(entry)
                    del _history[:-50]
                    await events.broker.publish(
                        events.TELEMETRY, {"type": "memguard", "event": entry}
                    )
                    last_kill = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("memguard tick failed")
        await asyncio.sleep(POLL_SECONDS)
