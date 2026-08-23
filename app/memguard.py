"""Host-memory watchdog.

On a unified-memory host the kernel OOM killer reacts too slowly to save an
interactive session: by the time it fires, the desktop has already been
swap-thrashing for tens of seconds. This watches the memory an engine actually
spends and kills the largest resident engine first, so the machine stays usable.

The invariant is not "kill only vLLM" — it is **kill only what we understand
well enough to know what dies with it**. That distinction matters now that there
are two engines: the watchdog asks `engines.recognise()`, so a hand-launched
llama-server holding 40 GiB is a candidate like any other. Before it was, the
failure was concrete and inverted — the real offender was invisible, so the
threshold was crossed and a *healthy vLLM engine* was killed in its place while
the memory stayed exactly where it was.

Everything else is still off limits: the dashboard, a fine-tuning run, a Heretic
job, and any container whose command line names no engine this build knows.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app import accel, docker_ctl, engines, events
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


async def _candidates() -> list[tuple[str, float | None, int]]:
    """Running engine containers, biggest real footprint first.

    Ranking on the utilisation fraction alone put the worst offender last: a
    container with no --gpu-memory-utilization collapsed to 0.0, when in fact
    vLLM had applied its own default and it was holding more than any of them.
    Ranking on bytes fixed that, and it is also what makes ranking across two
    engines work at all — one of them has no fraction to rank on.

    The bytes are returned alongside the util rather than discarded: for a
    llama.cpp victim there is no fraction to name in the kill notice, and "util
    0.92" would be a fabrication about a flag that engine does not have.
    """
    total = (await accel.pool_for(None)).total_bytes or 1
    out: list[tuple[str, float | None, int]] = []
    for row in await docker_ctl.ps(all_containers=False):
        name = str(row.get("Names", ""))
        if not name:
            continue
        state = await docker_ctl.state(name)
        engine, argv = engines.identify(state)
        if not state.running or engine is None:
            continue
        # The argv that matched, not Config.Cmd: an engine whose binary is in
        # its image's ENTRYPOINT would otherwise rank at zero bytes and never
        # be chosen, which is the inversion this widening exists to fix.
        params = await engine.resolve(engine.command_params(argv))
        util = engine.declared_util(params)
        if util is None:
            # Reproduces the old two-step exactly for vLLM — an undeclared
            # fraction is the engine's default, not zero — and leaves None for an
            # engine that has no default because it has no fraction.
            util = engine.implicit_util()
        out.append((name, None if util is None else float(util),
                    engine.footprint_bytes(params, total)))
    return sorted(out, key=lambda item: item[2], reverse=True)


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

    def reason(self, name: str, util: float | None, held_bytes: int = 0) -> str:
        label = "device free" if self.kind == "device" else "MemAvailable"
        # An engine that declares a fraction is named by it, exactly as before.
        # One that does not is named by what it is holding, because inventing a
        # fraction for it would put a number in the kill history and in the
        # Overview's live feed that describes a flag it never had.
        held = f"util {util:g}" if util is not None else f"holding {held_bytes // MIB} MiB"
        return (
            f"{label} {self.available_bytes // MIB} MiB below "
            f"{self.threshold_bytes // MIB} MiB — killing {name} ({held})"
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
                    name, util, held_bytes = victims[0]
                    reason = starvation.reason(name, util, held_bytes)
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
                            "bytes_committed": held_bytes,
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
                        "bytes_committed": held_bytes,
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
