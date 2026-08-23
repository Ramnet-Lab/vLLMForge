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

from app import docker_ctl, events, safety
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
    total = read_meminfo().total_bytes or 1
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


async def _restart_policy(name: str) -> str:
    info = await docker_ctl.inspect(name)
    policy = ((info or {}).get("HostConfig") or {}).get("RestartPolicy") or {}
    return str(policy.get("Name") or "unknown")


async def watch() -> None:
    threshold_bytes = settings.memguard_threshold_mib * MIB
    last_kill = 0.0
    while True:
        try:
            memory = read_meminfo()
            starved = memory.available_bytes and memory.available_bytes < threshold_bytes
            if starved and time.monotonic() - last_kill > COOLDOWN_SECONDS:
                victims = await _candidates()
                if victims:
                    name, util = victims[0]
                    reason = (
                        f"MemAvailable {memory.available_bytes // MIB} MiB below "
                        f"{settings.memguard_threshold_mib} MiB — killing {name} (util {util:g})"
                    )
                    log.warning(reason)
                    # A container set to `unless-stopped` would come back and
                    # re-reserve the memory just freed, so the policy has to go
                    # — but it is the operator's setting, not ours, so record
                    # what it was for whoever restarts the container.
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
