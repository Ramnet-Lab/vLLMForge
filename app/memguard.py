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
    """Running vLLM containers with the largest committed util first."""
    out: list[tuple[str, float]] = []
    for row in await docker_ctl.ps(all_containers=False):
        name = str(row.get("Names", ""))
        if not name:
            continue
        state = await docker_ctl.state(name)
        if not state.running or not safety.is_vllm_command(state.command):
            continue
        out.append((name, safety.parse_util(state.command) or 0.0))
    return sorted(out, key=lambda item: item[1], reverse=True)


async def watch() -> None:
    threshold_bytes = settings.memguard_threshold_mib * MIB
    last_kill = 0.0
    while True:
        try:
            memory = read_meminfo()
            if memory.available_bytes and memory.available_bytes < threshold_bytes:
                if time.monotonic() - last_kill > COOLDOWN_SECONDS:
                    victims = await _candidates()
                    if victims:
                        name, util = victims[0]
                        reason = (
                            f"MemAvailable {memory.available_bytes // MIB} MiB below "
                            f"{settings.memguard_threshold_mib} MiB — killing {name} (util {util:g})"
                        )
                        log.warning(reason)
                        await docker_ctl.kill(name)
                        # Stop it coming straight back and re-reserving the memory.
                        await docker_ctl.set_restart_policy(name, "no")
                        entry = {"ts": time.time(), "container": name, "util": util, "reason": reason}
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
