"""In-process pub/sub feeding the browser's EventSource connections.

Every stream in the UI — telemetry ticks, job logs, server status changes — is
one topic here. Subscribers get a bounded queue: a slow browser tab drops old
frames rather than growing the producer's memory without limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

DROPPED = object()


class Broker:
    def __init__(self, maxsize: int = 512) -> None:
        self._topics: dict[str, set[asyncio.Queue]] = {}
        self._maxsize = maxsize
        self._lock = asyncio.Lock()
        # asyncio only holds a weak reference to a running task, so a fire-and-
        # forget publish can be collected before it delivers.
        self._pending: set[asyncio.Task] = set()

    async def publish(self, topic: str, payload: Any) -> None:
        for queue in list(self._topics.get(topic, ())):
            if queue.full():
                # Drop the oldest frame so live data always wins over history.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)

    def publish_soon(self, topic: str, payload: Any) -> None:
        """Publish from synchronous code without awaiting."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.publish(topic, payload))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        async with self._lock:
            self._topics.setdefault(topic, set()).add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                subscribers = self._topics.get(topic)
                if subscribers:
                    subscribers.discard(queue)
                    if not subscribers:
                        self._topics.pop(topic, None)

    def subscriber_count(self, topic: str) -> int:
        return len(self._topics.get(topic, ()))


broker = Broker()


def sse(event: str, data: Any) -> dict[str, str]:
    """Shape a payload the way sse_starlette's EventSourceResponse expects."""
    return {"event": event, "data": json.dumps(data, default=str)}


# --- topic names ---------------------------------------------------------

TELEMETRY = "telemetry"
SERVERS = "servers"


def job_topic(job_id: str) -> str:
    return f"job:{job_id}"


JOBS = "jobs"
