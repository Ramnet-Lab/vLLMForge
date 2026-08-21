"""Job listing, log streaming and cancellation — shared by every job kind."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from app import events, jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
async def list_jobs(kind: str | None = None, limit: int = Query(100, le=500)) -> dict:
    return {"jobs": await asyncio.to_thread(jobs.manager.list, kind, limit)}


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    job = await asyncio.to_thread(jobs.manager.get, job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@router.get("/{job_id}/log", response_class=PlainTextResponse)
async def job_log(job_id: str) -> str:
    if await asyncio.to_thread(jobs.manager.get, job_id) is None:
        raise HTTPException(404, "no such job")
    return await asyncio.to_thread(jobs.manager.read_log, job_id)


@router.get("/{job_id}/stream")
async def stream_job(job_id: str) -> EventSourceResponse:
    """Replay what already happened, then follow live output."""
    job = await asyncio.to_thread(jobs.manager.get, job_id)
    if job is None:
        raise HTTPException(404, "no such job")

    async def generator():
        backlog = await asyncio.to_thread(jobs.manager.tail, job_id, 2000, readable=True)
        for line in backlog:
            yield events.sse("log", {"line": line})
        yield events.sse("status", {"status": job["status"], "progress": job.get("progress") or {}})
        if job["status"] in jobs.TERMINAL:
            yield events.sse("end", {"status": job["status"]})
            return
        async for message in events.broker.subscribe(events.job_topic(job_id)):
            yield events.sse(message.get("type", "log"), message)
            current = await asyncio.to_thread(jobs.manager.get, job_id)
            if current and current["status"] in jobs.TERMINAL:
                yield events.sse("end", {"status": current["status"], "job": current})
                return

    return EventSourceResponse(generator(), ping=15)


@router.get("/stream/all")
async def stream_all() -> EventSourceResponse:
    async def generator():
        async for message in events.broker.subscribe(events.JOBS):
            yield events.sse(message.get("type", "job"), message)

    return EventSourceResponse(generator(), ping=15)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    if await asyncio.to_thread(jobs.manager.get, job_id) is None:
        raise HTTPException(404, "no such job")
    return {"cancelled": await jobs.manager.cancel(job_id)}
