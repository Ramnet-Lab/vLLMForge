"""The per-job stream has to tell a watcher when the job ends.

When it does not, every pane a user opens leaves a connection open for the life
of the tab. Browsers allow about six per origin over HTTP/1.1, so a handful of
watched jobs exhausts the pool and the page stops responding entirely — which
is how this surfaced.
"""

from __future__ import annotations

import asyncio

import pytest

from app import db, events, jobs


@pytest.fixture
def finished_row():
    job_id = jobs.manager.create(
        jobs.JobSpec(kind="probe", title="stream probe", image="none",
                     command=["true"], env={}, mounts=[], gpu=False)
    )
    db.execute("UPDATE jobs SET status = ? WHERE id = ?", (jobs.RUNNING, job_id))
    return job_id


@pytest.mark.asyncio
async def test_a_status_change_reaches_the_per_job_topic(finished_row):
    topic = events.job_topic(finished_row)
    stream = events.broker.subscribe(topic)
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.02)

    jobs.manager.finish(finished_row, ok=True, progress={"percent": 100.0})

    # finish() writes progress and then status, so the terminal frame is the
    # second one; what matters is that it arrives at all.
    seen = [await asyncio.wait_for(pending, timeout=2)]
    while seen[-1]["status"] != jobs.SUCCEEDED and len(seen) < 5:
        seen.append(await asyncio.wait_for(anext(stream), timeout=2))

    assert seen[-1]["type"] == "progress"
    assert seen[-1]["status"] == jobs.SUCCEEDED
    assert seen[-1]["progress"]["percent"] == 100.0
    await stream.aclose()


@pytest.mark.asyncio
async def test_progress_updates_reach_the_per_job_topic(finished_row):
    topic = events.job_topic(finished_row)
    stream = events.broker.subscribe(topic)
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.02)

    jobs.manager.report(finished_row, {"percent": 42.0, "phase": "downloading"})
    message = await asyncio.wait_for(pending, timeout=2)

    # A pane following one job must see the numbers its parser produced, not sit
    # at zero because they were only published to the global topic.
    assert message["progress"]["percent"] == 42.0
    assert message["status"] == jobs.RUNNING
    await stream.aclose()


def test_the_stream_endpoint_ends_for_an_already_terminal_job(finished_row):
    from fastapi.testclient import TestClient

    from app.main import app

    jobs.manager.finish(finished_row, ok=True)
    with TestClient(app) as client, client.stream("GET", f"/api/jobs/{finished_row}/stream") as r:
        seen = []
        for line in r.iter_lines():
            if line.startswith("event:"):
                seen.append(line.split(":", 1)[1].strip())
                if seen[-1] == "end":
                    break
    assert "end" in seen, "the connection would stay open for the life of the tab"
