"""What happens to work that outlived the dashboard process.

Jobs are detached containers precisely so a restart does not lose them, so the
three cases — still running, already finished, gone entirely — each need to end
somewhere sensible rather than leaving a row stuck on "running" forever.
"""

from __future__ import annotations

import asyncio

import pytest

from app import db, docker_ctl, jobs


def _spec(kind: str = "probe") -> jobs.JobSpec:
    return jobs.JobSpec(
        kind=kind, title="recovery probe", image="none", command=["true"],
        env={}, mounts=[], gpu=False,
    )


@pytest.fixture
def orphan():
    """A job row left behind mid-flight, as a crash would leave it."""
    job_id = jobs.manager.create(_spec())
    db.execute("UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
               (jobs.RUNNING, db.now(), job_id))
    return job_id


@pytest.mark.asyncio
async def test_a_job_whose_container_vanished_is_marked_failed(orphan, monkeypatch):
    async def gone(name):
        return docker_ctl.ContainerState(name=name, exists=False)

    monkeypatch.setattr(jobs.docker_ctl, "state", gone)
    await jobs.manager.reconcile()

    row = jobs.manager.get(orphan)
    assert row["status"] == jobs.FAILED
    assert "disappeared" in row["error"]
    assert row["finished_at"]


@pytest.mark.asyncio
async def test_a_still_running_container_is_reattached_not_restarted(orphan, monkeypatch):
    started = asyncio.Event()

    async def alive(name):
        return docker_ctl.ContainerState(
            name=name, exists=True, status="running", running=True, command=["true"]
        )

    async def never_ending(name, **kwargs):
        started.set()
        # Hold the stream open the way a live container's log would.
        await asyncio.Event().wait()
        yield ("", False)  # pragma: no cover

    launched = []

    async def must_not_run(**kwargs):
        launched.append(kwargs)
        return "id"

    monkeypatch.setattr(jobs.docker_ctl, "state", alive)
    monkeypatch.setattr(jobs.docker_ctl, "stream_logs", never_ending)
    monkeypatch.setattr(jobs.docker_ctl, "run_detached", must_not_run)

    await jobs.manager.reconcile()
    await asyncio.wait_for(started.wait(), timeout=2)

    assert jobs.manager.get(orphan)["status"] == jobs.RUNNING
    assert launched == [], "reconcile must adopt the container, never start a second one"
    assert "reattached after restart" in "\n".join(jobs.manager.tail(orphan, 20))
    await jobs.manager.shutdown()


@pytest.mark.asyncio
async def test_reattach_resumes_from_where_the_log_stopped(orphan, monkeypatch):
    asked: dict = {}

    async def alive(name):
        return docker_ctl.ContainerState(
            name=name, exists=True, status="running", running=True, command=["true"]
        )

    async def record(name, **kwargs):
        asked.update(kwargs)
        return
        yield  # pragma: no cover

    monkeypatch.setattr(jobs.docker_ctl, "state", alive)
    monkeypatch.setattr(jobs.docker_ctl, "stream_logs", record)
    jobs.manager.log(orphan, "output written before the crash")

    await jobs.manager.reconcile()
    await asyncio.sleep(0.05)

    # The cursor is what bounds the replay. Pairing it with --tail 0 suppressed
    # all history regardless, silently dropping everything the container printed
    # while the dashboard was down.
    assert asked.get("since"), "reattach must resume from a cursor"
    assert asked.get("tail") == "all", "--since alone bounds it; --tail 0 drops the outage"
    await jobs.manager.shutdown()


@pytest.mark.asyncio
async def test_cancelling_an_adopted_job_actually_stops_it(monkeypatch):
    """An image build has no container to stop, so stopping one was a no-op
    that reported success while the build carried on."""
    job_id = jobs.manager.create(_spec("build"))
    running = asyncio.Event()
    cancelled = asyncio.Event()

    async def work():
        jobs.manager.begin(job_id)
        running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def no_container(name):
        return docker_ctl.ContainerState(name=name, exists=False)

    monkeypatch.setattr(jobs.docker_ctl, "state", no_container)
    jobs.manager.adopt(job_id, work())
    await asyncio.wait_for(running.wait(), timeout=2)

    assert await jobs.manager.cancel(job_id) is True
    await asyncio.wait_for(cancelled.wait(), timeout=2)
    assert jobs.manager.get(job_id)["status"] == jobs.CANCELLED
