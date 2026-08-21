"""Long-running work: downloads, fine-tuning runs, Heretic runs, image builds.

Every job is one detached container. That is deliberate — a job survives a
dashboard restart, and reattaching is just `docker logs --since`. Output is
written to a file under the state dir as well as fanned out over the event
broker, so a browser that connects late still sees the whole run.

Per-kind line parsers turn raw output into structured progress (percent, step,
loss, trial number) without the workers needing a side channel.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import db, docker_ctl, events
from app.config import settings

PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED = (
    "pending", "running", "succeeded", "failed", "cancelled"
)
TERMINAL = {SUCCEEDED, FAILED, CANCELLED}

ProgressParser = Callable[[str, dict], dict | None]

_parsers: dict[str, ProgressParser] = {}


def register_parser(kind: str) -> Callable[[ProgressParser], ProgressParser]:
    def decorate(fn: ProgressParser) -> ProgressParser:
        _parsers[kind] = fn
        return fn
    return decorate


# Progress bars rewrite one line with \r; keep only the final state of each.
_CR = re.compile(r"\r+")


def split_carriage(line: str) -> str:
    parts = _CR.split(line)
    return parts[-1] if parts else line


@dataclass
class JobSpec:
    kind: str
    title: str
    image: str
    command: Sequence[str]
    env: dict[str, str]
    mounts: Sequence[docker_ctl.Mount]
    gpu: bool = True
    entrypoint: str | None = None
    workdir: str | None = None
    network: str = "host"
    meta: dict[str, Any] | None = None


class JobManager:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()

    # --- persistence ---------------------------------------------------

    @staticmethod
    def _row(job_id: str) -> dict | None:
        return db.hydrate(
            db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,)),
            ("spec", "result", "progress"),
        )

    def get(self, job_id: str) -> dict | None:
        row = self._row(job_id)
        if row:
            row["log_path"] = str(db.log_path(job_id))
        return row

    @staticmethod
    def list(kind: str | None = None, limit: int = 100) -> list[dict]:
        if kind:
            rows = db.query(
                "SELECT * FROM jobs WHERE kind = ? ORDER BY created_at DESC LIMIT ?", (kind, limit)
            )
        else:
            rows = db.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [db.hydrate(row, ("spec", "result", "progress")) for row in rows]

    def _update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        for json_field in ("spec", "result", "progress"):
            if json_field in fields:
                fields[json_field] = db.dumps(fields[json_field])
        assignments = ", ".join(f"{key} = ?" for key in fields)
        db.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id))
        row = self.get(job_id)
        if row:
            events.broker.publish_soon(events.JOBS, {"type": "job", "job": row})

    # --- lifecycle ------------------------------------------------------

    def create(self, spec: JobSpec) -> str:
        job_id = uuid.uuid4().hex[:12]
        container = settings.container_name(spec.kind, job_id)
        db.execute(
            "INSERT INTO jobs (id, kind, title, status, spec, result, progress, container_name,"
            " created_at)"
            " VALUES (?, ?, ?, ?, ?, '{}', '{}', ?, ?)",
            (
                job_id,
                spec.kind,
                spec.title,
                PENDING,
                db.dumps(
                    {
                        "image": spec.image,
                        "command": list(spec.command),
                        "env": {k: v for k, v in spec.env.items() if "TOKEN" not in k.upper()},
                        "mounts": [m.as_flag() for m in spec.mounts],
                        "meta": spec.meta or {},
                    }
                ),
                container,
                db.now(),
            ),
        )
        db.log_path(job_id).write_text("")
        return job_id

    def start(self, job_id: str, spec: JobSpec) -> None:
        task = asyncio.create_task(self._run(job_id, spec), name=f"job-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(job_id, None))

    async def submit(self, spec: JobSpec) -> str:
        job_id = self.create(spec)
        self.start(job_id, spec)
        return job_id

    async def _run(self, job_id: str, spec: JobSpec) -> None:
        container = settings.container_name(spec.kind, job_id)
        await docker_ctl.remove(container, force=True)
        argv = docker_ctl.build_run_argv(
            name=container,
            image=spec.image,
            command=list(spec.command),
            env=spec.env,
            mounts=list(spec.mounts),
            gpu=spec.gpu,
            network=spec.network,
            entrypoint=spec.entrypoint,
            workdir=spec.workdir,
        )
        self._append(job_id, f"$ {docker_ctl.preview(argv)}")
        try:
            await docker_ctl.run_detached(
                name=container,
                image=spec.image,
                command=list(spec.command),
                env=spec.env,
                mounts=list(spec.mounts),
                gpu=spec.gpu,
                network=spec.network,
                entrypoint=spec.entrypoint,
                workdir=spec.workdir,
            )
        except docker_ctl.DockerError as exc:
            self._append(job_id, f"failed to start container: {exc}")
            self._update(job_id, status=FAILED, error=str(exc), finished_at=db.now())
            return

        self._update(job_id, status=RUNNING, started_at=db.now())
        await self._follow(job_id, spec.kind, container)

    def _resume_cursor(self, job_id: str) -> str | None:
        """Where to resume a container's output after the dashboard restarted.

        The log file's mtime is the moment we last ingested a line, so asking
        docker for everything since then avoids replaying — and re-parsing —
        output we already have. A few seconds of overlap is deliberate: a
        handful of duplicated lines is a far better failure than a silently
        lost tail.
        """
        path = db.log_path(job_id)
        if not path.exists() or path.stat().st_size == 0:
            return None
        moment = dt.datetime.fromtimestamp(path.stat().st_mtime - 5, tz=dt.UTC)
        return moment.isoformat().replace("+00:00", "Z")

    async def _follow(
        self, job_id: str, kind: str, container: str, *, since: str | None = None
    ) -> None:
        """Tail a running container to completion and record its outcome."""
        parser = _parsers.get(kind)
        progress: dict[str, Any] = (self.get(job_id) or {}).get("progress") or {}
        last_publish = 0.0
        last_transient_write = 0.0
        try:
            async for line, transient in docker_ctl.stream_logs(
                container, tail=0 if since else "all", since=since
            ):
                if not line.strip():
                    continue
                now = time.monotonic()
                if transient:
                    # A live progress bar: show it, but only checkpoint it to the
                    # log file occasionally so a long download does not write a
                    # megabyte of redraws.
                    events.broker.publish_soon(
                        events.job_topic(job_id), {"type": "progress-line", "line": line}
                    )
                    if now - last_transient_write > 5.0:
                        last_transient_write = now
                        self._write(job_id, line)
                else:
                    self._append(job_id, line)
                if parser:
                    update = parser(line, progress)
                    if update:
                        progress.update(update)
                        _dedupe_series(progress)
                        if now - last_publish > 0.4:
                            last_publish = now
                            self._update(job_id, progress=progress)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a broken log pipe must not lose the job
            self._append(job_id, f"[dashboard] log stream ended: {exc!r}")

        state = await docker_ctl.state(container)
        if progress:
            self._update(job_id, progress=progress)

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            self._update(job_id, status=CANCELLED, finished_at=db.now(), exit_code=state.exit_code)
        elif state.oom_killed:
            self._update(
                job_id,
                status=FAILED,
                error="container was OOM-killed — the host ran out of memory",
                exit_code=state.exit_code,
                finished_at=db.now(),
            )
        elif state.exit_code == 0:
            self._update(job_id, status=SUCCEEDED, exit_code=0, finished_at=db.now())
        else:
            tail = self.tail(job_id, 40)
            self._update(
                job_id,
                status=FAILED,
                exit_code=state.exit_code,
                error=(state.error or _first_error(tail) or f"exited with code {state.exit_code}"),
                finished_at=db.now(),
            )
        await docker_ctl.remove(container, force=True)

    async def cancel(self, job_id: str) -> bool:
        row = self.get(job_id)
        if not row or row["status"] in TERMINAL:
            return False
        self._cancelled.add(job_id)
        self._append(job_id, "[dashboard] cancel requested")
        await docker_ctl.stop(row["container_name"], timeout=10)
        task = self._tasks.get(job_id)
        if task is None:
            self._update(job_id, status=CANCELLED, finished_at=db.now())
        return True

    # --- logs -----------------------------------------------------------

    @staticmethod
    def _write(job_id: str, line: str) -> None:
        with db.log_path(job_id).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _append(self, job_id: str, line: str) -> None:
        self._write(job_id, line)
        events.broker.publish_soon(events.job_topic(job_id), {"type": "log", "line": line})

    @staticmethod
    def tail(job_id: str, lines: int = 200) -> list[str]:
        path = db.log_path(job_id)
        if not path.exists():
            return []
        content = path.read_text(errors="replace").splitlines()
        return content[-lines:]

    @staticmethod
    def read_log(job_id: str) -> str:
        path = db.log_path(job_id)
        return path.read_text(errors="replace") if path.exists() else ""

    # --- crash recovery --------------------------------------------------

    async def reconcile(self) -> None:
        """Re-adopt jobs that outlived the dashboard process."""
        for row in db.query("SELECT * FROM jobs WHERE status IN (?, ?)", (RUNNING, PENDING)):
            job_id, kind, container = row["id"], row["kind"], row["container_name"]
            state = await docker_ctl.state(container or "")
            since = self._resume_cursor(job_id)
            if state.running:
                self._append(job_id, "[dashboard] reattached after restart")
                task = asyncio.create_task(self._follow(job_id, kind, container, since=since))
                self._tasks[job_id] = task
                task.add_done_callback(lambda _t, jid=job_id: self._tasks.pop(jid, None))
            elif state.exists:
                await self._follow(job_id, kind, container, since=since)
            else:
                self._update(
                    job_id,
                    status=FAILED,
                    error="container disappeared while the dashboard was down",
                    finished_at=db.now(),
                )

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


# A series entry identifies itself by one of these when it is a dict.
SERIES_KEYS = ("step", "trial", "index")


def _dedupe_series(progress: dict[str, Any]) -> None:
    """Keep accumulating series honest across a replayed reattach.

    Parsers build up series — loss_history as [step, value] pairs, trials as
    dicts carrying a trial number — and a resumed log can hand them the same
    entries twice. The manager owns replay, so it owns making replay idempotent
    rather than asking every parser to defend against it. Later entries win, so
    a corrected value replaces the earlier one.
    """
    for key, value in progress.items():
        if not isinstance(value, list) or not value:
            continue
        is_history = key.endswith("_history")
        identified = [
            entry for entry in value
            if isinstance(entry, dict) and any(k in entry for k in SERIES_KEYS)
        ]
        if not is_history and len(identified) != len(value):
            continue

        by_id: dict[Any, Any] = {}
        for position, entry in enumerate(value):
            if isinstance(entry, dict):
                ident = next((entry[k] for k in SERIES_KEYS if k in entry), position)
            elif isinstance(entry, (list, tuple)) and entry:
                ident = entry[0]
            else:
                ident = position
            by_id[ident] = entry
        progress[key] = list(by_id.values())


def _first_error(lines: list[str]) -> str:
    for line in reversed(lines):
        stripped = line.strip()
        if any(
            marker in stripped
            for marker in ("Error", "error:", "Traceback", "Exception", "CUDA out of memory")
        ):
            return stripped[:400]
    return ""


def output_dir(job_id: str, name: str) -> Path:
    """Where a job's artefacts land on the host, created before the container runs."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "output"
    path = settings.output_dir / f"{safe}-{job_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


manager = JobManager()
