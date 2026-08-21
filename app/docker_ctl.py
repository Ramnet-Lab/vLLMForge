"""Asyncio wrapper around the docker CLI.

The CLI rather than the SDK: every vLLM process on this box is already launched
this way, the flags are identical to the hand-written scripts in the hand-written scripts this replaces,
and it sidesteps socket-permission differences between the dashboard user and
the daemon. Nothing here shells out through a shell, so model ids and file paths
containing odd characters are passed through untouched.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DOCKER = "docker"


class DockerError(RuntimeError):
    def __init__(self, argv: Sequence[str], returncode: int, stderr: str):
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr.strip()
        super().__init__(f"docker {' '.join(argv[:3])}… exited {returncode}: {self.stderr[-800:]}")


@dataclass(frozen=True)
class Mount:
    source: str | Path
    target: str
    read_only: bool = False

    def as_flag(self) -> str:
        suffix = ":ro" if self.read_only else ""
        return f"{self.source}:{self.target}{suffix}"


@dataclass(frozen=True)
class ContainerState:
    name: str
    exists: bool
    status: str = "absent"          # created|running|paused|restarting|removing|exited|dead|absent
    running: bool = False
    exit_code: int | None = None
    oom_killed: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    health: str | None = None
    error: str = ""
    image: str = ""
    command: list[str] | None = None

    @property
    def ui_status(self) -> str:
        """Collapse docker's state machine into something a badge can render."""
        if not self.exists:
            return "absent"
        if self.running:
            if self.health in ("starting", "unhealthy"):
                return self.health
            return "running"
        if self.status == "exited":
            if self.oom_killed:
                return "oom-killed"
            return "exited" if self.exit_code == 0 else "failed"
        return self.status


async def _run(argv: Sequence[str], *, check: bool = True) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    if check and proc.returncode != 0:
        raise DockerError(argv, proc.returncode or -1, stderr)
    return proc.returncode or 0, stdout, stderr


async def version() -> str:
    _, out, _ = await _run([DOCKER, "version", "--format", "{{.Server.Version}}"])
    return out.strip()


async def available() -> bool:
    try:
        await version()
        return True
    except (DockerError, FileNotFoundError, OSError):
        return False


def build_run_argv(
    *,
    name: str,
    image: str,
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
    mounts: Sequence[Mount] = (),
    gpu: bool = True,
    network: str = "host",
    ipc: str = "host",
    detach: bool = True,
    remove: bool = False,
    entrypoint: str | None = None,
    restart: str | None = None,
    workdir: str | None = None,
    shm_size: str | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Assemble a `docker run` invocation.

    Defaults mirror the conventions already in use on this cluster: host
    networking (vLLM binds the port directly), host IPC and unlimited memlock
    (vLLM's shared-memory transport needs both), and the nvidia runtime.
    """
    argv = [DOCKER, "run", "--name", name]
    if detach:
        argv.append("-d")
    if remove:
        argv.append("--rm")
    if gpu:
        argv += ["--runtime", "nvidia", "--gpus", "all"]
    if network:
        argv += ["--network", network]
    if ipc:
        argv += ["--ipc", ipc]
    argv += ["--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]
    if shm_size:
        argv += ["--shm-size", shm_size]
    if restart:
        argv += ["--restart", restart]
    if workdir:
        argv += ["--workdir", workdir]
    if entrypoint is not None:
        argv += ["--entrypoint", entrypoint]
    for mount in mounts:
        argv += ["-v", mount.as_flag()]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    argv += list(extra)
    argv.append(image)
    argv += list(command)
    return argv


async def run_detached(**kwargs) -> str:
    argv = build_run_argv(detach=True, **kwargs)
    _, out, _ = await _run(argv)
    return out.strip()


async def run_capture(
    *,
    image: str,
    command: Sequence[str],
    name: str | None = None,
    timeout: float | None = 300.0,
    **kwargs,
) -> tuple[int, str, str]:
    """Run a throwaway container and collect its output."""
    argv = build_run_argv(
        name=name or f"llmd-tmp-{id(command) & 0xFFFFFF:x}",
        image=image,
        command=command,
        detach=False,
        remove=True,
        **kwargs,
    )
    try:
        return await asyncio.wait_for(_run(argv, check=False), timeout=timeout)
    except asyncio.TimeoutError:
        raise DockerError(argv, -1, f"timed out after {timeout}s") from None


async def inspect(name: str) -> dict | None:
    code, out, _ = await _run([DOCKER, "inspect", name], check=False)
    if code != 0:
        return None
    try:
        payload = json.loads(out)
    except ValueError:
        return None
    return payload[0] if payload else None


async def state(name: str) -> ContainerState:
    info = await inspect(name)
    if info is None:
        return ContainerState(name=name, exists=False)
    st = info.get("State", {}) or {}
    health = (st.get("Health") or {}).get("Status")
    return ContainerState(
        name=name,
        exists=True,
        status=st.get("Status", "unknown"),
        running=bool(st.get("Running")),
        exit_code=st.get("ExitCode"),
        oom_killed=bool(st.get("OOMKilled")),
        started_at=st.get("StartedAt"),
        finished_at=st.get("FinishedAt"),
        health=health,
        error=st.get("Error", "") or "",
        image=(info.get("Config", {}) or {}).get("Image", ""),
        command=(info.get("Config", {}) or {}).get("Cmd"),
    )


async def states(names: Sequence[str]) -> dict[str, ContainerState]:
    if not names:
        return {}
    results = await asyncio.gather(*(state(n) for n in names))
    return {s.name: s for s in results}


async def ps(prefix: str | None = None, *, all_containers: bool = True) -> list[dict]:
    argv = [DOCKER, "ps", "--format", "{{json .}}"]
    if all_containers:
        argv.append("-a")
    code, out, _ = await _run(argv, check=False)
    if code != 0:
        return []
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if prefix and not str(row.get("Names", "")).startswith(prefix):
            continue
        rows.append(row)
    return rows


async def stop(name: str, timeout: int = 30) -> None:
    await _run([DOCKER, "stop", "-t", str(timeout), name], check=False)


async def kill(name: str) -> None:
    await _run([DOCKER, "kill", name], check=False)


async def remove(name: str, *, force: bool = True) -> None:
    argv = [DOCKER, "rm", name]
    if force:
        argv.insert(2, "-f")
    await _run(argv, check=False)


async def set_restart_policy(name: str, policy: str) -> None:
    await _run([DOCKER, "update", "--restart", policy, name], check=False)


async def logs(name: str, *, tail: int | str = 400) -> str:
    _, out, err = await _run([DOCKER, "logs", "--tail", str(tail), name], check=False)
    return out + err


async def stream_logs(
    name: str,
    *,
    tail: int | str = 200,
    since: str | None = None,
    progress_interval: float = 0.4,
) -> AsyncIterator[tuple[str, bool]]:
    """Follow a container's output as (text, transient) pairs.

    Reading with readline() is not an option here: vLLM, tqdm and huggingface_hub
    all redraw progress with a bare carriage return and no newline, so a
    line-oriented reader blocks for minutes on a single "line" that is really a
    thousand redraws of the same bar. This reads raw chunks instead and splits
    on both terminators. A segment ending in \r is *transient* — the latest
    state of a progress bar — and is emitted at most every `progress_interval`
    seconds so callers can show it without writing it to disk a thousand times.

    stdout and stderr are drained concurrently into one queue; vLLM logs to
    stderr and the container's own output goes to stdout, and a log pane wants
    them interleaved in arrival order.
    """
    argv = [DOCKER, "logs", "-f", "--tail", str(tail)]
    if since:
        argv += ["--since", since]
    argv.append(name)
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    queue: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue(maxsize=8192)

    async def pump(stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        buffer = ""
        last_progress = 0.0
        last_emitted = ""
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            buffer += chunk.decode(errors="replace")
            # Complete records end in \n; everything before the final newline is
            # settled output, with \r-redraws inside collapsed to their last state.
            while "\n" in buffer:
                record, buffer = buffer.split("\n", 1)
                text = record.split("\r")[-1].rstrip()
                if text:
                    await queue.put((text, False))
                    last_emitted = ""
            # What is left is a partial line — usually a live progress bar.
            if "\r" in buffer:
                partial = buffer.split("\r")[-1].rstrip()
                now = asyncio.get_running_loop().time()
                if partial and partial != last_emitted and now - last_progress >= progress_interval:
                    last_progress = now
                    last_emitted = partial
                    await queue.put((partial, True))
        leftover = buffer.split("\r")[-1].rstrip()
        if leftover:
            await queue.put((leftover, False))

    pumps = [asyncio.create_task(pump(proc.stdout)), asyncio.create_task(pump(proc.stderr))]

    async def closer() -> None:
        await asyncio.gather(*pumps, return_exceptions=True)
        await queue.put(None)

    closing = asyncio.create_task(closer())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        for task in (*pumps, closing):
            task.cancel()
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()


async def image_exists(tag: str) -> bool:
    code, _, _ = await _run([DOCKER, "image", "inspect", tag], check=False)
    return code == 0


async def build_image(
    tag: str, dockerfile: Path, context: Path, *, build_args: Mapping[str, str] | None = None
) -> AsyncIterator[str]:
    argv = [DOCKER, "build", "-t", tag, "-f", str(dockerfile)]
    for key, value in (build_args or {}).items():
        argv += ["--build-arg", f"{key}={value}"]
    argv.append(str(context))
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        yield raw.decode(errors="replace").rstrip("\n")
    await proc.wait()
    if proc.returncode != 0:
        raise DockerError(argv, proc.returncode or -1, "build failed")


def preview(argv: Sequence[str]) -> str:
    """A copy-pasteable rendering of a command, for the UI's 'show command' box."""
    return " ".join(shlex.quote(part) for part in argv)
