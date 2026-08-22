"""Copy a model from this machine's cache to a peer's, over the fast link.

Pulling the same 50 GB from HuggingFace onto every node is the obvious way and
the wrong one when the boxes are wired together: the cluster fabric here
measures about 14 Gb/s on a plain TCP stream, and the bytes are already on
disk. rsync over ssh reuses the key that is already configured.

The rsync runs ON THE DESTINATION, pulling. That is not arbitrary — this
machine's cache is root-owned (containers write it as root), so a push would
have to become root somewhere. A peer whose cache its own user can write needs
no privilege at all, and where the peer's cache is also root-owned the job says
so rather than failing halfway through with permission errors.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from typing import Any

from app import db, hf, jobs, nodes
from app.config import settings

KIND = "sync"

# rsync --info=progress2 emits a single rewritten line:
#   1,234,567,890  42%  512.34MB/s    0:01:23  (xfr#12, to-chk=30/94)
PROGRESS = re.compile(
    r"([\d,]+)\s+(\d+)%\s+([\d.]+)([kMG]?B/s)\s+(\d+:\d+:\d+)"
    r"(?:\s+\(xfr#(\d+),\s*to-chk=(\d+)/(\d+)\))?"
)
UNIT = {"B/s": 1, "kB/s": 10 ** 3, "MB/s": 10 ** 6, "GB/s": 10 ** 9}


@jobs.register_parser(KIND)
def parse_sync(line: str, progress: dict) -> dict | None:
    match = PROGRESS.search(line)
    if not match:
        if "speedup is" in line:
            return {"phase": "done", "percent": 100.0}
        return None
    transferred, percent, rate, unit, elapsed, _xfr, remaining, total = match.groups()
    update: dict[str, Any] = {
        "phase": "copying",
        "transferred_bytes": int(transferred.replace(",", "")),
        "percent": float(percent),
        "speed_bps": float(rate) * UNIT.get(unit, 1),
        "elapsed": elapsed,
    }
    if total:
        update["files_total"] = int(total)
        update["files_done"] = int(total) - int(remaining)
    return update


def repo_dir_name(repo_id: str, repo_type: str = "model") -> str:
    _segment, prefix = hf._repo_type(repo_type)
    hf.check_repo_id(repo_id)
    return f"{prefix}{repo_id.replace('/', '--')}"


async def _run_on(node: nodes.Node, command: str, timeout: float = 60) -> tuple[int, str]:
    return await nodes._ssh(node.name or node.address, command)


async def preflight(repo_id: str, target: nodes.Node, repo_type: str = "model") -> dict:
    """Can this copy work, and is it even needed?"""
    directory = repo_dir_name(repo_id, repo_type)
    source = settings.hf_cache / "hub" / directory
    if not source.is_dir():
        return {"ok": False, "reason": f"{repo_id} is not in this machine's cache"}

    size = await asyncio.to_thread(
        lambda: sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    )

    remote_hub = f"{settings.hf_cache}/hub"
    code, out = await _run_on(
        target,
        f"test -d {shlex.quote(remote_hub)} && "
        f"(test -w {shlex.quote(remote_hub)} && echo writable || echo readonly); "
        f"du -sb {shlex.quote(remote_hub + '/' + directory)} 2>/dev/null | cut -f1; "
        f"df -PB1 {shlex.quote(remote_hub)} | tail -1 | awk '{{print $4}}'",
    )
    if code != 0:
        return {"ok": False, "reason": f"could not reach {target.name}: {out.strip()[:200]}"}

    lines = [line.strip() for line in out.splitlines() if line.strip()]
    writable = "writable" in lines
    already = int(lines[1]) if len(lines) > 2 and lines[1].isdigit() else 0
    free = int(lines[-1]) if lines and lines[-1].isdigit() else 0

    if not writable:
        return {
            "ok": False,
            "reason": (
                f"{target.name}'s cache at {remote_hub} is not writable by its own user — it was "
                "created by a container running as root. Fix it there with: "
                f"sudo chown -R $USER {remote_hub}"
            ),
        }
    if free and size - already > free:
        return {"ok": False, "reason": f"{target.name} has {free / 2**30:.0f} GiB free, "
                                       f"needs about {(size - already) / 2**30:.0f} GiB"}
    return {
        "ok": True,
        "repo_id": repo_id,
        "directory": directory,
        "source_bytes": size,
        "already_there_bytes": already,
        "to_copy_bytes": max(size - already, 0),
        "target_free_bytes": free,
        "complete": already >= size,
    }


async def submit(repo_id: str, target_name: str, repo_type: str = "model") -> str:
    target = nodes.by_name(target_name)
    if target.is_local:
        raise ValueError("that is this machine; pick a peer")

    check = await preflight(repo_id, target, repo_type)
    if not check["ok"]:
        raise ValueError(check["reason"])

    directory = check["directory"]
    hub = f"{settings.hf_cache}/hub"
    source_alias = db.get_setting("sync_source_alias", "") or _default_source_alias()

    # Runs on the destination, pulling: this machine's cache is root-owned, so a
    # push would need privilege the peer's own user does not need.
    # trees/ is xet's dedup bookkeeping and .locks/ is download state; neither is
    # needed to load a model, and both are written 0600 by the root container
    # that downloaded it, so a non-root rsync cannot read them. Excluding them
    # is what turns the transfer from "exit 23, some files failed" into a clean
    # copy. The peer simply re-derives them if it ever downloads there itself.
    remote = (
        "rsync -a --info=progress2 --inplace --no-compress "
        "--exclude=trees/ --exclude=.locks/ "
        "-e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' "
        f"{shlex.quote(f'{source_alias}:{hub}/{directory}/')} "
        f"{shlex.quote(f'{hub}/{directory}/')}"
    )

    job_id = jobs.manager.create(
        jobs.JobSpec(
            kind=KIND,
            title=f"Sync {repo_id} to {target.name}",
            image="(ssh)",
            command=[remote],
            env={},
            mounts=[],
            gpu=False,
            meta={
                "repo_id": repo_id,
                "repo_type": repo_type,
                "target": target.name,
                "directory": directory,
                "bytes": check["to_copy_bytes"],
                "source": source_alias,
            },
        )
    )
    jobs.manager.adopt(job_id, _drive(job_id, target, remote))
    return job_id


def _default_source_alias() -> str:
    """How the peer refers back to this machine.

    A peer normally has an ssh alias for this box configured with the key that
    authenticates it; the raw address usually does not work, because the key is
    bound to the alias.
    """
    import socket

    return socket.gethostname()


async def _drive(job_id: str, target: nodes.Node, remote: str) -> None:
    jobs.manager.begin(job_id)
    jobs.manager.log(job_id, f"$ ssh {target.name} {remote}")

    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        target.name or target.address, remote,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    progress: dict[str, Any] = {"phase": "copying"}
    assert proc.stdout is not None
    try:
        buffer = ""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk.decode(errors="replace")
            # rsync rewrites one line with \r; split on both terminators.
            while True:
                index = min(
                    (i for i in (buffer.find("\n"), buffer.find("\r")) if i >= 0), default=-1
                )
                if index < 0:
                    break
                line, buffer = buffer[:index].strip(), buffer[index + 1:]
                if not line:
                    continue
                update = parse_sync(line, progress)
                if update:
                    progress.update(update)
                    jobs.manager.report(job_id, progress)
                else:
                    jobs.manager.log(job_id, line)
    except asyncio.CancelledError:
        proc.kill()
        raise

    code = await proc.wait()
    if code == 0:
        jobs.manager.finish(
            job_id, ok=True,
            progress={**progress, "phase": "done", "percent": 100.0},
            result={"target": target.name},
        )
    else:
        jobs.manager.finish(job_id, ok=False, error=f"rsync exited {code}", exit_code=code)
