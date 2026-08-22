"""The machines this dashboard can place work on.

A cluster here is a handful of identical boxes on one fast subnet, so a node is
just "somewhere the docker CLI can reach". `docker -H ssh://peer` runs the
client locally and talks to the remote daemon over the ssh key that is already
there, which means nothing has to listen on the network and there is no second
transport to maintain — every existing call in app.docker_ctl works against a
peer by passing its host string.

Membership is explicit. Peers are discovered by probing the cluster subnet and
then added deliberately, rather than adopted automatically: a box that answers
ssh is not necessarily one you want models scheduled onto.
"""

from __future__ import annotations

import asyncio
import ipaddress
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import db, docker_ctl
from app.config import settings

LOCAL = "local"
GIB = 1024 ** 3
SSH_TIMEOUT = 8
PROBE_CONCURRENCY = 16


@dataclass
class Node:
    name: str
    address: str = ""
    docker_host: str | None = None      # None means this machine
    enabled: bool = True
    note: str = ""

    @property
    def is_local(self) -> bool:
        return self.docker_host is None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "address": self.address,
            "docker_host": self.docker_host,
            "local": self.is_local,
            "enabled": self.enabled,
            "note": self.note,
        }


@dataclass
class NodeStatus:
    node: Node
    reachable: bool = False
    docker: str = ""
    gpu: str = ""
    has_nvidia_runtime: bool = False
    telemetry: dict = field(default_factory=dict)
    total_bytes: int = 0
    available_bytes: int = 0
    free_bytes: int = 0
    containers: list = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict:
        return {
            **self.node.as_dict(),
            "reachable": self.reachable,
            "docker": self.docker,
            "gpu": self.gpu,
            "has_nvidia_runtime": self.has_nvidia_runtime,
            "telemetry": self.telemetry,
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "free_bytes": self.free_bytes,
            "containers": self.containers,
            "error": self.error,
        }


# --- registry ------------------------------------------------------------

def local_node() -> Node:
    return Node(name=LOCAL, address="127.0.0.1", docker_host=None, note="this machine")


def registered() -> list[Node]:
    rows = db.get_setting("nodes", []) or []
    peers = [
        Node(
            name=row["name"],
            address=row.get("address", ""),
            docker_host=row.get("docker_host") or f"ssh://{row['name']}",
            enabled=bool(row.get("enabled", True)),
            note=row.get("note", ""),
        )
        for row in rows
    ]
    return [local_node(), *peers]


def by_name(name: str | None) -> Node:
    """Resolve a stored node name. An unknown or empty name means this machine,
    so a server saved before the cluster existed keeps working."""
    if not name or name == LOCAL:
        return local_node()
    for node in registered():
        if node.name == name:
            return node
    return local_node()


def add(name: str, *, address: str = "", docker_host: str = "", note: str = "") -> Node:
    rows = [r for r in (db.get_setting("nodes", []) or []) if r["name"] != name]
    rows.append({
        "name": name,
        "address": address,
        "docker_host": docker_host or f"ssh://{name}",
        "enabled": True,
        "note": note,
    })
    db.set_setting("nodes", rows)
    return by_name(name)


def remove(name: str) -> None:
    rows = [r for r in (db.get_setting("nodes", []) or []) if r["name"] != name]
    db.set_setting("nodes", rows)


# --- live status ---------------------------------------------------------

async def _ssh(address: str, command: str) -> tuple[int, str]:
    if not shutil.which("ssh"):
        return 127, "ssh is not installed"
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_TIMEOUT}",
        "-o", "StrictHostKeyChecking=accept-new", address, command,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=SSH_TIMEOUT + 10)
    except TimeoutError:
        proc.kill()
        return 124, "timed out"
    return proc.returncode or 0, (out or err).decode(errors="replace")


# One ssh round trip for everything the local telemetry loop reads, with a
# marker between sections so the same parsers can be used on the output. A peer
# is a machine, not a Ray node — its temperature and power draw are worth
# knowing whether or not anything is pooled onto it.
TELEMETRY_SCRIPT = (
    "echo '@@MEM@@'; cat /proc/meminfo; "
    "echo '@@LOAD@@'; cat /proc/loadavg; nproc; "
    "echo '@@GPU@@'; nvidia-smi {gpu} 2>/dev/null; "
    "echo '@@APPS@@'; nvidia-smi {apps} 2>/dev/null; "
    "echo '@@DISK@@'; df -PB1 {cache} 2>/dev/null | tail -1"
)


def _section(out: str, name: str) -> str:
    marker = f"@@{name}@@"
    if marker not in out:
        return ""
    rest = out.split(marker, 1)[1]
    for other in ("@@MEM@@", "@@LOAD@@", "@@GPU@@", "@@APPS@@", "@@DISK@@"):
        if other in rest:
            rest = rest.split(other, 1)[0]
    return rest.strip("\n")


async def remote_telemetry(node: Node) -> dict:
    """Everything the Overview shows for this machine, for a peer.

    Deliberately independent of Ray: knowing a box is there, how hot it is and
    what it is holding matters before anything is pooled onto it, and most of
    the time nothing is.
    """
    from app import telemetry

    command = TELEMETRY_SCRIPT.format(
        gpu=" ".join(telemetry.GPU_QUERY),
        apps=" ".join(telemetry.APPS_QUERY),
        cache=settings.hf_cache,
    )
    code, out = await _ssh(node.name or node.address, command)
    if code != 0:
        return {"ok": False, "error": out.strip()[:300]}

    load_raw = _section(out, "LOAD").split()
    try:
        load = [float(x) for x in load_raw[:3]]
        cpus = int(load_raw[-1])
    except (ValueError, IndexError):
        load, cpus = [], 0

    disk_raw = _section(out, "DISK").split()
    disk = {}
    if len(disk_raw) >= 4 and disk_raw[1].isdigit():
        disk = {"path": str(settings.hf_cache), "total_bytes": int(disk_raw[1]),
                "used_bytes": int(disk_raw[2]), "free_bytes": int(disk_raw[3])}

    return {
        "ok": True,
        "memory": telemetry.parse_meminfo(_section(out, "MEM")),
        "gpu": telemetry.parse_gpu_csv(_section(out, "GPU")),
        "gpu_processes": telemetry.parse_compute_apps(_section(out, "APPS")),
        "load": load,
        "cpu_count": cpus,
        "disk": disk,
    }


async def _remote_memory(node: Node) -> tuple[int, int, int]:
    code, out = await _ssh(node.name or node.address, "cat /proc/meminfo")
    if code != 0:
        return 0, 0, 0
    values = {}
    for line in out.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                values[key] = int(parts[0]) * 1024
            except ValueError:
                continue
    return (values.get("MemTotal", 0), values.get("MemAvailable", 0), values.get("MemFree", 0))


async def status(node: Node) -> NodeStatus:
    result = NodeStatus(node=node)

    if node.is_local:
        from app import telemetry as local_telemetry

        snapshot = await local_telemetry.snapshot()
        result.telemetry = snapshot
        memory = snapshot["memory"]
        result.total_bytes = memory["total_bytes"]
        result.available_bytes = memory["available_bytes"]
        result.free_bytes = memory["free_bytes"]
    else:
        remote = await remote_telemetry(node)
        if remote.get("ok"):
            result.telemetry = remote
            memory = remote["memory"]
            result.total_bytes = memory["total_bytes"]
            result.available_bytes = memory["available_bytes"]
            result.free_bytes = memory["free_bytes"]
        else:
            result.error = remote.get("error", "")

    try:
        result.docker = await docker_ctl.version(node.docker_host)
        result.reachable = bool(result.docker)
    except Exception as exc:
        result.error = str(exc)[-300:]
        return result

    runtimes = await docker_ctl._run(
        docker_ctl.docker_argv(node.docker_host, "info", "--format", "{{json .Runtimes}}"),
        check=False,
    )
    result.has_nvidia_runtime = '"nvidia"' in runtimes[1]

    containers = await docker_ctl.ps(all_containers=False, host=node.docker_host)
    result.containers = [
        {"name": c.get("Names"), "image": c.get("Image"), "status": c.get("Status")}
        for c in containers
    ]
    return result


async def status_all() -> list[dict]:
    nodes = registered()
    results = await asyncio.gather(*(status(n) for n in nodes), return_exceptions=True)
    out = []
    for node, result in zip(nodes, results, strict=True):
        if isinstance(result, BaseException):
            out.append(NodeStatus(node=node, error=str(result)[-200:]).as_dict())
        else:
            out.append(result.as_dict())
    return out


# --- discovery -----------------------------------------------------------

def _candidate_addresses() -> list[str]:
    """The cluster subnet, derived from the RoCE interface rather than guessed."""
    import subprocess

    try:
        raw = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", settings.roce_interface],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    for token in raw.split():
        if "/" in token:
            try:
                network = ipaddress.ip_interface(token).network
            except ValueError:
                continue
            if network.num_addresses > 1024:
                return []
            mine = ipaddress.ip_interface(token).ip
            return [str(host) for host in network.hosts() if host != mine]
    return []


def _ssh_aliases() -> dict[str, str]:
    """Host aliases from ssh config, resolved to the address each points at.

    Probing bare IPs is not enough: a cluster is usually wired up with per-host
    keys under an alias, and ssh to the raw address then fails on authentication
    even though the box is perfectly reachable as `node2`. `ssh -G` resolves the
    effective config, Include directives and all, without parsing it ourselves.
    """
    import re
    import subprocess

    config = Path.home() / ".ssh" / "config"
    if not config.is_file():
        return {}

    names: list[str] = []
    seen_files = {config}
    pending = [config]
    while pending:
        current = pending.pop()
        try:
            text = current.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("include "):
                for pattern in line.split(None, 1)[1].split():
                    for extra in Path("/").glob(pattern.lstrip("/")):
                        if extra not in seen_files:
                            seen_files.add(extra)
                            pending.append(extra)
            elif line.lower().startswith("host "):
                names += [n for n in line.split()[1:] if not re.search(r"[*?!]", n)]

    resolved: dict[str, str] = {}
    for name in dict.fromkeys(names):
        try:
            out = subprocess.run(["ssh", "-G", name], capture_output=True,
                                 text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            if line.startswith("hostname "):
                resolved[name] = line.split(None, 1)[1].strip()
                break
    return resolved


async def _probe(address: str) -> dict | None:
    code, out = await _ssh(address, "hostname; command -v docker >/dev/null && echo has-docker")
    if code != 0:
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return None
    return {
        "address": address,
        "hostname": lines[0],
        "docker": "has-docker" in lines,
    }


async def discover(scan_subnet: bool = False) -> dict:
    """Find boxes this machine can already ssh into.

    ssh aliases come first and are the fast path: a cluster is normally wired up
    with a per-host key under a name, and that name — not the address behind it
    — is what authenticates. Sweeping the subnet by IP is the fallback for
    clusters wired without aliases, and it is opt-in because a /24 of ssh
    attempts takes minutes where the alias probe takes a second.

    Nothing is added by this call. A box answering ssh is not the same as one
    you want models placed on, so adoption stays a deliberate act.
    """
    aliases = await asyncio.to_thread(_ssh_aliases)
    subnet = set(_candidate_addresses())
    targets = list(aliases)
    if scan_subnet:
        targets += sorted(subnet - set(aliases.values()))

    if not targets:
        return {"candidates": [], "scanned": 0,
                "note": "no ssh aliases configured; try the subnet scan"}

    known = {node.address for node in registered()} | {node.name for node in registered()}
    semaphore = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def guarded(target: str):
        async with semaphore:
            hit = await _probe(target)
            if hit:
                hit["target"] = target
            return hit

    found = await asyncio.gather(*(guarded(t) for t in targets))

    candidates: list[dict] = []
    seen_hosts = {n.name for n in registered() if n.is_local}
    # This machine answers its own ssh alias; it is already the local node.
    import socket

    seen_hosts.add(socket.gethostname())

    for hit in found:
        if not hit or hit["hostname"] in seen_hosts:
            continue
        seen_hosts.add(hit["hostname"])
        candidates.append({
            **hit,
            "via_alias": hit["target"] in aliases,
            "on_cluster_subnet": aliases.get(hit["target"], hit["target"]) in subnet,
            "registered": hit["hostname"] in known or hit["target"] in known,
        })
    return {"candidates": candidates, "scanned": len(targets),
            "interface": settings.roce_interface, "scanned_subnet": scan_subnet}


def combine(statuses: list[dict]) -> dict[str, Any]:
    """The cluster as one pool.

    A pooled engine's memory is the sum of what each node can commit, so the
    number that matters for "will this model fit" is the combined ceiling, not
    any one machine's. Unreachable nodes contribute nothing rather than being
    counted optimistically.
    """
    from app.config import settings

    live = [s for s in statuses if s.get("reachable") and s.get("total_bytes")]
    reserve = int(settings.mem_reserve_gib * GIB)
    total = sum(s["total_bytes"] for s in live)
    available = sum(s["available_bytes"] for s in live)
    ceiling = sum(max(0, s["total_bytes"] - reserve) for s in live)
    biggest = max((max(0, s["total_bytes"] - reserve) for s in live), default=0)
    return {
        "nodes": len(live),
        "unreachable": len(statuses) - len(live),
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": max(0, total - available),
        # What one pooled engine could claim across every reachable node.
        "pooled_ceiling_bytes": ceiling,
        # What the largest single machine could claim, for the comparison that
        # tells you whether pooling is worth it for a given model.
        "single_node_ceiling_bytes": biggest,
        "reserve_bytes_per_node": reserve,
    }


async def summary() -> dict[str, Any]:
    statuses = await status_all()
    return {"nodes": statuses, "local": LOCAL, "combined": combine(statuses)}


# --- getting our own code onto a peer -------------------------------------

WORKER_DIR_REMOTE = ".llmd/workers"


async def ensure_workers(node: Node) -> str:
    """Put the worker scripts on a peer and return the path to bind-mount.

    A bind mount is resolved by the daemon that runs the container, not by the
    client asking for it. Mounting this repo's app/workers into a container on
    node2 therefore mounts *node2's* copy of that path — which does not exist,
    and the container dies with "can't open file '/worker/...'". The scripts are
    small and change with the dashboard, so they are pushed every time rather
    than cached and left to go stale.
    """
    if node.is_local:
        from app.config import settings

        return str(settings.hf_cache.parent) and str(
            __import__("pathlib").Path(__file__).resolve().parent / "workers"
        )

    local_dir = Path(__file__).resolve().parent / "workers"
    target = node.name or node.address
    remote = f"{WORKER_DIR_REMOTE}"

    code, out = await _ssh(target, f"mkdir -p ~/{remote} && echo $HOME")
    if code != 0:
        raise RuntimeError(f"could not prepare {target}: {out.strip()[:200]}")
    home = out.strip().splitlines()[-1]

    proc = await asyncio.create_subprocess_exec(
        "rsync", "-a", "--delete",
        "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
        f"{local_dir}/", f"{target}:{home}/{remote}/",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not copy the workers to {target}: {stdout.decode(errors='replace')[-300:]}"
        )
    return f"{home}/{remote}"
