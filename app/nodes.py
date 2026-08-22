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
        from app.telemetry import read_meminfo

        memory = read_meminfo()
        result.total_bytes = memory.total_bytes
        result.available_bytes = memory.available_bytes
        result.free_bytes = memory.free_bytes
    else:
        result.total_bytes, result.available_bytes, result.free_bytes = await _remote_memory(node)

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


async def summary() -> dict[str, Any]:
    return {"nodes": await status_all(), "local": LOCAL}
