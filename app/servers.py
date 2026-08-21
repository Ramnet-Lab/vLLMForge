"""vLLM server lifecycle.

A "server" is a saved parameter set plus a container. The dashboard also
discovers vLLM containers it did not start — the hand-launched ones from
the hand-written scripts this replaces show up alongside managed servers so the memory picture on the
Overview page is the truth about the host, not just about this app.

Managed containers deliberately get restart policy "no": on a unified-memory
host, a crash-looping engine that keeps re-reserving 60 GiB is worse than a
stopped one, and the memory watchdog needs to be able to kill something and
have it stay dead.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx

from app import db, docker_ctl, events, safety, vllm_spec
from app.config import settings

JSON_FIELDS = ("args", "env")
HEALTH_TIMEOUT = 2.0
DEFAULT_PORT_RANGE = range(8010, 8100)

# vLLM's own metrics, cherry-picked for the UI. Everything else stays in the
# raw /metrics passthrough.
INTERESTING_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:num_preemptions_total",
)

_METRIC_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)$")


# --- CRUD ---------------------------------------------------------------

def list_servers() -> list[dict]:
    return [db.hydrate(row, JSON_FIELDS) for row in db.query("SELECT * FROM servers ORDER BY id")]


def get_server(server_id: int) -> dict | None:
    return db.hydrate(db.query_one("SELECT * FROM servers WHERE id = ?", (server_id,)), JSON_FIELDS)


def get_by_name(name: str) -> dict | None:
    return db.hydrate(db.query_one("SELECT * FROM servers WHERE name = ?", (name,)), JSON_FIELDS)


def create_server(payload: dict) -> dict:
    now = db.now()
    cursor = db.execute(
        "INSERT INTO servers (name, model, served_name, port, image, args, env, notes, autostart,"
        " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            payload["name"],
            payload["model"],
            payload.get("served_name") or "",
            int(payload["port"]),
            payload.get("image") or settings.vllm_image,
            db.dumps(payload.get("args") or {}),
            db.dumps(payload.get("env") or {}),
            payload.get("notes", ""),
            1 if payload.get("autostart") else 0,
            now,
            now,
        ),
    )
    return get_server(int(cursor.lastrowid))


def update_server(server_id: int, payload: dict) -> dict | None:
    existing = get_server(server_id)
    if existing is None:
        return None
    merged = {**existing, **{k: v for k, v in payload.items() if v is not None}}
    db.execute(
        "UPDATE servers SET name = ?, model = ?, served_name = ?, port = ?, image = ?, args = ?,"
        " env = ?, notes = ?, autostart = ?, updated_at = ? WHERE id = ?",
        (
            merged["name"],
            merged["model"],
            merged.get("served_name") or "",
            int(merged["port"]),
            merged.get("image") or settings.vllm_image,
            db.dumps(merged.get("args") or {}),
            db.dumps(merged.get("env") or {}),
            merged.get("notes", ""),
            1 if merged.get("autostart") else 0,
            db.now(),
            server_id,
        ),
    )
    return get_server(server_id)


def delete_server(server_id: int) -> None:
    db.execute("DELETE FROM servers WHERE id = ?", (server_id,))


def container_name(server: dict) -> str:
    return settings.container_name("vllm", server["id"])


def used_ports() -> set[int]:
    return {int(row["port"]) for row in db.query("SELECT port FROM servers")}


def suggest_port() -> int:
    taken = used_ports() | {8000, 8001, 8080, 8265, settings.port}
    for candidate in DEFAULT_PORT_RANGE:
        if candidate not in taken:
            return candidate
    return max(taken) + 1


# --- command assembly ---------------------------------------------------

def build_command(server: dict) -> list[str]:
    args = dict(server.get("args") or {})
    if server.get("served_name") and not args.get("served_model_name"):
        args["served_model_name"] = server["served_name"]
    return vllm_spec.build_argv(server["model"], args, port=int(server["port"]))


def build_env(server: dict) -> dict[str, str]:
    env = {"HF_HOME": "/hf", **settings.nccl_env()}
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token
    env.update({str(k): str(v) for k, v in (server.get("env") or {}).items()})
    if (server.get("args") or {}).get("enable_lora"):
        env.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "1")
    return env


def launch_preview(server: dict) -> str:
    return docker_ctl.preview(
        docker_ctl.build_run_argv(
            name=container_name(server),
            image=server.get("image") or settings.vllm_image,
            command=build_command(server),
            env=build_env(server),
            mounts=_mounts(),
        )
    )


def _mounts() -> list[docker_ctl.Mount]:
    return [
        docker_ctl.Mount(settings.hf_cache, "/hf"),
        docker_ctl.Mount(settings.output_dir, "/outputs"),
    ]


# --- lifecycle ----------------------------------------------------------

async def start(server_id: int, *, force: bool = False) -> dict:
    server = get_server(server_id)
    if server is None:
        raise KeyError(server_id)

    name = container_name(server)
    util = vllm_spec.gpu_memory_utilization(server.get("args") or {})
    verdict = await safety.check_launch(util, replacing=name)
    if not verdict.ok and not force:
        return {"started": False, "safety": verdict.as_dict()}

    await docker_ctl.remove(name, force=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    await docker_ctl.run_detached(
        name=name,
        image=server.get("image") or settings.vllm_image,
        command=build_command(server),
        env=build_env(server),
        mounts=_mounts(),
        gpu=True,
        network="host",
    )
    db.execute("UPDATE servers SET last_started_at = ? WHERE id = ?", (db.now(), server_id))
    await events.broker.publish(events.SERVERS, {"type": "started", "id": server_id})
    return {"started": True, "safety": verdict.as_dict(), "container": name}


async def stop(server_id: int) -> None:
    server = get_server(server_id)
    if server is None:
        raise KeyError(server_id)
    await docker_ctl.stop(container_name(server))
    await events.broker.publish(events.SERVERS, {"type": "stopped", "id": server_id})


async def remove_container(server_id: int) -> None:
    server = get_server(server_id)
    if server is not None:
        await docker_ctl.remove(container_name(server), force=True)


async def logs(server_id: int, tail: int = 400) -> str:
    server = get_server(server_id)
    if server is None:
        raise KeyError(server_id)
    return await docker_ctl.logs(container_name(server), tail=tail)


# --- health & metrics ---------------------------------------------------

async def probe(port: int, *, host: str = "127.0.0.1") -> dict:
    """Ask a vLLM frontend whether it is actually serving yet."""
    base = f"http://{host}:{port}"
    result: dict[str, Any] = {"reachable": False, "healthy": False, "models": []}
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            health = await client.get(f"{base}/health")
            result["reachable"] = True
            result["healthy"] = health.status_code == 200
            if result["healthy"]:
                models = await client.get(f"{base}/v1/models")
                if models.status_code == 200:
                    result["models"] = [m["id"] for m in models.json().get("data", [])]
    except (httpx.HTTPError, OSError):
        pass
    return result


def parse_metrics(text: str) -> dict[str, Any]:
    """Minimal Prometheus text parse: last value wins per metric name."""
    gauges: dict[str, float] = {}
    counters: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name, raw = match.group("name"), match.group("value")
        try:
            value = float(raw)
        except ValueError:
            continue
        target = counters if name.endswith("_total") else gauges
        target[name] = target.get(name, 0.0) + value if name.endswith("_total") else value
    merged = {**gauges, **counters}
    return {
        "selected": {k: merged[k] for k in INTERESTING_METRICS if k in merged},
        "all": merged,
    }


async def metrics(port: int, *, host: str = "127.0.0.1") -> dict:
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            response = await client.get(f"http://{host}:{port}/metrics")
        if response.status_code != 200:
            return {"selected": {}, "all": {}}
        return parse_metrics(response.text)
    except (httpx.HTTPError, OSError):
        return {"selected": {}, "all": {}}


# --- discovery of containers we did not start ---------------------------

_PORT_FLAG = re.compile(r"--port(?:=(\d+))?")


def _port_from_command(command: list[str] | None) -> int | None:
    if not command:
        return None
    for index, token in enumerate(command):
        match = _PORT_FLAG.fullmatch(token)
        if match:
            if match.group(1):
                return int(match.group(1))
            if index + 1 < len(command):
                try:
                    return int(command[index + 1])
                except ValueError:
                    return None
    return None


def _model_from_command(command: list[str] | None) -> str:
    if not command:
        return ""
    for index, token in enumerate(command):
        if token == "serve" and index + 1 < len(command):
            candidate = command[index + 1]
            return "" if candidate.startswith("-") else candidate
    return ""


async def discover_foreign() -> list[dict]:
    """vLLM containers on this host that the dashboard does not manage."""
    managed = {container_name(s) for s in list_servers()}
    found: list[dict] = []
    for row in await docker_ctl.ps(all_containers=False):
        name = str(row.get("Names", ""))
        if not name or name in managed:
            continue
        state = await docker_ctl.state(name)
        if not safety.is_vllm_command(state.command):
            continue
        port = _port_from_command(state.command)
        found.append(
            {
                "name": name,
                "image": state.image,
                "model": _model_from_command(state.command),
                "port": port,
                "util": safety.parse_util(state.command),
                "status": state.ui_status,
                "started_at": state.started_at,
                "command": state.command,
            }
        )
    return found


# --- aggregate view -----------------------------------------------------

async def status_all() -> dict:
    servers = list_servers()
    names = [container_name(s) for s in servers]
    states, foreign = await asyncio.gather(docker_ctl.states(names), discover_foreign())

    async def decorate(server: dict) -> dict:
        state = states.get(container_name(server))
        entry = {
            **server,
            "container": container_name(server),
            "status": state.ui_status if state else "absent",
            "exit_code": state.exit_code if state else None,
            "oom_killed": state.oom_killed if state else False,
            "started_at": state.started_at if state else None,
            "util": vllm_spec.gpu_memory_utilization(server.get("args") or {}),
            "url": f"http://127.0.0.1:{server['port']}",
        }
        if entry["status"] in ("running", "starting", "unhealthy"):
            entry["health"] = await probe(int(server["port"]))
            # vLLM does not bind its port until the weights are loaded and CUDA
            # graphs are captured, which on a 27B model is minutes. A refused
            # connection therefore means "still loading", not "broken".
            if entry["status"] == "running" and not entry["health"]["reachable"]:
                entry["status"] = "loading"
        else:
            entry["health"] = {"reachable": False, "healthy": False, "models": []}
        return entry

    decorated = await asyncio.gather(*(decorate(s) for s in servers)) if servers else []

    for item in foreign:
        if item.get("port"):
            item["health"] = await probe(int(item["port"]))

    return {
        "servers": list(decorated),
        "foreign": foreign,
        "budget": (await safety.current_budget()).as_dict(),
    }


async def endpoints() -> list[dict]:
    """Everything the playground can talk to, managed or not."""
    status = await status_all()
    out: list[dict] = []
    for server in status["servers"]:
        if server["health"]["healthy"]:
            out.append(
                {
                    "id": f"server:{server['id']}",
                    "label": server["name"],
                    "url": server["url"],
                    "models": server["health"]["models"],
                    "managed": True,
                }
            )
    for item in status["foreign"]:
        health = item.get("health") or {}
        if health.get("healthy"):
            out.append(
                {
                    "id": f"container:{item['name']}",
                    "label": item["name"],
                    "url": f"http://127.0.0.1:{item['port']}",
                    "models": health.get("models", []),
                    "managed": False,
                }
            )
    return out


async def autostart() -> None:
    """Bring up servers flagged autostart, newest definition last."""
    for server in list_servers():
        if not server.get("autostart"):
            continue
        state = await docker_ctl.state(container_name(server))
        if state.running:
            continue
        try:
            await start(int(server["id"]))
        except Exception:  # one bad definition must not block startup
            import logging

            logging.getLogger("llmd.servers").exception("autostart failed for %s", server["name"])
