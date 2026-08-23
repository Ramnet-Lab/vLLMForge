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
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from app import cluster, db, docker_ctl, events, hf, jobs, nodes, safety, vllm_spec
from app.config import settings

JSON_FIELDS = ("args", "env", "pool_nodes")

# Measuring the budget and creating the container have to happen together.
# Two starts that each measure before the other has launched will both be
# admitted, and the pool is then over-committed by design rather than accident.
LAUNCH_LOCK = asyncio.Lock()
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
        " node, pool_nodes, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            payload.get("node") or nodes.LOCAL,
            db.dumps(payload.get("pool_nodes") or []),
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
        " env = ?, notes = ?, autostart = ?, node = ?, pool_nodes = ?, updated_at = ? WHERE id = ?",
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
            merged.get("node") or nodes.LOCAL,
            db.dumps(merged.get("pool_nodes") or []),
            db.now(),
            server_id,
        ),
    )
    return get_server(server_id)


def delete_server(server_id: int) -> None:
    db.execute("DELETE FROM servers WHERE id = ?", (server_id,))


def container_name(server: dict) -> str:
    return settings.container_name("vllm", server["id"])


def pool_of(server: dict) -> list[str]:
    """The nodes a pooled server spans. Empty means an ordinary single-box server."""
    pool = server.get("pool_nodes") or []
    return [str(n) for n in pool] if isinstance(pool, list) else []


def is_pooled(server: dict) -> bool:
    return len(pool_of(server)) > 1


def node_of(server: dict) -> nodes.Node:
    return nodes.by_name(server.get("node"))


def host_of(server: dict) -> str | None:
    """Where this server's vLLM container runs. For a pooled engine that is the
    Ray head — the other nodes hold shards, not the HTTP frontend."""
    pool = pool_of(server)
    if pool:
        return nodes.by_name(pool[0]).docker_host
    return node_of(server).docker_host


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
    hf_token = hf.token()
    if hf_token:
        env["HF_TOKEN"] = hf_token
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


OUTPUTS_MOUNT = "/outputs"
CACHE_MOUNT = "/hf"


def _mounts() -> list[docker_ctl.Mount]:
    return [
        docker_ctl.Mount(settings.hf_cache, CACHE_MOUNT),
        docker_ctl.Mount(settings.output_dir, OUTPUTS_MOUNT),
    ]


def container_path(host_path: str | Path) -> str:
    """Where a vLLM container sees a path that a job produced on the host.

    Models built by the Heretic and fine-tuning tabs live under the output
    directory, which is bind-mounted into every server container. Handing vLLM
    the host path instead would fail at load with a missing-directory error, and
    only after the container had already started.
    """
    resolved = Path(host_path).resolve()
    root = settings.output_dir.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return str(resolved)
    return f"{OUTPUTS_MOUNT}/{relative}" if str(relative) != "." else OUTPUTS_MOUNT


# --- lifecycle ----------------------------------------------------------

def _pool_safety(plan: dict) -> dict:
    """A pooled plan in the shape the single-node verdict has.

    Callers — the router, and through it the form — read `safety` off every
    start result. A pooled failure used to omit it, and the router's
    `**result["safety"]` turned that into a KeyError, which its own
    `except KeyError` then reported as "no such server". A launch refused for
    running out of memory therefore came back as a 404.
    """
    refused = [node for node in plan.get("nodes") or []
               if (node.get("verdict") or {}).get("ok") is False]
    worst = next((node["verdict"] for node in refused), None)
    return {
        "ok": bool(plan.get("ok")),
        "level": "block" if refused else ("ok" if plan.get("ok") else "warn"),
        "message": plan.get("reason") or "",
        "requested_util": (worst or {}).get("requested_util", 0.0),
        "requested_bytes": (worst or {}).get("requested_bytes", 0),
        "suggested_util": plan.get("free_util"),
        "budget": (worst or {}).get("budget", {}),
        "nodes": [
            {"name": node["name"], "verdict": node.get("verdict")}
            for node in plan.get("nodes") or []
        ],
    }


async def start_pooled(server: dict, *, force: bool = False) -> dict:
    """One engine across several machines, as K containers that meet over the fabric.

    Rank 0 serves HTTP and every other rank runs `--headless`; they rendezvous
    at rank 0's fabric address on a port belonging to this engine alone. That
    per-engine port is what allows more than one pooled engine to exist — the
    Ray shape this replaced had a fixed head port and a fixed worker container
    name, so a second pooled launch killed the first and then failed itself.

    The whole thing runs under LAUNCH_LOCK, plan included. The plan reads every
    node's free memory and picks a rendezvous port, and both answers are only
    true until somebody else launches; deciding outside the lock is how two
    launches seconds apart both pass a budget only one of them fits in.
    """
    pool = pool_of(server)
    base = container_name(server)

    async with LAUNCH_LOCK:
        # The arguments go with it: a pooled launch used to skip the memory guard
        # entirely, which is how a definition asking for 0.95 of a box with 0.88
        # free reached vLLM and died four minutes in.
        plan = await cluster.plan(pool, server.get("model") or "", server.get("args") or {},
                                  replacing=base)
        if not plan["ok"] and not force:
            safety_block = _pool_safety(plan)
            # A plan refused because the memory would not fit is the same kind of
            # answer the single-node path gives — the request was well-formed and
            # the machines cannot take it. Anything else (a missing image, no
            # cluster interface) is an environment fault and reads as an error.
            refused_for_memory = safety_block["level"] == "block"
            return {
                "started": False,
                "error": "" if refused_for_memory
                         else plan.get("reason", "cannot pool across these nodes"),
                "safety": safety_block,
                "plan": plan,
            }

        prefix = cluster._subnet_prefix()
        wirings = [await cluster.wiring(nodes.by_name(name), prefix) for name in pool]
        head = wirings[0]
        names = cluster.rank_names(base, len(wirings))

        # This engine's own ranks must not make it look like the port is taken.
        master_port = await cluster.allocate_master_port(
            [w.node for w in wirings], exclude=set(names))

        args = dict(server.get("args") or {})
        args.setdefault("tensor_parallel_size", 1)
        # Distributed wiring is appended by rank_argv, not routed through the
        # schema: build_argv drops any value equal to the flag's default, which
        # would silently swallow --node-rank 0.
        args.pop("pipeline_parallel_size", None)
        args.pop("distributed_executor_backend", None)
        if server.get("served_name") and not args.get("served_model_name"):
            args["served_model_name"] = server["served_name"]

        serve_argv = vllm_spec.build_argv(server["model"], args, port=int(server["port"]))
        settings.output_dir.mkdir(parents=True, exist_ok=True)

        # Every rank name, on every node, before anything is started: a rank left
        # over from a previous launch takes its name and fails the new one with
        # "container name already in use".
        await cluster.stop_ranks(base, wirings, remove=True)
        try:
            for rank, w in enumerate(wirings):
                await docker_ctl.run_detached(
                    name=names[rank],
                    image=server.get("image") or settings.vllm_image,
                    command=cluster.rank_argv(serve_argv, rank, len(wirings), head, master_port),
                    env=cluster.rank_env(w, head, build_env(server)),
                    mounts=_mounts(),
                    gpu=True,
                    network="host",
                    host=w.node.docker_host,
                )
        except Exception as exc:
            # A half-started engine is worse than none: the ranks that did come
            # up hold their full share of their machines for a world size that
            # will never be reached.
            await cluster.stop_ranks(base, wirings, remove=True)
            return {"started": False, "error": str(exc)[-800:],
                    "safety": _pool_safety(plan), "plan": plan}

    db.execute("UPDATE servers SET last_started_at = ? WHERE id = ?", (db.now(), server["id"]))
    await events.broker.publish(events.SERVERS, {"type": "started", "id": server["id"]})
    return {
        "started": True,
        "pooled": True,
        "safety": _pool_safety(plan),
        "container": base,
        "containers": names,
        "head": head.node.name,
        "master_port": master_port,
        "pipeline_parallel_size": len(wirings),
        "pooled_bytes": plan["pooled_bytes"],
        "plan": plan,
    }


async def start(server_id: int, *, force: bool = False) -> dict:
    server = get_server(server_id)
    if server is None:
        raise KeyError(server_id)

    if is_pooled(server):
        return await start_pooled(server, force=force)

    name = container_name(server)
    args = server.get("args") or {}
    util = vllm_spec.gpu_memory_utilization(args)

    node = node_of(server)
    async with LAUNCH_LOCK:
        verdict = await safety.check_launch(util, replacing=name, params=args, node=node)
        if not verdict.ok and not force:
            return {"started": False, "safety": verdict.as_dict()}

        await docker_ctl.remove(name, force=True, host=node.docker_host)
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            await docker_ctl.run_detached(
                name=name,
                image=server.get("image") or settings.vllm_image,
                command=build_command(server),
                env=build_env(server),
                mounts=_mounts(),
                gpu=True,
                network="host",
                host=node.docker_host,
            )
        except docker_ctl.DockerError as exc:
            # docker's own refusal is the useful message here — a port already
            # bound, an image that is not present, a bad flag. Losing it to a
            # bare 500 leaves the UI with nothing to show.
            return {"started": False, "error": exc.stderr or str(exc),
                    "safety": verdict.as_dict()}
    db.execute("UPDATE servers SET last_started_at = ? WHERE id = ?", (db.now(), server_id))
    await events.broker.publish(events.SERVERS, {"type": "started", "id": server_id})
    return {"started": True, "safety": verdict.as_dict(), "container": name, "node": node.name}


async def _pool_wirings(server: dict) -> list:
    prefix = cluster._subnet_prefix()
    return [await cluster.wiring(nodes.by_name(n), prefix) for n in pool_of(server)]


async def stop(server_id: int) -> None:
    server = get_server(server_id)
    if server is None:
        raise KeyError(server_id)
    if is_pooled(server):
        # Every rank, on every node. A rank left running holds its full share of
        # a machine for an engine that no longer exists, and once the row is
        # gone nothing else knows its name.
        await cluster.stop_ranks(container_name(server), await _pool_wirings(server))
    else:
        await docker_ctl.stop(container_name(server), host=host_of(server))
    await events.broker.publish(events.SERVERS, {"type": "stopped", "id": server_id})


async def remove_container(server_id: int) -> None:
    server = get_server(server_id)
    if server is None:
        return
    if is_pooled(server):
        await cluster.stop_ranks(container_name(server), await _pool_wirings(server),
                                 remove=True)
    else:
        await docker_ctl.remove(container_name(server), force=True, host=host_of(server))


async def logs(server_id: int, tail: int = 400) -> str:
    server = get_server(server_id)
    if server is None:
        raise KeyError(server_id)
    return await docker_ctl.logs(container_name(server), tail=tail, host=host_of(server))


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


def engine_of(container: str) -> dict | None:
    """The server definition a container belongs to, rank containers included."""
    for definition in list_servers():
        base = container_name(definition)
        if container in cluster.rank_names(base, len(pool_of(definition)) or 1):
            return definition
    return None


async def engine_containers(container: str) -> list[tuple[str, str | None]]:
    """Everything that has to die with this container, as (name, docker host).

    For an ordinary server that is just itself. For one rank of a pooled engine
    it is every rank, because killing one does not free what a watchdog thinks
    it frees: the world size is fixed, so the engine aborts anyway, and every
    other rank stays resident holding its full share of its own machine. The
    memory comes back only if the engine goes as a unit.
    """
    owner = engine_of(container)
    if owner is None or not is_pooled(owner):
        return [(container, None)]
    base = container_name(owner)
    wirings = await _pool_wirings(owner)
    return [(cluster.rank_container(base, rank), w.node.docker_host)
            for rank, w in enumerate(wirings)]


async def discover_foreign() -> list[dict]:
    """vLLM containers on this host that the dashboard does not manage.

    A pooled server owns a container per rank, and its far ranks can land on
    this machine. Counting only rank 0 as managed listed them here as somebody
    else's hand-launched engines, complete with a Stop button — one click on
    which aborts a healthy multi-machine engine and leaves its other ranks
    resident, holding their memory for a world size that can never re-form.
    """
    managed = set()
    for definition in list_servers():
        base = container_name(definition)
        managed.update(cluster.rank_names(base, len(pool_of(definition)) or 1))
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
    registry = nodes.registered()
    by_host: dict[str | None, list[dict]] = {}
    for server in list_servers():
        by_host.setdefault(node_of(server).docker_host, []).append(server)

    states: dict[str, docker_ctl.ContainerState] = {}
    for host, group in by_host.items():
        states |= await docker_ctl.states([container_name(s) for s in group], host=host)

    async def decorate(server: dict) -> dict:
        node = node_of(server)
        state = states.get(container_name(server))
        # A server on a peer answers on that peer's address, not on loopback.
        base_host = "127.0.0.1" if node.is_local else (node.address or node.name)
        entry = {
            **server,
            "container": container_name(server),
            "node": node.name,
            "node_local": node.is_local,
            "status": state.ui_status if state else "absent",
            "exit_code": state.exit_code if state else None,
            "oom_killed": state.oom_killed if state else False,
            "started_at": state.started_at if state else None,
            "util": vllm_spec.gpu_memory_utilization(server.get("args") or {}),
            "url": f"http://{base_host}:{server['port']}",
        }
        if entry["status"] in ("running", "starting", "unhealthy"):
            entry["health"] = await probe(int(server["port"]), host=base_host)
            # vLLM does not bind its port until the weights are loaded and the
            # CUDA graphs captured, so a refused connection means "still
            # loading", not "broken".
            if entry["status"] == "running" and not entry["health"]["reachable"]:
                entry["status"] = "loading"
        else:
            entry["health"] = {"reachable": False, "healthy": False, "models": []}
        return entry

    servers_out = await asyncio.gather(*(decorate(s) for s in list_servers())) if by_host else []

    foreign = await discover_foreign()
    for item in foreign:
        if item.get("port"):
            item["health"] = await probe(int(item["port"]))

    budgets = {}
    for node in registry:
        budgets[node.name] = (await safety.current_budget(node=node)).as_dict()

    return {
        "servers": list(servers_out),
        "foreign": foreign,
        "nodes": [n.as_dict() for n in registry],
        "budgets": budgets,
        # The local budget stays at the top level so anything reading the old
        # shape keeps working.
        "budget": budgets.get(nodes.LOCAL, {}),
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
                    "label": server["name"] if server["node_local"]
                             else f"{server['name']} on {server['node']}",
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


async def reap_partial_pools_once() -> list[str]:
    """Stop the surviving ranks of any engine that has already lost one.

    A pooled engine has a fixed world size: when one rank dies the others do not
    carry on shorthanded, they abort or hang. What they do keep is their memory
    — a full utilisation share of a machine each, held for an engine that can
    never re-form and that nothing else will name once the operator has moved
    on. On a box where GPU memory is host memory, that is the difference between
    a failed launch and a machine with nothing left to give.

    Only partial pools are touched. An engine with every rank down was stopped
    on purpose, and an engine with every rank up is working.
    """
    reaped: list[str] = []
    for definition in list_servers():
        if not is_pooled(definition):
            continue
        base = container_name(definition)
        wirings = await _pool_wirings(definition)
        if len(wirings) != len(pool_of(definition)):
            continue
        states = [
            await docker_ctl.state(cluster.rank_container(base, rank), w.node.docker_host)
            for rank, w in enumerate(wirings)
        ]
        running = [st.running for st in states]
        if not any(running) or all(running):
            continue
        lost = [w.node.name for w, alive in zip(wirings, running, strict=True) if not alive]
        logging.getLogger("llmd.servers").warning(
            "%s lost its rank on %s; stopping the %d rank(s) still holding memory",
            base, ", ".join(lost), sum(running))
        await cluster.stop_ranks(base, wirings)
        await events.broker.publish(events.SERVERS, {"type": "stopped", "id": definition["id"]})
        reaped.append(base)
    return reaped


async def reap_partial_pools(interval: float = 30.0) -> None:
    while True:
        try:
            await reap_partial_pools_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger("llmd.servers").exception("pool reaper failed")
        await asyncio.sleep(interval)


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


# --- launching, including whatever has to happen first --------------------

LAUNCH_KIND = "launch"


@jobs.register_parser(LAUNCH_KIND)
def _parse_launch(line: str, progress: dict) -> dict | None:
    if line.startswith("[launch] "):
        return {"phase": line[len("[launch] "):][:120]}
    return None


async def submit_launch(server_id: int, *, force: bool = False) -> str:
    """Start a server, doing the work it needs first.

    A model missing from a node used to be a refusal. It is really just a step:
    every node loads its own shard from its own disk, so the model has to be
    there — and the dashboard already knows how to put it there over the
    cluster link. Making that part of the launch means the user asks for a
    server and gets a server.
    """
    server = get_server(server_id)
    if server is None:
        raise KeyError(server_id)

    job_id = jobs.manager.create(
        jobs.JobSpec(
            kind=LAUNCH_KIND,
            title=f"Launch {server['name']}",
            image="(orchestration)",
            command=[],
            env={},
            mounts=[],
            gpu=False,
            meta={"server_id": server_id, "pool": pool_of(server), "model": server["model"]},
        )
    )
    jobs.manager.adopt(job_id, _drive_launch(job_id, server_id, force))
    return job_id


async def _drive_launch(job_id: str, server_id: int, force: bool) -> None:
    from app import sync

    jobs.manager.begin(job_id)
    server = get_server(server_id)
    if server is None:
        jobs.manager.finish(job_id, ok=False, error="the server was deleted")
        return

    pool = pool_of(server)
    try:
        if pool:
            missing = await cluster.missing_model(server["model"], pool)
            for node_name in missing:
                jobs.manager.log(
                    job_id,
                    f"[launch] copying {server['model']} to {node_name} over the cluster link",
                )
                jobs.manager.report(job_id, {"phase": f"syncing to {node_name}"})
                sync_id = await sync.submit(server["model"], node_name)
                jobs.manager.log(job_id, f"[launch] sync job {sync_id}")
                if not await _await_job(job_id, sync_id):
                    jobs.manager.finish(
                        job_id, ok=False,
                        error=f"could not copy {server['model']} to {node_name}; see job {sync_id}",
                    )
                    return
                jobs.manager.log(job_id, f"[launch] {node_name} has the model")

        jobs.manager.log(job_id, "[launch] starting the engine")
        jobs.manager.report(job_id, {"phase": "starting the engine"})
        result = await start(server_id, force=force)
    except Exception as exc:
        jobs.manager.finish(job_id, ok=False, error=str(exc)[-600:])
        return

    if result.get("started"):
        jobs.manager.log(job_id, "[launch] engine started")
        jobs.manager.finish(job_id, ok=True, result=result,
                            progress={"phase": "started", "percent": 100.0})
    else:
        detail = result.get("error") or (result.get("safety") or {}).get("message") or "refused"
        jobs.manager.log(job_id, f"[launch] refused: {detail}")
        jobs.manager.finish(job_id, ok=False, error=detail, result=result)


async def _await_job(parent: str, child: str, poll: float = 2.0) -> bool:
    """Follow a child job, mirroring its progress onto the launch."""
    while True:
        await asyncio.sleep(poll)
        row = await asyncio.to_thread(jobs.manager.get, child)
        if row is None:
            return False
        child_progress = row.get("progress") or {}
        if child_progress:
            jobs.manager.report(parent, {"phase": "syncing", "child": child, **child_progress})
        if row["status"] in jobs.TERMINAL:
            return row["status"] == jobs.SUCCEEDED
