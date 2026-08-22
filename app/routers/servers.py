"""vLLM server definitions and their containers."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from app import nodes as nodes_svc
from app import servers as svc
from app import vllm_spec
from app.config import settings

router = APIRouter(prefix="/servers", tags=["servers"])


class ServerIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1)
    port: int = Field(ge=1024, le=65535)
    served_name: str | None = None
    image: str | None = None
    args: dict = Field(default_factory=dict)
    env: dict = Field(default_factory=dict)
    notes: str = ""
    autostart: bool = False
    node: str = "local"
    pool_nodes: list[str] = Field(default_factory=list)

    @field_validator("args")
    @classmethod
    def _known_flags(cls, value: dict) -> dict:
        problems = vllm_spec.validate(value)
        if problems:
            raise ValueError("; ".join(problems))
        return value


class ServerPatch(BaseModel):
    name: str | None = None
    model: str | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)
    served_name: str | None = None
    image: str | None = None
    args: dict | None = None
    env: dict | None = None
    notes: str | None = None
    autostart: bool | None = None
    node: str | None = None
    pool_nodes: list[str] | None = None


@router.get("/schema")
async def parameter_schema() -> dict:
    """The full vLLM parameter surface, generated from the image itself."""
    return vllm_spec.ui_model()


@router.get("")
async def list_servers() -> dict:
    return await svc.status_all()


@router.get("/endpoints")
async def endpoints() -> dict:
    return {"endpoints": await svc.endpoints()}


@router.post("/pool/plan")
async def pool_plan(payload: dict) -> dict:
    """What pooling across these nodes would give, and what stands in the way."""
    from app import cluster

    return await cluster.plan(
        payload.get("nodes") or [],
        payload.get("model") or "",
        payload.get("args") if isinstance(payload.get("args"), dict) else None,
    )


@router.get("/pool/status")
async def pool_status(nodes: str = "", server_id: int | None = None) -> dict:
    """Pool health. Pass server_id to ask the engine itself — it is the Ray head,
    so its `ray status` is the authoritative view of who joined."""
    from app import cluster

    container = ""
    names = [n for n in nodes.split(",") if n]
    if server_id is not None:
        server = await asyncio.to_thread(svc.get_server, server_id)
        if server is None:
            raise HTTPException(404, "no such server")
        container = svc.container_name(server)
        names = names or svc.pool_of(server)
    return await cluster.status(names, container)


@router.get("/paths")
async def paths() -> dict:
    """What each path-valued serve flag can be set to, as the container sees it.

    Every entry is a value that can be handed to vLLM verbatim: a Hub id it will
    resolve through the mounted cache, or a path under /hf or /outputs. A host
    path would start the engine and then fail to find its file.
    """
    from app import catalog

    return await catalog.path_options()


@router.get("/suggest")
async def suggest() -> dict:
    return {"port": await asyncio.to_thread(svc.suggest_port), "image": settings.vllm_image}


@router.post("/recommend")
async def recommend(payload: dict) -> dict:
    """Which of the ~190 flags to set for this model on this node, and why.

    A POST because the current argument set is part of the question: a
    recommendation says what to change, and that depends on what is already set.
    """
    from app import recommend as recommender

    args = payload.get("args")
    pool = payload.get("pool")
    server_id = payload.get("server_id")
    result = await recommender.build(
        str(payload.get("model") or ""),
        str(payload.get("node") or ""),
        args if isinstance(args, dict) else None,
        [str(name) for name in pool] if isinstance(pool, list) else None,
        int(server_id) if isinstance(server_id, int) else None,
    )
    return result.to_dict()


@router.get("/profile")
async def profile(model: str = Query(""), node: str = Query("")) -> dict:
    """What the files beside a model's weights say it is.

    Read from the node that would run it, because a model cached here and a
    model cached on a peer are different questions and only one of them is
    about to be loaded.
    """
    from app import model_profile

    if not model.strip():
        return {"found": False, "reference": "", "source": "missing"}
    if node and node != nodes_svc.LOCAL:
        result = await model_profile.read_remote(model, node)
    else:
        result = await asyncio.to_thread(model_profile.read, model)
    return result.to_dict()


@router.post("", status_code=201)
async def create(payload: ServerIn) -> dict:
    if await asyncio.to_thread(svc.get_by_name, payload.name):
        raise HTTPException(409, f"a server named '{payload.name}' already exists")
    return await asyncio.to_thread(svc.create_server, payload.model_dump())


# Declared before the /{server_id} routes: FastAPI matches in order, and
# 'foreign' would otherwise be handed to the int path parameter and rejected.
@router.get("/foreign/metrics")
async def foreign_metrics(port: int) -> dict:
    """Metrics for a vLLM container the dashboard does not manage."""
    return await svc.metrics(port)


@router.post("/foreign/{name}/stop")
async def stop_foreign(name: str) -> dict:
    """Stop a hand-launched vLLM container to free memory for a managed one."""
    from app import docker_ctl

    state = await docker_ctl.state(name)
    if not state.exists:
        raise HTTPException(404, "no such container")
    from app import safety

    if not safety.is_vllm_command(state.command):
        raise HTTPException(400, "refusing to stop a container that is not running vLLM")
    await docker_ctl.stop(name)
    return {"stopped": True, "container": name}


@router.get("/{server_id}")
async def get(server_id: int) -> dict:
    server = await asyncio.to_thread(svc.get_server, server_id)
    if server is None:
        raise HTTPException(404, "no such server")
    return server


@router.patch("/{server_id}")
async def update(server_id: int, payload: ServerPatch) -> dict:
    if payload.args is not None:
        problems = vllm_spec.validate(payload.args)
        if problems:
            raise HTTPException(422, "; ".join(problems))
    if payload.name:
        # Names are unique in the table. Create checks and answers 409; rename
        # did not, so it reached sqlite and came back as an unhandled 500 —
        # which is now easy to hit, since a name derived from the model is
        # exactly the name a second server for that model would want.
        clash = await asyncio.to_thread(svc.get_by_name, payload.name)
        if clash and int(clash["id"]) != server_id:
            raise HTTPException(409, f"a server named '{payload.name}' already exists")
    server = await asyncio.to_thread(
        svc.update_server, server_id, payload.model_dump(exclude_none=True)
    )
    if server is None:
        raise HTTPException(404, "no such server")
    return server


@router.delete("/{server_id}", status_code=204)
async def delete(server_id: int) -> None:
    await svc.remove_container(server_id)
    await asyncio.to_thread(svc.delete_server, server_id)


@router.get("/{server_id}/preview")
async def preview(server_id: int) -> dict:
    server = await asyncio.to_thread(svc.get_server, server_id)
    if server is None:
        raise HTTPException(404, "no such server")
    return {
        "argv": svc.build_command(server),
        "command": svc.launch_preview(server),
    }


@router.post("/{server_id}/launch", status_code=202)
async def launch(server_id: int, force: bool = False) -> dict:
    """Start a server, doing whatever has to happen first.

    Returns a job rather than a result: a pooled launch may have to copy the
    model to another machine, which takes minutes, and that is progress to
    watch rather than an error to report.
    """
    try:
        return {"job_id": await svc.submit_launch(server_id, force=force)}
    except KeyError:
        raise HTTPException(404, "no such server") from None


@router.post("/{server_id}/start")
async def start(server_id: int, force: bool = False) -> dict:
    if await asyncio.to_thread(svc.get_server, server_id) is None:
        raise HTTPException(404, "no such server")
    # Deliberately not wrapped in `except KeyError`. It used to be, and any
    # KeyError raised anywhere inside the launch — including by this function's
    # own result handling — came back to the operator as "no such server" for a
    # server that plainly existed.
    result = await svc.start(server_id, force=force)
    if not result["started"]:
        safety = result.get("safety") or {}
        if result.get("error"):
            # docker refused the launch itself — a bound port, a missing image.
            raise HTTPException(502, detail={"message": result["error"], **safety})
        # 409: the request was well-formed, the host just cannot take it.
        raise HTTPException(409, detail=safety)
    return result


@router.post("/{server_id}/stop")
async def stop(server_id: int) -> dict:
    try:
        await svc.stop(server_id)
    except KeyError:
        raise HTTPException(404, "no such server") from None
    return {"stopped": True}


@router.post("/{server_id}/restart")
async def restart(server_id: int, force: bool = False) -> dict:
    try:
        await svc.stop(server_id)
        return await svc.start(server_id, force=force)
    except KeyError:
        raise HTTPException(404, "no such server") from None


@router.get("/{server_id}/logs", response_class=PlainTextResponse)
async def logs(server_id: int, tail: int = Query(400, le=10000)) -> str:
    try:
        return await svc.logs(server_id, tail=tail)
    except KeyError:
        raise HTTPException(404, "no such server") from None


@router.get("/{server_id}/metrics")
async def metrics(server_id: int) -> dict:
    server = await asyncio.to_thread(svc.get_server, server_id)
    if server is None:
        raise HTTPException(404, "no such server")
    return await svc.metrics(int(server["port"]))
