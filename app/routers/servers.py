"""Server definitions and their containers, for either inference engine."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from app import engines
from app import nodes as nodes_svc
from app import servers as svc

router = APIRouter(prefix="/servers", tags=["servers"])


class ServerIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    # Declared BEFORE `args`, and the order is load-bearing rather than
    # cosmetic: pydantic runs field validators in declaration order and
    # `info.data` holds only the fields already validated, so an `engine`
    # declared after `args` would be invisible to the validator below — which
    # would then check every llama.cpp argument against vLLM's schema and reject
    # all of them as unknown flags.
    engine: str = Field(default=engines.DEFAULT)
    model: str = Field(min_length=1)
    port: int = Field(ge=1024, le=65535)
    served_name: str | None = None
    image: str | None = None
    args: dict = Field(default_factory=dict)
    # The editor's memory of the OTHER engines' argument sets. Declared, because
    # pydantic drops what it does not declare — and a silently dropped stash is
    # a switch back to vLLM that finds the flags gone.
    args_by_engine: dict = Field(default_factory=dict)
    env: dict = Field(default_factory=dict)
    notes: str = ""
    autostart: bool = False
    node: str = "local"
    pool_nodes: list[str] = Field(default_factory=list)

    @field_validator("engine")
    @classmethod
    def _known_engine(cls, value: str) -> str:
        if not engines.known(value):
            raise ValueError(
                f"unknown engine '{value}'; this build has {', '.join(engines.names())}")
        return value

    @field_validator("args")
    @classmethod
    def _known_flags(cls, value: dict, info: ValidationInfo) -> dict:
        problems = engines.get(info.data.get("engine")).validate(value)
        if problems:
            raise ValueError("; ".join(problems))
        return value

    @model_validator(mode="after")
    def _pooling_is_vllms(self) -> ServerIn:
        engine = engines.get(self.engine)
        if self.pool_nodes and not engine.supports_pooling:
            raise ValueError(
                f"{engine.label} cannot be pooled across machines: pooling here is vLLM's "
                "pipeline-parallel path, and llama.cpp distributes over --rpc instead")
        return self


class ServerPatch(BaseModel):
    name: str | None = None
    # Present so a PATCH carrying an engine is not silently dropped by pydantic,
    # which would make an engine change look like it succeeded and change
    # nothing. The args it arrives with are validated against it, and the change
    # is refused outright while a container exists — see `update`.
    engine: str | None = None
    model: str | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)
    served_name: str | None = None
    image: str | None = None
    args: dict | None = None
    args_by_engine: dict | None = None
    env: dict | None = None
    notes: str | None = None
    autostart: bool | None = None
    node: str | None = None
    pool_nodes: list[str] | None = None

    @field_validator("engine")
    @classmethod
    def _known_engine(cls, value: str | None) -> str | None:
        if value is not None and not engines.known(value):
            raise ValueError(
                f"unknown engine '{value}'; this build has {', '.join(engines.names())}")
        return value


@router.get("/engines")
async def engine_list() -> dict:
    """Which engines this build can serve with, for the editor's dropdown."""
    return {
        "default": engines.DEFAULT,
        "engines": [
            {
                "name": engine.name,
                "label": engine.label,
                "image": engine.default_image,
                "supports_pooling": engine.supports_pooling,
                "declares_util": engine.implicit_util() is not None,
                "version": engine.ui_model().get("version", "unknown"),
            }
            for engine in engines.order()
        ],
    }


def _engine_or_404(name: str):
    """The engine by name, refusing a name this build does not have.

    `engines.get` falls back to vLLM, which is right for a stored row — a
    corrupt `engine` column must not 500 the Serve page. It is wrong for a query
    parameter: answering `?engine=llamacp` with vLLM's form, and a payload whose
    own `engine` key says vllm, is a typo that looks like a working request. The
    write path already 422s an unknown engine; this makes the read path agree.
    """
    if not engines.known(name):
        raise HTTPException(
            404, f"unknown engine '{name}'; this build has {', '.join(engines.names())}")
    return engines.get(name)


@router.get("/schema")
async def parameter_schema(engine: str = Query(engines.DEFAULT)) -> dict:
    """One engine's full parameter surface, as generated from its own binary.

    Unparameterised this returns exactly what it always returned, so a client
    that predates the second engine is unaffected.
    """
    return _engine_or_404(engine).ui_model()


@router.get("")
async def list_servers() -> dict:
    return await svc.status_all()


@router.get("/endpoints")
async def endpoints() -> dict:
    return {"endpoints": await svc.endpoints()}


@router.post("/pool/plan")
async def pool_plan(payload: dict) -> dict:
    """What pooling across these nodes would give, and what stands in the way.

    `server_id` names the definition being edited, so its own ranks come off the
    budget. It matters more than it looks: this plan gates the Save & start
    button, and a far rank is an ordinary tenant of the machine it runs on, so
    without it every existing pooled server is judged with its own memory
    counted against itself on every node it spans.
    """
    from app import cluster

    engine = engines.get(payload.get("engine"))
    if not engine.supports_pooling:
        return {"ok": False, "reason": (
            f"{engine.label} cannot be pooled across machines: pooling here is vLLM's "
            "pipeline-parallel path, and llama.cpp distributes over --rpc instead")}

    server_id = payload.get("server_id")
    replacing = None
    image = str(payload.get("image") or "")
    if isinstance(server_id, int):
        server = await asyncio.to_thread(svc.get_server, server_id)
        if server is not None:
            replacing = svc.container_name(server)
            image = image or svc.image_of(server)
    return await cluster.plan(
        payload.get("nodes") or [],
        payload.get("model") or "",
        payload.get("args") if isinstance(payload.get("args"), dict) else None,
        replacing=replacing,
        image=image,
    )


@router.get("/pool/status")
async def pool_status(nodes: str = "", server_id: int | None = None) -> dict:
    """Pool health: the state of every rank of the engine.

    Pass server_id and the ranks are resolved from the definition, which is the
    only way to know what the far ranks are called."""
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
async def paths(engine: str = Query(engines.DEFAULT)) -> dict:
    """What each path-valued serve flag can be set to, as the container sees it.

    Every entry is a value that can be handed to the engine verbatim: a Hub id
    it will resolve through the mounted cache, or a path under /hf or /outputs.
    A host path would start the engine and then fail to find its file.

    The engine matters for one kind and one only: `model`. vLLM is pointed at a
    repo and resolves the weights itself; llama.cpp is pointed at one .gguf FILE,
    and a repo holding six quantisations is six different answers.
    """
    from app import catalog

    return await catalog.path_options(engine=_engine_or_404(engine).name)


@router.get("/suggest")
async def suggest(engine: str = Query(engines.DEFAULT)) -> dict:
    """A free port, and the image that engine runs in.

    The image has to follow the engine or a llama.cpp definition is created
    pointing at the vLLM image and dies at `docker run`.
    """
    chosen = _engine_or_404(engine)
    return {"port": await asyncio.to_thread(svc.suggest_port),
            "image": chosen.default_image, "engine": chosen.name}


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
    env = payload.get("env")

    # The two advisors share nothing but their output shape, and that is not a
    # gap: almost every rule in the vLLM one is reasoning about vLLM's own
    # source. Routing here rather than branching inside means a llama.cpp form
    # never sees a suggestion naming a flag it does not have — or the GGUF
    # refusal, which for that engine is the success condition.
    if engines.get(payload.get("engine")).name == "llamacpp":
        from app import recommend_llamacpp

        result = await recommend_llamacpp.build(
            str(payload.get("model") or ""),
            str(payload.get("node") or ""),
            args if isinstance(args, dict) else None,
            int(server_id) if isinstance(server_id, int) else None,
        )
        return result.to_dict()

    result = await recommender.build(
        str(payload.get("model") or ""),
        str(payload.get("node") or ""),
        args if isinstance(args, dict) else None,
        [str(name) for name in pool] if isinstance(pool, list) else None,
        int(server_id) if isinstance(server_id, int) else None,
        env if isinstance(env, dict) else None,
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
async def foreign_metrics(port: int, engine: str = Query(engines.DEFAULT)) -> dict:
    """Metrics for an engine container the dashboard does not manage.

    The engine has to come along: filtering a llama.cpp scrape through vLLM's
    series names returns an empty panel that looks like a broken server.
    """
    return await svc.metrics(port, engine=engine)


@router.post("/foreign/{name}/stop")
async def stop_foreign(name: str) -> dict:
    """Stop a hand-launched engine container to free memory for a managed one.

    Recognising both engines here is what makes the memory guard's own advice —
    "stop something first" — actionable. Until it did, an operator could see a
    llama-server holding 40 GiB in the foreign table and still had no way to
    stop the container that was actually holding the memory.
    """
    from app import docker_ctl

    state = await docker_ctl.state(name)
    if not state.exists:
        raise HTTPException(404, "no such container")
    if engines.recognise(state) is None:
        raise HTTPException(
            400, "refusing to stop a container that is not running a recognised engine")
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
    # The row is loaded first because the patch cannot be judged without it: the
    # arguments have to be validated against the engine this server *is*, not
    # against vLLM's schema, or every edit to a llama.cpp definition comes back
    # as "unknown parameter for this vLLM build".
    existing = await asyncio.to_thread(svc.get_server, server_id)
    if existing is None:
        raise HTTPException(404, "no such server")

    engine = engines.get(payload.engine or existing.get("engine"))
    if payload.engine and engine.name != engines.get(existing.get("engine")).name:
        # The engine name is also the container kind, and the container's name is
        # recomputed from the row. Changing it under a live container renames
        # that container out from under everything that addresses it: stop()
        # would miss it, it would keep its full share of the machine, and it
        # would reappear as somebody else's engine with no owner.
        running = await svc.container_running(existing)
        if running is not False:
            unreachable = " (its node did not answer, so this cannot be checked)" \
                if running is None else ""
            raise HTTPException(409, (
                f"stop {existing['name']} before changing its engine{unreachable} — its "
                f"container is named after the engine, so switching now would leave "
                f"{svc.container_name(existing)} running with nothing able to stop it"))

    if payload.args is not None:
        problems = engine.validate(payload.args)
        if problems:
            raise HTTPException(422, "; ".join(problems))

    pool = payload.pool_nodes if payload.pool_nodes is not None else svc.pool_of(existing)
    if pool and not engine.supports_pooling:
        raise HTTPException(422, (
            f"{engine.label} cannot be pooled across machines: pooling here is vLLM's "
            "pipeline-parallel path, and llama.cpp distributes over --rpc instead"))

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
    # The row knows its engine, so the caller does not have to say — and the
    # engine is what decides which series are promoted out of the scrape.
    return await svc.metrics(int(server["port"]), engine=svc.engine_for(server).name)
