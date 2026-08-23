"""Host telemetry, memory budget, images and dashboard settings."""

from __future__ import annotations

import asyncio
import platform

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app import db, docker_ctl, engines, events, memguard, safety, telemetry, vllm_spec
from app.config import settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/info")
async def info() -> dict:
    docker_version = await docker_ctl.version() if await docker_ctl.available() else ""
    schema = vllm_spec.schema()
    return {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "docker": docker_version,
        # The three vllm_* scalars are retained rather than replaced: they are
        # what the header brand and the Overview's environment panel have always
        # read. `engines` is the shape that can answer for more than one.
        "vllm_image": settings.vllm_image,
        "vllm_version": schema.get("vllm_version", "unknown"),
        "vllm_flags": len(schema.get("args", [])),
        "engines": [
            {
                "name": engine.name,
                "label": engine.label,
                "image": engine.default_image,
                "version": engine.ui_model().get("version", "unknown"),
                # How many flags the FORM offers, which is the schema's count
                # less the ones the dashboard sets itself. Deliberately a
                # different number from `vllm_flags` above, which is the raw
                # schema count and is kept only because existing readers expect
                # it — the panel that renders this labels it per engine, so the
                # two never appear side by side claiming to be the same thing.
                "flags": sum(len(section["flags"]) for group in ("featured", "advanced")
                             for section in engine.ui_model().get(group, [])),
                "supports_pooling": engine.supports_pooling,
            }
            for engine in engines.order()
        ],
        "hf_cache": str(settings.hf_cache),
        "output_dir": str(settings.output_dir),
        "dataset_dir": str(settings.dataset_dir),
        "state_dir": str(settings.state_dir),
        "hf_token_set": bool(settings.hf_token),
        "memguard": {
            "enabled": settings.memguard_enabled,
            "threshold_mib": settings.memguard_threshold_mib,
        },
        "images": {
            "vllm": settings.vllm_image,
            "llamacpp": settings.llamacpp_image,
            "heretic": settings.heretic_image,
            "finetune": settings.finetune_image,
        },
    }


@router.get("/telemetry")
async def snapshot() -> dict:
    containers = await docker_ctl.ps(all_containers=False)
    return await telemetry.snapshot(
        containers=[
            {
                "name": c.get("Names"),
                "image": c.get("Image"),
                "status": c.get("Status"),
                "state": c.get("State"),
            }
            for c in containers
        ]
    )


@router.get("/telemetry/stream")
async def telemetry_stream() -> EventSourceResponse:
    async def generator():
        yield events.sse("telemetry", await snapshot())
        async for message in events.broker.subscribe(events.TELEMETRY):
            if isinstance(message, dict) and message.get("type") == "memguard":
                yield events.sse("memguard", message["event"])
            else:
                yield events.sse("telemetry", message)

    return EventSourceResponse(generator(), ping=15)


@router.get("/budget")
async def budget() -> dict:
    return (await safety.current_budget()).as_dict()


@router.get("/budget/check")
async def budget_check(util: float | None = None) -> dict:
    """What a vLLM launch at this utilisation would cost.

    Kept exactly as it was — a fraction and nothing else — because that is what
    it means and what every existing caller sends. A configuration that is not
    a fraction has a POST of its own below.
    """
    return (await safety.check_launch(util)).as_dict()


@router.post("/budget/check")
async def budget_check_config(payload: dict) -> dict:
    """What THIS configuration would cost, asked of the engine that will run it.

    A POST because the question is the whole argument set, not one number. It
    has to be: llama.cpp declares no fraction, so its footprint can only be
    worked out from the model file, the layer count and the context length
    together — and even for vLLM the fraction alone understates a config that
    sets --kv-cache-memory or --cpu-offload-gb.

    `server_id` names the definition being edited so its own container comes off
    the budget. Without it, reconfiguring a resident server is judged with that
    server's memory charged against its own restart.
    """
    from app import nodes as nodes_svc
    from app import servers as servers_svc

    engine = engines.get(payload.get("engine"))
    args = dict(payload.get("args") or {}) if isinstance(payload.get("args"), dict) else {}
    # The model comes separately because it is a managed flag and never lives in
    # `args` — and for an engine priced FROM its weights file, a configuration
    # without it cannot be sized at all.
    model = str(payload.get("model") or "")
    if model:
        args["model"] = model

    node = None
    name = str(payload.get("node") or "")
    if name and name != nodes_svc.LOCAL:
        node = nodes_svc.by_name(name)

    replacing = None
    server_id = payload.get("server_id")
    if isinstance(server_id, int):
        server = await asyncio.to_thread(servers_svc.get_server, server_id)
        if server is not None:
            replacing = servers_svc.container_name(server)
            args.setdefault("model", server.get("model") or "")

    verdict = await safety.check_launch(
        engine.declared_util(args), params=args, node=node,
        engine=engine.name, replacing=replacing)
    return verdict.as_dict()


@router.get("/memguard")
async def memguard_history() -> dict:
    return {
        "enabled": settings.memguard_enabled,
        "threshold_mib": settings.memguard_threshold_mib,
        "events": memguard.history(),
    }


@router.get("/images")
async def images() -> dict:
    """Which worker images are present, so the UI can offer to build them."""
    # Every one of these is built from docker/, not pulled — the UI needs the
    # file name so it can hand over a command that works.
    #
    # `required` says which images the box cannot work without: the vLLM one has
    # always been that, and llama.cpp is optional in the same way Heretic and
    # fine-tuning are — a box that never selects it never needs it, and a red
    # badge for an image nobody asked for is noise.
    wanted = {
        "vllm": (settings.vllm_image, "vllm.Dockerfile", True),
        "llamacpp": (settings.llamacpp_image, "llamacpp.Dockerfile", False),
        "heretic": (settings.heretic_image, "heretic.Dockerfile", False),
        "finetune": (settings.finetune_image, "finetune.Dockerfile", False),
    }
    present = await asyncio.gather(*(
        docker_ctl.image_exists(tag) for tag, _file, _req in wanted.values()))
    local = await docker_ctl.ps(all_containers=True)
    return {
        "required": [
            {"role": role, "tag": tag, "dockerfile": dockerfile,
             "present": ok, "essential": essential}
            for (role, (tag, dockerfile, essential)), ok in zip(
                wanted.items(), present, strict=True)
        ],
        "containers": len(local),
    }


@router.get("/settings")
async def get_settings() -> dict:
    return {
        "hf_token_set": bool(settings.hf_token or db.get_setting("hf_token", "")),
        "stored": db.get_setting("ui", {}),
    }


@router.put("/settings")
async def put_settings(payload: dict) -> dict:
    if "hf_token" in payload:
        db.set_setting("hf_token", payload.pop("hf_token") or "")
    if payload:
        db.set_setting("ui", {**(db.get_setting("ui", {}) or {}), **payload})
    return await get_settings()
