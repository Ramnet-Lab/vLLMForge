"""Host telemetry, memory budget, images and dashboard settings."""

from __future__ import annotations

import asyncio
import platform

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app import db, docker_ctl, events, memguard, safety, telemetry, vllm_spec
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
        "vllm_image": settings.vllm_image,
        "vllm_version": schema.get("vllm_version", "unknown"),
        "vllm_flags": len(schema.get("args", [])),
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
    return (await safety.check_launch(util)).as_dict()


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
    wanted = {
        "vllm": settings.vllm_image,
        "heretic": settings.heretic_image,
        "finetune": settings.finetune_image,
    }
    present = await asyncio.gather(*(docker_ctl.image_exists(tag) for tag in wanted.values()))
    local = await docker_ctl.ps(all_containers=True)
    return {
        "required": [
            {"role": role, "tag": tag, "present": ok}
            for (role, tag), ok in zip(wanted.items(), present, strict=True)
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
