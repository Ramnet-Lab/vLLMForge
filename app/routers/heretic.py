"""Heretic abliteration runs."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app import heretic as svc
from app import jobs, safety, servers
from app.config import settings

router = APIRouter(prefix="/heretic", tags=["heretic"])

# Anything above this and a Heretic result on this box would be competing with
# the vLLM servers it was abliterated to replace.
MAX_SERVE_UTIL = 0.35
MIN_SERVE_UTIL = 0.10


class BuildIn(BaseModel):
    ref: str = Field(default="master", min_length=1, max_length=120)


@router.get("/status")
async def status() -> dict:
    image = await svc.image_status()
    return {**image, "defaults": svc.defaults().model_dump()}


@router.get("/defaults")
async def form_schema() -> dict:
    return svc.ui_model()


@router.post("/build", status_code=202)
async def build(payload: BuildIn) -> dict:
    return {"job_id": await svc.submit_build(payload.ref)}


@router.post("/check")
async def check(cfg: svc.HereticSettings) -> dict:
    return await svc.preflight(cfg)


@router.post("/jobs", status_code=201)
async def submit(cfg: svc.HereticSettings, force: bool = False) -> dict:
    if not (await svc.image_status())["present"]:
        raise HTTPException(
            503, f"{settings.heretic_image} has not been built — POST /api/heretic/build first"
        )
    verdict = await svc.preflight(cfg)
    if not verdict["ok"] and not force:
        # Same 409 contract as a blocked vLLM launch, so the UI renders it with
        # the same component and can retry with ?force=true.
        raise HTTPException(409, detail=verdict)
    spec = await asyncio.to_thread(svc.build_job, cfg)
    job_id = await jobs.manager.submit(spec)
    return {"job_id": job_id, "safety": verdict, "meta": spec.meta}


@router.get("/jobs")
async def list_jobs(limit: int = Query(50, le=500)) -> dict:
    rows = await asyncio.to_thread(jobs.manager.list, "heretic", limit)
    return {"jobs": [_decorate(row) for row in rows]}


def _decorate(row: dict) -> dict:
    meta = (row.get("spec") or {}).get("meta") or {}
    out = Path(meta.get("output_dir", ""))
    return {
        **row,
        "meta": meta,
        # A directory only counts as a servable model once the config lands next
        # to the weights; Heretic writes both at the very end.
        "has_output": bool(meta.get("output_dir")) and (out / "config.json").is_file(),
    }


@router.post("/jobs/{job_id}/serve", status_code=201)
async def serve(job_id: str) -> dict:
    job = await asyncio.to_thread(jobs.manager.get, job_id)
    if job is None or job["kind"] != "heretic":
        raise HTTPException(404, "no such heretic job")
    meta = (job.get("spec") or {}).get("meta") or {}
    out = Path(meta["output_dir"]) if meta.get("output_dir") else None
    if out is None or not (out / "config.json").is_file():
        raise HTTPException(409, f"{out or 'this job'} does not hold a finished model yet")
    if (meta.get("settings") or {}).get("export_strategy") == "adapter":
        raise HTTPException(
            409,
            "this job exported a LoRA adapter, not a standalone model — serve the base model "
            "with --enable-lora and register the adapter instead",
        )

    budget = await safety.current_budget()
    util = round(max(MIN_SERVE_UTIL, min(MAX_SERVE_UTIL, budget.free_util - 0.05)), 2)
    pool = budget.total_bytes / safety.GIB
    explanation = (
        f"gpu_memory_utilization is {util:g}, not vLLM's default 0.92, because GPU memory is host "
        f"memory here: 0.92 would reserve {0.92 * pool:.0f} GiB of the {pool:.0f} GiB pool and "
        f"starve everything already resident. {util:g} reserves {util * pool:.0f} GiB, and "
        "max_model_len is capped at 4096 so the KV cache stays inside it. Raise both once you "
        "know what else is running."
    )
    payload = {
        "name": await asyncio.to_thread(_free_name, meta.get("model", job_id), job_id),
        "model": servers.container_path(out),
        "served_name": f"heretic-{job_id}",
        "port": await asyncio.to_thread(servers.suggest_port),
        # Stated rather than defaulted: a Heretic export is safetensors, so vLLM
        # is genuinely the right engine here and not an accident of what the
        # default happened to be on the day this was written.
        "engine": "vllm",
        "image": settings.vllm_image,
        "args": {"gpu_memory_utilization": util, "max_model_len": 4096},
        "notes": (
            f"Abliterated from {meta.get('model', '?')} by Heretic job {job_id}. {explanation}"
        ),
    }
    server = await asyncio.to_thread(servers.create_server, payload)
    return {
        "server": server,
        "explanation": explanation,
        "safety": (await safety.check_launch(util)).as_dict(),
    }


def _free_name(model: str, job_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", model.split("/")[-1]).strip("-") or "model"
    candidate = f"heretic-{stem}"[:56]
    if servers.get_by_name(candidate) is None:
        return candidate
    return f"{candidate}-{job_id[:6]}"
