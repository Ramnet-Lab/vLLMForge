"""Unsloth fine-tuning: image, datasets, runs and promoting a result to a server."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app import finetune as svc
from app import hf, jobs, servers
from app.config import settings

router = APIRouter(prefix="/finetune", tags=["finetune"])

MAX_UPLOAD_BYTES = 512 * 1024 * 1024


class BuildIn(BaseModel):
    build_args: dict[str, str] = Field(
        default_factory=dict,
        description="Dockerfile ARGs, e.g. {'UNSLOTH_VERSION': '2026.8.19'} to pin a release.",
    )


class ServeIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    port: int | None = Field(default=None, ge=1024, le=65535)
    served_name: str = ""
    gpu_memory_utilization: float = Field(0.25, gt=0.0, lt=1.0)
    max_model_len: int | None = None
    args: dict = Field(default_factory=dict)


@router.get("/status")
async def status(probe: bool = False) -> dict:
    """Image presence and, on request, proof that unsloth imports inside it."""
    return await svc.status(probe=probe)


@router.post("/build", status_code=202)
async def build(payload: BuildIn | None = None) -> dict:
    try:
        job_id = await svc.build_image_job((payload.build_args if payload else None) or None)
    except FileNotFoundError as exc:
        raise HTTPException(500, f"missing dockerfile: {exc}") from None
    return {"job_id": job_id, "image": settings.finetune_image}


@router.get("/defaults")
async def form_schema() -> dict:
    return svc.ui_schema()


@router.post("/check")
async def check(config: svc.FinetuneConfig) -> dict:
    return await svc.check_memory(config)


@router.post("/jobs", status_code=201)
async def submit(config: svc.FinetuneConfig, force: bool = False) -> dict:
    if not await svc.image_present():
        raise HTTPException(409, f"{settings.finetune_image} is not built yet — POST /build first")
    verdict = await svc.check_memory(config)
    if not verdict["ok"] and not force:
        # Same contract as starting a server: well-formed request, unhappy host.
        raise HTTPException(409, detail=verdict)
    try:
        spec = await asyncio.to_thread(svc.build_job, config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    job_id = await jobs.manager.submit(spec)
    meta = spec.meta or {}
    return {"job_id": job_id, "run_dir": meta.get("run_dir"), "safety": verdict}


@router.get("/jobs")
async def runs(limit: int = Query(20, le=200)) -> dict:
    return {"jobs": await asyncio.to_thread(_recent, limit)}


def _recent(limit: int) -> list[dict]:
    rows = jobs.manager.list(svc.KIND, limit)
    for row in rows:
        meta = (row.get("spec") or {}).get("meta") or {}
        run_dir = Path(meta.get("run_dir") or "")
        row["artifacts"] = (
            sorted(p.name for p in run_dir.iterdir() if p.is_dir()) if run_dir.is_dir() else []
        )
    return rows


@router.get("/datasets")
async def datasets() -> dict:
    """Uploads, loose files in the dataset directory, and Hub datasets already
    pulled into the shared cache — the three places a training set can come
    from, in one list."""
    local, cached = await asyncio.gather(
        asyncio.to_thread(svc.list_datasets),
        hf.local_models(),
    )
    hub = [
        {
            "repo_id": repo["repo_id"],
            "source": "hub",
            "reference": repo["repo_id"],
            "size_bytes": repo["size_on_disk"],
            "nb_files": repo["nb_files"],
            "last_modified": repo["last_modified"],
            "path": repo["path"],
        }
        for repo in cached.get("repos", [])
        if repo.get("repo_type") == "dataset"
    ]
    return {**local, "cached": hub, "cache_ok": bool(cached.get("ok"))}


@router.post("/datasets/upload", status_code=201)
async def upload(
    file: UploadFile = File(...),
    name: str = Form(""),
    replace: bool = Form(False),
) -> dict:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB")
    label = name or Path(file.filename or "dataset").stem
    try:
        return await asyncio.to_thread(svc.store_upload, label, raw, replace=replace)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@router.post("/jobs/{job_id}/serve", status_code=201)
async def serve(job_id: str, payload: ServeIn) -> dict:
    job = await asyncio.to_thread(jobs.manager.get, job_id)
    if job is None or job["kind"] != svc.KIND:
        raise HTTPException(404, "no such fine-tune job")
    if job["status"] != jobs.SUCCEEDED:
        raise HTTPException(409, f"this run is {job['status']}, not succeeded")
    if await asyncio.to_thread(servers.get_by_name, payload.name):
        raise HTTPException(409, f"a server named '{payload.name}' already exists")

    try:
        plan = await asyncio.to_thread(svc.serve_plan, job, name=payload.name)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None

    from app import engines

    engine = engines.get(plan.get("engine"))
    args = {**plan["args"], **payload.args}
    # The utilisation fraction and max-model-len are vLLM's, and a gguf export
    # is served by llama.cpp — forcing them onto it would produce a definition
    # that fails validation with flags the operator never typed.
    if engine.name == "vllm":
        args["gpu_memory_utilization"] = payload.gpu_memory_utilization
        if payload.max_model_len:
            args["max_model_len"] = payload.max_model_len
    problems = engine.validate(args)
    if problems:
        raise HTTPException(422, "; ".join(problems))

    port = payload.port or await asyncio.to_thread(servers.suggest_port)
    server = await asyncio.to_thread(
        servers.create_server,
        {
            "name": payload.name,
            "engine": engine.name,
            "model": plan["model"],
            "served_name": payload.served_name or payload.name,
            "port": port,
            "args": args,
            "notes": plan["note"],
        },
    )
    return {"server": server, "note": plan["note"], "adapter": plan["adapter"]}
