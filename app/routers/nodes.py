"""The machines this dashboard can place work on."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import nodes as svc
from app import sync

router = APIRouter(prefix="/nodes", tags=["nodes"])


class NodeIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    address: str = ""
    docker_host: str = ""
    note: str = ""


@router.get("")
async def list_nodes() -> dict:
    return {"nodes": await svc.status_all(), "local": svc.LOCAL}


@router.get("/discover")
async def discover(scan_subnet: bool = False) -> dict:
    """Boxes this machine can already ssh into. Adding one stays deliberate."""
    return await svc.discover(scan_subnet=scan_subnet)


@router.post("", status_code=201)
async def add_node(payload: NodeIn) -> dict:
    if payload.name == svc.LOCAL:
        raise HTTPException(409, f"'{svc.LOCAL}' is this machine and is always present")
    node = await asyncio.to_thread(
        svc.add, payload.name, address=payload.address,
        docker_host=payload.docker_host, note=payload.note,
    )
    status = await svc.status(node)
    if not status.reachable:
        # Registered anyway: a peer that is merely powered off should not have
        # to be re-added when it comes back.
        return {**status.as_dict(), "warning": "added, but docker there is not reachable yet"}
    return status.as_dict()


@router.delete("/{name}", status_code=204)
async def remove_node(name: str) -> None:
    if name == svc.LOCAL:
        raise HTTPException(400, "this machine cannot be removed")
    await asyncio.to_thread(svc.remove, name)


class SyncIn(BaseModel):
    repo_id: str = Field(min_length=1)
    repo_type: str = "model"


@router.post("/{name}/sync", status_code=202)
async def sync_model(name: str, payload: SyncIn) -> dict:
    """Copy a cached model to a peer over the cluster link rather than pulling
    it from the internet again."""
    try:
        job_id = await sync.submit(payload.repo_id, name, payload.repo_type)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"job_id": job_id}


@router.get("/{name}/sync/check")
async def sync_check(name: str, repo_id: str, repo_type: str = "model") -> dict:
    node = svc.by_name(name)
    if node.is_local:
        raise HTTPException(400, "that is this machine; pick a peer")
    return await sync.preflight(repo_id, node, repo_type)


@router.get("/{name}/models")
async def node_models(name: str) -> dict:
    """What is in a peer's cache, so the UI can show what is missing there."""
    node = svc.by_name(name)
    if node.is_local:
        from app import hf

        return await hf.local_models()
    code, out = await svc._ssh(
        node.name or node.address,
        f"ls {svc.settings.hf_cache}/hub 2>/dev/null | grep -E '^(models|datasets)--' || true",
    )
    if code != 0:
        return {"ok": False, "error": out.strip()[:300], "repos": []}
    repos = []
    for entry in out.split():
        kind, _, rest = entry.partition("--")
        repos.append({
            "repo_id": rest.replace("--", "/", 1),
            "repo_type": "dataset" if kind == "datasets" else "model",
            "directory": entry,
        })
    return {"ok": True, "repos": repos}
