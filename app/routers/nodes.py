"""The machines this dashboard can place work on."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import nodes as svc

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
