"""HuggingFace Hub browsing, sizing, pulls and the local cache."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app import db, hf

router = APIRouter(prefix="/hub", tags=["hub"])


class DownloadIn(BaseModel):
    repo_id: str = Field(min_length=1)
    revision: str = "main"
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None


class TokenIn(BaseModel):
    token: str = ""


def _http(exc: hf.HubError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


@router.get("/search")
async def search(
    q: str = "",
    pipeline_tag: str | None = None,
    author: str | None = None,
    sort: str = "downloads",
    limit: int = Query(30, ge=1, le=100),
    gated: bool | None = None,
) -> dict:
    try:
        models = await hf.search(
            q, pipeline_tag=pipeline_tag, author=author, sort=sort, limit=limit, gated=gated
        )
    except hf.HubError as exc:
        raise _http(exc) from None
    return {"models": models, "query": q}


@router.get("/local")
async def local() -> dict:
    return await hf.local_models()


@router.get("/token")
async def token() -> dict:
    source = await asyncio.to_thread(hf.token_source)
    return {"configured": bool(source), "source": source}


@router.put("/token")
async def put_token(payload: TokenIn) -> dict:
    """Persist a token; an empty string clears it back to whatever the env holds."""
    await asyncio.to_thread(db.set_setting, "hf_token", payload.token.strip())
    source = await asyncio.to_thread(hf.token_source)
    return {"configured": bool(source), "source": source}


@router.post("/download", status_code=202)
async def download(payload: DownloadIn) -> dict:
    try:
        job_id = await hf.submit_download(
            payload.repo_id,
            revision=payload.revision,
            allow_patterns=payload.allow_patterns,
            ignore_patterns=payload.ignore_patterns,
        )
    except hf.HubError as exc:
        raise _http(exc) from None
    return {"job_id": job_id}


@router.delete("/local/{repo_id:path}")
async def delete_local(repo_id: str) -> dict:
    try:
        return await hf.delete_local(repo_id)
    except hf.HubError as exc:
        raise _http(exc) from None


@router.get("/models/{repo_id:path}")
async def model(repo_id: str, revision: str = "main") -> dict:
    try:
        return await hf.model_detail(repo_id, revision)
    except hf.HubError as exc:
        raise _http(exc) from None
