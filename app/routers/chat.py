"""Playground: talk to any running engine with the full parameter surface.

Requests are proxied rather than sent from the browser so that one origin works
regardless of which port an engine listens on, and so the dashboard can persist
transcripts. The proxy resolves endpoints by id from the discovered set; a raw
URL is accepted only for loopback and this cluster's own subnet, so the
dashboard cannot be turned into a general-purpose request forwarder.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import db, sampling_spec
from app import servers as svc

router = APIRouter(prefix="/chat", tags=["chat"])

# Long generations are normal; only the connect phase should be impatient.
TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)

ALLOWED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/24"),   # RoCE fabric
    ipaddress.ip_network("10.0.1.0/24"),
    ipaddress.ip_network("192.168.99.0/24"),   # LAN
]


def _url_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return False
    return any(address in network for network in ALLOWED_NETWORKS)


async def resolve(endpoint_id: str | None, url: str | None) -> str:
    if endpoint_id:
        for endpoint in await svc.endpoints():
            if endpoint["id"] == endpoint_id:
                return endpoint["url"]
        raise HTTPException(404, f"endpoint '{endpoint_id}' is not currently reachable")
    if url:
        if not _url_allowed(url):
            raise HTTPException(400, "that URL is outside the hosts this dashboard may talk to")
        return url.rstrip("/")
    raise HTTPException(422, "give either endpoint_id or url")


class ProxyRequest(BaseModel):
    endpoint_id: str | None = None
    url: str | None = None
    path: str = "/v1/chat/completions"
    body: dict = Field(default_factory=dict)


@router.get("/endpoints")
async def endpoints() -> dict:
    return {"endpoints": await svc.endpoints()}


@router.get("/params")
async def params(endpoint_id: str | None = None, url: str | None = None, kind: str = "chat") -> dict:
    """Parameter form for one endpoint, generated from its own OpenAPI schema."""
    try:
        base = await resolve(endpoint_id, url)
        payload = await sampling_spec.fetch(base)
    except HTTPException:
        payload = {**sampling_spec.fallback(), "source": "bundled snapshot"}
    name = "CompletionRequest" if kind == "completion" else "ChatCompletionRequest"
    return sampling_spec.ui_model(payload, name)


@router.post("/completions")
async def completions(request: ProxyRequest):
    """Forward one request, streaming the response through untouched."""
    base = await resolve(request.endpoint_id, request.url)
    if not request.path.startswith("/"):
        raise HTTPException(400, "path must be absolute")
    target = f"{base}{request.path}"
    streaming = bool(request.body.get("stream"))

    if not streaming:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(target, json=request.body)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"engine unreachable: {exc}") from None
        return StreamingResponse(
            iter([response.content]),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    async def relay() -> AsyncIterator[bytes]:
        client = httpx.AsyncClient(timeout=TIMEOUT)
        try:
            async with client.stream("POST", target, json=request.body) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode(errors="replace")
                    yield f"data: {json.dumps({'error': detail[:2000]})}\n\n".encode()
                    return
                async for chunk in response.aiter_raw():
                    yield chunk
        except httpx.HTTPError as exc:
            yield f"data: {json.dumps({'error': f'engine unreachable: {exc}'})}\n\n".encode()
        finally:
            await client.aclose()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/tokenize")
async def tokenize(request: ProxyRequest) -> dict:
    base = await resolve(request.endpoint_id, request.url)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{base}/tokenize", json=request.body)
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f"tokenize failed: {exc}") from None


# --- saved conversations -------------------------------------------------

class ChatIn(BaseModel):
    title: str = "Untitled"
    endpoint: str = ""
    model: str = ""
    params: dict = Field(default_factory=dict)
    messages: list = Field(default_factory=list)


def _row(chat_id: int) -> dict | None:
    return db.hydrate(
        db.query_one("SELECT * FROM chats WHERE id = ?", (chat_id,)), ("params", "messages")
    )


@router.get("/conversations")
async def list_conversations() -> dict:
    rows = await asyncio.to_thread(
        db.query,
        "SELECT id, title, endpoint, model, created_at, updated_at FROM chats"
        " ORDER BY updated_at DESC LIMIT 200",
    )
    return {"conversations": rows}


@router.post("/conversations", status_code=201)
async def create_conversation(payload: ChatIn) -> dict:
    now = db.now()
    cursor = await asyncio.to_thread(
        db.execute,
        "INSERT INTO chats (title, endpoint, model, params, messages, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            payload.title,
            payload.endpoint,
            payload.model,
            db.dumps(payload.params),
            db.dumps(payload.messages),
            now,
            now,
        ),
    )
    return await asyncio.to_thread(_row, int(cursor.lastrowid))


@router.get("/conversations/{chat_id}")
async def get_conversation(chat_id: int) -> dict:
    row = await asyncio.to_thread(_row, chat_id)
    if row is None:
        raise HTTPException(404, "no such conversation")
    return row


@router.put("/conversations/{chat_id}")
async def save_conversation(chat_id: int, payload: ChatIn) -> dict:
    if await asyncio.to_thread(_row, chat_id) is None:
        raise HTTPException(404, "no such conversation")
    await asyncio.to_thread(
        db.execute,
        "UPDATE chats SET title = ?, endpoint = ?, model = ?, params = ?, messages = ?,"
        " updated_at = ? WHERE id = ?",
        (
            payload.title,
            payload.endpoint,
            payload.model,
            db.dumps(payload.params),
            db.dumps(payload.messages),
            db.now(),
            chat_id,
        ),
    )
    return await asyncio.to_thread(_row, chat_id)


@router.delete("/conversations/{chat_id}", status_code=204)
async def delete_conversation(chat_id: int) -> None:
    await asyncio.to_thread(db.execute, "DELETE FROM chats WHERE id = ?", (chat_id,))


# --- presets -------------------------------------------------------------

class PresetIn(BaseModel):
    kind: str = "sampling"
    name: str
    data: dict = Field(default_factory=dict)


@router.get("/presets")
async def list_presets(kind: str | None = None) -> dict:
    sql = "SELECT * FROM presets" + (" WHERE kind = ?" if kind else "") + " ORDER BY name"
    rows = await asyncio.to_thread(db.query, sql, (kind,) if kind else ())
    return {"presets": [db.hydrate(row, ("data",)) for row in rows]}


@router.post("/presets", status_code=201)
async def save_preset(payload: PresetIn) -> dict:
    await asyncio.to_thread(
        db.execute,
        "INSERT INTO presets (kind, name, data, created_at) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(kind, name) DO UPDATE SET data = excluded.data",
        (payload.kind, payload.name, db.dumps(payload.data), db.now()),
    )
    return {"saved": True}


@router.delete("/presets/{preset_id}", status_code=204)
async def delete_preset(preset_id: int) -> None:
    await asyncio.to_thread(db.execute, "DELETE FROM presets WHERE id = ?", (preset_id,))
