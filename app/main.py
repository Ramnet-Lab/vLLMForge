"""Application entry point.

Serves the API under /api and the no-build-step frontend from /web. Startup
adopts any work that outlived the previous process, then begins the telemetry
loop and the host-memory watchdog.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import db, docker_ctl, events, jobs, memguard, telemetry
from app import servers as server_service
from app.config import settings
from app.routers import chat, finetune, heretic, hub, servers, system
from app.routers import jobs as jobs_router
from app.routers import nodes as nodes_router

log = logging.getLogger("llmd")

BACKGROUND: list[asyncio.Task] = []


async def telemetry_loop() -> None:
    """One poll for all connected browsers, fanned out over the broker."""
    while True:
        try:
            if events.broker.subscriber_count(events.TELEMETRY):
                containers = await docker_ctl.ps(all_containers=False)
                snapshot = await telemetry.snapshot(
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
                await events.broker.publish(events.TELEMETRY, snapshot)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("telemetry poll failed")
        await asyncio.sleep(settings.telemetry_interval)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.ensure_dirs()
    await asyncio.to_thread(db.init_db)

    if not await docker_ctl.available():
        log.warning("docker is not reachable — container features will fail")

    with contextlib.suppress(Exception):
        await jobs.manager.reconcile()

    BACKGROUND.append(asyncio.create_task(telemetry_loop(), name="telemetry"))
    if settings.memguard_enabled:
        BACKGROUND.append(asyncio.create_task(memguard.watch(), name="memguard"))
    await server_service.autostart()

    try:
        yield
    finally:
        for task in BACKGROUND:
            task.cancel()
        for task in BACKGROUND:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await jobs.manager.shutdown()


app = FastAPI(
    title="LLM Dashboard",
    version="0.1.0",
    summary="vLLM serving, HuggingFace pulls, Unsloth fine-tuning and Heretic in one place",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    system.router,
    nodes_router.router,
    jobs_router.router,
    servers.router,
    hub.router,
    chat.router,
    finetune.router,
    heretic.router,
):
    app.include_router(router, prefix="/api")


@app.exception_handler(KeyError)
async def _missing(_request, exc: KeyError) -> JSONResponse:
    missing = exc.args[0] if exc.args else ""
    return JSONResponse({"detail": f"not found: {missing}"}, status_code=404)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    return {"ok": True}


# Static frontend last so /api wins any name collision.
if settings.web_dir.exists():
    app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(settings.web_dir / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        # The catch-all exists so a deep link like /serve loads the shell, but
        # it must not swallow a mistyped API route: returning the HTML shell
        # there surfaces as a JSON parse error in the browser instead of the
        # 404 that would actually explain the problem.
        if path == "api" or path.startswith("api/"):
            return JSONResponse({"detail": f"no such endpoint: /{path}"}, status_code=404)

        # Joining a request path onto a directory is an arbitrary file read
        # unless the result is checked: '../../etc/passwd' resolves cleanly and
        # Path.is_file() is perfectly happy to confirm it. Anything that does
        # not land inside web/ falls through to the shell.
        root = settings.web_dir.resolve()
        try:
            candidate = (root / path).resolve()
            inside = candidate == root or root in candidate.parents
        except (OSError, ValueError):
            inside = False
        if inside and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(root / "index.html")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        # SSE streams must not be buffered or timed out by the server.
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    main()
