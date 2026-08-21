"""What models are on this machine and ready to be pointed at.

Three things count as a loadable model here, and a picker that offers only the
first is missing most of what the box actually holds:

  * a repo in the shared HuggingFace cache,
  * the merged model an earlier Heretic run wrote,
  * the merged export of an earlier fine-tune.

Adapters and datasets are deliberately excluded: neither can be loaded on its
own, and offering them in a model picker only produces a failure minutes later.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app import hf, jobs

# A directory only counts as a model if transformers could actually open it.
REQUIRED_FILE = "config.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")


def _is_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / REQUIRED_FILE).is_file():
        return False
    return any(child.suffix in WEIGHT_SUFFIXES for child in path.iterdir())


def _dir_bytes(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def _entry(value: str, label: str, *, kind: str, detail: str = "", note: str = "") -> dict:
    return {"value": value, "label": label, "kind": kind, "detail": detail, "note": note}


def _cached_models(local: dict) -> list[dict]:
    if not local.get("ok"):
        return []
    out = []
    for repo in local.get("repos", []):
        if repo.get("repo_type") != "model":
            continue
        # scan_cache_dir cannot tell a LoRA from a model, so the revision scan
        # that hf.local_models() already did is what decides.
        if any(rev.get("kind") == "adapter" for rev in repo.get("revisions", [])):
            continue
        out.append(
            _entry(
                repo["repo_id"],
                repo["repo_id"],
                kind="hub",
                detail=f"{repo['size_on_disk'] / 1024 ** 3:.1f} GiB cached",
            )
        )
    return sorted(out, key=lambda e: e["label"].lower())


def _heretic_outputs() -> list[dict]:
    out = []
    for job in jobs.manager.list("heretic", limit=100):
        if job["status"] != jobs.SUCCEEDED:
            continue
        meta = (job.get("spec") or {}).get("meta") or {}
        directory = Path(meta.get("output_dir") or "")
        if not _is_model_dir(directory):
            continue
        source = meta.get("model") or "unknown"
        out.append(
            _entry(
                str(directory),
                f"{source.split('/')[-1]} (abliterated)",
                kind="path",
                detail=f"{_dir_bytes(directory) / 1024 ** 3:.1f} GiB on disk",
                note=f"Heretic job {job['id']} from {source}",
            )
        )
    return out


def _finetune_outputs() -> list[dict]:
    out = []
    for job in jobs.manager.list("finetune", limit=100):
        if job["status"] != jobs.SUCCEEDED:
            continue
        meta = (job.get("spec") or {}).get("meta") or {}
        result = job.get("result") or (job.get("progress") or {}).get("result") or {}
        export = meta.get("export") or result.get("export") or "adapter"
        run_dir = Path(meta.get("run_dir") or "")
        # An adapter is not loadable on its own; only a merged export is.
        candidate = run_dir / export if export.startswith("merged") else run_dir / "model"
        if not _is_model_dir(candidate):
            continue
        source = meta.get("model") or result.get("model") or "unknown"
        name = (meta.get("config") or {}).get("name") or job["id"]
        out.append(
            _entry(
                str(candidate),
                f"{name} ({export})",
                kind="path",
                detail=f"{_dir_bytes(candidate) / 1024 ** 3:.1f} GiB on disk",
                note=f"fine-tune of {source}",
            )
        )
    return out


def _groups(local: dict) -> list[dict]:
    groups = [
        {"id": "cached", "label": "Cached models", "items": _cached_models(local)},
        {"id": "heretic", "label": "Heretic results", "items": _heretic_outputs()},
        {"id": "finetune", "label": "Fine-tune results", "items": _finetune_outputs()},
    ]
    return [group for group in groups if group["items"]]


async def loadable_models() -> dict[str, Any]:
    """Every model on this box something could be pointed at, grouped for a picker."""
    try:
        local = await hf.local_models()
    except Exception as exc:  # a broken cache must not empty the whole picker
        local = {"ok": False, "error": str(exc), "repos": []}

    groups = await asyncio.to_thread(_groups, local)
    return {
        "groups": groups,
        "count": sum(len(group["items"]) for group in groups),
        "cache_ok": bool(local.get("ok")),
        "cache_error": local.get("error", ""),
    }
