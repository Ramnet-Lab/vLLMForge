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

from app import hf, jobs, servers
from app.config import settings

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


# --- what a path-valued server flag can be set to -------------------------
#
# Everything here is expressed the way the vLLM container sees it. The engine
# runs with the model cache bind-mounted at /hf and the output directory at
# /outputs, so offering a host path would produce a launch that starts and then
# fails to find its file — minutes later, in a log.

CACHE_MOUNT = "/hf"

# Only these are worth walking; the model cache holds tens of thousands of blob
# files and is never the right answer for a template or a plugin.
SCAN_ROOTS = ("output_dir", "dataset_dir")

SUFFIXES = {
    "template": (".jinja", ".jinja2", ".j2"),
    "plugin": (".py",),
    "cert": (".pem", ".crt", ".key", ".cer"),
    "json": (".json",),
}
MAX_DEPTH = 4
MAX_HITS = 200


def _roots() -> list[Path]:
    return [getattr(settings, name) for name in SCAN_ROOTS]


def _walk(root: Path, keep) -> list[Path]:
    """Bounded walk. An unbounded one over a model directory would enumerate
    every shard and every blob for no benefit."""
    found: list[Path] = []
    if not root.is_dir():
        return found
    stack = [(root, 0)]
    while stack and len(found) < MAX_HITS:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if depth < MAX_DEPTH:
                    stack.append((entry, depth + 1))
            elif keep(entry):
                found.append(entry)
    return found


def _file_options(kind: str) -> list[dict]:
    suffixes = SUFFIXES[kind]
    out = []
    for root in _roots():
        for path in _walk(root, lambda p: p.suffix.lower() in suffixes):
            out.append(
                _entry(
                    servers.container_path(path),
                    str(path.relative_to(root)),
                    kind="path",
                    detail=f"{path.stat().st_size / 1024:.0f} KiB",
                    note=str(path),
                )
            )
    return sorted(out, key=lambda e: e["label"])


def _directory_options() -> list[dict]:
    out = [
        _entry(CACHE_MOUNT, "the shared model cache", kind="path", note=str(settings.hf_cache)),
        _entry("/outputs", "the output directory", kind="path", note=str(settings.output_dir)),
    ]
    for root in _roots():
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                out.append(
                    _entry(servers.container_path(entry), entry.name, kind="path", note=str(entry))
                )
    return out


def _adapter_options(local: dict) -> list[dict]:
    """LoRA adapters: cached ones from the Hub, and what fine-tuning produced."""
    out = []
    for repo in local.get("repos", []) if local.get("ok") else []:
        if repo.get("repo_type") != "model":
            continue
        if any(rev.get("kind") == "adapter" for rev in repo.get("revisions", [])):
            out.append(
            _entry(repo["repo_id"], repo["repo_id"], kind="hub", detail="cached adapter")
        )

    for job in jobs.manager.list("finetune", limit=100):
        if job["status"] != jobs.SUCCEEDED:
            continue
        meta = (job.get("spec") or {}).get("meta") or {}
        adapter = Path(meta.get("run_dir") or "") / "adapter"
        if not (adapter / "adapter_config.json").is_file():
            continue
        name = (meta.get("config") or {}).get("name") or job["id"]
        # vLLM takes --lora-modules as name=path.
        out.append(
            _entry(
                f"{name}={servers.container_path(adapter)}",
                f"{name} (fine-tune {job['id']})",
                kind="path",
                detail=f"{_dir_bytes(adapter) / 1024 ** 2:.0f} MiB",
                note=str(adapter),
            )
        )
    return out


async def path_options() -> dict[str, Any]:
    """Everything a path-valued serve flag could be set to, by kind."""
    try:
        local = await hf.local_models()
    except Exception as exc:
        local = {"ok": False, "error": str(exc), "repos": []}

    def build() -> dict[str, list[dict]]:
        models = _cached_models(local) + _heretic_outputs() + _finetune_outputs()
        return {
            "model": models,
            "adapter": _adapter_options(local),
            "template": _file_options("template"),
            "plugin": _file_options("plugin"),
            "cert": _file_options("cert"),
            "json": _file_options("json"),
            "directory": _directory_options(),
        }

    options = await asyncio.to_thread(build)
    return {
        "options": options,
        "counts": {kind: len(items) for kind, items in options.items()},
        "cache_ok": bool(local.get("ok")),
    }
