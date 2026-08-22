"""HuggingFace Hub: search, sizing, the local cache, and pull jobs.

Sizing goes over raw HTTP rather than through huggingface_hub. The two calls
that matter — `treesize` for the headline number and the recursive `tree` for
per-file bytes — are one request each and need no cache lock, whereas the
library's equivalents want a writable cache this process does not have.

The cache at settings.hf_cache is root-owned. The dashboard user can *read* it,
so scanning and "is this blob already here" checks are cheap and local; every
write — download, delete — has to happen inside a container running as root.
That split is the reason this module is half httpx and half docker.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import scan_cache_dir

from app import db, docker_ctl, jobs, telemetry
from app.config import REPO_ROOT, settings

ENDPOINT = "https://huggingface.co"
API_TIMEOUT = 20.0
GIB = 1024 ** 3

PROGRESS_MARKER = "@@PROGRESS@@"
RESULT_MARKER = "@@RESULT@@"

WORKER_DIR = REPO_ROOT / "app" / "workers"
CONTAINER_CACHE = "/hf"

# Exactly the fields the list view renders. `expand[]` is cheaper than
# `full=true`, which would drag a full siblings array along for every hit.
SEARCH_EXPAND = (
    "author", "downloads", "gated", "lastModified", "library_name",
    "likes", "pipeline_tag", "private", "tags",
)
SORT_ALIASES = {"trending": "trendingScore", "updated": "lastModified", "created": "createdAt"}

# One path component, no leading dot: that also rules out "..", which matters
# because this string ends up naming a directory a root container deletes.
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")
_CACHE_DIR = re.compile(r"(?:models|datasets)--[A-Za-z0-9._-]+")

# The Hub mirrors its whole API under /api/models and /api/datasets, and
# huggingface_hub mirrors the cache layout the same way, so one code path serves
# both once the segment and the directory prefix are parameters.
REPO_TYPES = {"model": ("models", "models--"), "dataset": ("datasets", "datasets--")}


def _repo_type(repo_type: str) -> tuple[str, str]:
    try:
        return REPO_TYPES[repo_type]
    except KeyError:
        raise HubError(f"unknown repo type '{repo_type}'", 400) from None
_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')
_RATE_RESET = re.compile(r"t=(\d+)")


class HubError(RuntimeError):
    """A Hub failure with an HTTP status the router can hand straight back."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


# --- plumbing -----------------------------------------------------------

def token() -> str:
    """The token to hand anything that talks to the Hub.

    The environment wins over the one stored from the Models tab, but both reach
    every container the dashboard launches. Letting a stored token authenticate
    a download while a server launched from the same UI went out anonymous was a
    difference nobody could be expected to predict.
    """
    if settings.hf_token:
        return settings.hf_token
    return db.get_setting("hf_token", "") or ""


async def stored_token() -> str:
    return await asyncio.to_thread(token)


def token_source() -> str:
    if settings.hf_token:
        return "env"
    return "stored" if db.get_setting("hf_token", "") else ""


def _fail(response: httpx.Response, repo_id: str, authenticated: bool) -> None:
    code = response.status_code
    if code == 429:
        match = _RATE_RESET.search(response.headers.get("ratelimit", ""))
        wait = f" Retry in {match.group(1)}s." if match else ""
        hint = "" if authenticated else " Configuring an HF token doubles the anonymous limit."
        raise HubError(f"HuggingFace rate limit reached.{wait}{hint}", 429)
    if code in (401, 403):
        # The Hub deliberately does not distinguish private from nonexistent.
        raise HubError(
            f"'{repo_id}' is not accessible with this token — it may be private, gated, "
            "or may not exist.",
            403,
        )
    if code == 404:
        raise HubError(f"'{repo_id}' was not found on the Hub.", 404)
    raise HubError(f"the Hub returned HTTP {code} for '{repo_id}'.", 502)


async def _get(url: str, params: Any = None, *, label: str = "") -> httpx.Response:
    token = await stored_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if not url.startswith("http"):
        url = ENDPOINT + url
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url, params=params, headers=headers)
    except httpx.HTTPError as exc:
        raise HubError(f"could not reach huggingface.co: {exc}", 503) from exc
    if response.status_code >= 400:
        _fail(response, label or url, bool(token))
    return response


def check_repo_id(repo_id: str) -> str:
    if not _REPO_ID.fullmatch(repo_id or ""):
        raise HubError(f"'{repo_id}' is not a valid HuggingFace repo id.", 400)
    return repo_id


# --- search -------------------------------------------------------------

def _normalise(item: dict) -> dict:
    repo_id = item.get("id") or item.get("modelId") or ""
    return {
        "id": repo_id,
        "author": item.get("author") or (repo_id.split("/")[0] if "/" in repo_id else ""),
        "downloads": int(item.get("downloads") or 0),
        "likes": int(item.get("likes") or 0),
        "updated": item.get("lastModified") or item.get("createdAt") or "",
        "pipeline_tag": item.get("pipeline_tag") or "",
        "tags": list(item.get("tags") or []),
        # false | "auto" | "manual" — kept verbatim, the UI distinguishes them.
        "gated": item.get("gated", False),
        "private": bool(item.get("private")),
        "library": item.get("library_name") or "",
    }


async def search(
    query: str = "",
    *,
    pipeline_tag: str | None = None,
    author: str | None = None,
    sort: str = "downloads",
    limit: int = 30,
    gated: bool | None = None,
    repo_type: str = "model",
) -> list[dict]:
    segment, _prefix = _repo_type(repo_type)
    dataset = repo_type == "dataset"
    params: list[tuple[str, str]] = [("limit", str(max(1, min(limit, 100))))]
    if query:
        params.append(("search", query))
    if pipeline_tag and not dataset:
        params.append(("pipeline_tag", pipeline_tag))
    if author:
        params.append(("author", author))
    if gated is not None:
        params.append(("gated", "true" if gated else "false"))
    if sort:
        params.append(("sort", SORT_ALIASES.get(sort, sort)))
        params.append(("direction", "-1"))
    params += [("expand[]", field) for field in SEARCH_EXPAND if not dataset]

    response = await _get(f"/api/{segment}", params, label=f"{repo_type} search")
    payload = response.json()
    return [_normalise(item) for item in payload] if isinstance(payload, list) else []


# --- detail & sizing ----------------------------------------------------

async def _model_info(repo_id: str, revision: str, repo_type: str = "model") -> dict:
    segment, _prefix = _repo_type(repo_type)
    path = f"/api/{segment}/{repo_id}"
    if revision != "main":
        path += f"/revision/{revision}"
    return (await _get(path, label=repo_id)).json()


async def _tree(repo_id: str, revision: str, repo_type: str = "model") -> list[dict]:
    """Every file in a revision, following the Hub's cursor pagination."""
    segment, _prefix = _repo_type(repo_type)
    url: str | None = f"/api/{segment}/{repo_id}/tree/{revision}"
    params: Any = {"recursive": "true", "limit": "1000"}
    entries: list[dict] = []
    while url and len(entries) < 20000:
        response = await _get(url, params, label=repo_id)
        batch = response.json()
        if not isinstance(batch, list):
            break
        entries.extend(batch)
        match = _NEXT_LINK.search(response.headers.get("link", ""))
        url, params = (match.group(1) if match else None), None
    return entries


async def _raw_json(repo_id: str, revision: str, filename: str) -> dict:
    """A file's contents from /resolve/, or {} when it is absent or blocked."""
    try:
        response = await _get(f"/{repo_id}/resolve/{revision}/{filename}", label=repo_id)
        payload = response.json()
    except (HubError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _files(tree: list[dict]) -> list[dict]:
    out = [
        {
            "path": entry.get("path", ""),
            "size": int(entry.get("size") or 0),
            # The blob the cache would store this under: git sha1 normally,
            # the LFS sha256 for large files.
            "etag": (entry.get("lfs") or {}).get("oid") or entry.get("oid") or "",
            "lfs": bool(entry.get("lfs")),
        }
        for entry in tree
        if entry.get("type") == "file"
    ]
    return sorted(out, key=lambda f: f["path"])


def _blob_dir(repo_id: str, repo_type: str = "model") -> Path:
    _segment, prefix = _repo_type(repo_type)
    return settings.hf_cache / "hub" / f"{prefix}{repo_id.replace('/', '--')}" / "blobs"


def _cached_bytes(
    repo_id: str, files: list[dict], repo_type: str = "model"
) -> tuple[int, int]:
    """How much of a revision is already on disk, by blob presence."""
    blobs = _blob_dir(repo_id, repo_type)
    if not blobs.is_dir():
        return 0, 0
    total = count = 0
    for entry in files:
        if entry["etag"] and (blobs / entry["etag"]).exists():
            total += entry["size"]
            count += 1
    return total, count


def _estimate(
    repo_id: str, files: list[dict], total_bytes: int, repo_type: str = "model"
) -> dict:
    cached_bytes, cached_files = _cached_bytes(repo_id, files, repo_type)
    return {
        "total_bytes": total_bytes,
        "cached_bytes": cached_bytes,
        "fetch_bytes": max(total_bytes - cached_bytes, 0),
        "files_total": len(files),
        "files_cached": cached_files,
        "cache_dir": str(settings.hf_cache / "hub"),
    }


async def estimate_download(
    repo_id: str, revision: str = "main", repo_type: str = "model"
) -> dict:
    """What a pull would cost: treesize as the headline, minus local blobs."""
    check_repo_id(repo_id)
    segment, _prefix = _repo_type(repo_type)
    size, tree = await asyncio.gather(
        _get(f"/api/{segment}/{repo_id}/treesize/{revision}", label=repo_id),
        _tree(repo_id, revision, repo_type),
    )
    total = int((size.json() or {}).get("size") or 0)
    files = _files(tree)
    return await asyncio.to_thread(_estimate, repo_id, files, total, repo_type)


def _config_summary(raw: dict) -> dict:
    # Multimodal repos keep the language model's context length one level down.
    nested = raw.get("text_config") or {}
    return {
        "architectures": raw.get("architectures") or nested.get("architectures") or [],
        "model_type": raw.get("model_type") or "",
        "max_position_embeddings": (
            raw.get("max_position_embeddings") or nested.get("max_position_embeddings")
        ),
        "quantization_config": raw.get("quantization_config") or None,
        "torch_dtype": (
            raw.get("torch_dtype") or raw.get("dtype") or nested.get("torch_dtype") or ""
        ),
    }


def _gib(value: int) -> str:
    return f"{value / GIB:.1f}"


def _compatibility(detail: dict, available_bytes: int) -> dict:
    """One sentence about serving this repo with vLLM on this box."""
    if detail["is_adapter"]:
        base = detail.get("base_model") or "its base model"
        return {
            "level": "block",
            "note": f"LoRA adapter, not a servable model — serve {base} with --enable-lora "
                    "and attach this on top.",
        }
    if not detail["has_safetensors"]:
        kind = "GGUF" if detail["has_gguf"] else "PyTorch .bin"
        return {
            "level": "block",
            "note": f"{kind} weights only, no safetensors — the vLLM image on this box "
                    "will not load it.",
        }

    config = detail["config"]
    quant = config["quantization_config"] or {}
    parts = [
        (config["architectures"] or [""])[0],
        quant.get("quant_method") or quant.get("format") or config["torch_dtype"],
        f"{config['max_position_embeddings']:,} ctx" if config["max_position_embeddings"] else "",
    ]
    head = ", ".join(part for part in parts if part)
    if not head:
        # A gated repo's metadata is public but its config.json is not, so
        # there is genuinely nothing to check until the token is approved.
        head = "config.json is not readable yet" if detail["gated"] else "unknown architecture"
    weights = _gib(detail["total_bytes"])

    level = "ok"
    tail = f"{weights} GiB on disk"
    if available_bytes:
        tail += f" against {_gib(available_bytes)} GiB free in the unified pool"
        if detail["total_bytes"] > available_bytes:
            level = "warn"
            tail += " — stop a server before loading it"
    note = f"{head}; {tail}."
    if detail["gated"]:
        level = "warn"
        note += " Gated: the pull needs a token with approved access."
    if quant:
        note += " Confirm this vLLM build serves that quantisation on sm_121."
    return {"level": level, "note": note}


async def model_detail(repo_id: str, revision: str = "main") -> dict:
    check_repo_id(repo_id)
    info, tree, raw_config, memory = await asyncio.gather(
        _model_info(repo_id, revision),
        _tree(repo_id, revision),
        _raw_json(repo_id, revision, "config.json"),
        asyncio.to_thread(telemetry.read_meminfo),
    )
    files = _files(tree)
    paths = {entry["path"] for entry in files}
    detail = {
        "id": info.get("id") or repo_id,
        "revision": revision,
        "sha": info.get("sha") or "",
        "gated": info.get("gated", False),
        "private": bool(info.get("private")),
        "disabled": bool(info.get("disabled")),
        "tags": list(info.get("tags") or []),
        "downloads": int(info.get("downloads") or 0),
        "likes": int(info.get("likes") or 0),
        "updated": info.get("lastModified") or "",
        "pipeline_tag": info.get("pipeline_tag") or "",
        "library": info.get("library_name") or "",
        # dtype -> parameter count, absent when the repo has no safetensors.
        "parameters": (info.get("safetensors") or {}).get("parameters") or {},
        "files": [{"path": f["path"], "size": f["size"]} for f in files],
        "total_bytes": sum(f["size"] for f in files),
        "config": _config_summary(raw_config),
        "is_adapter": "adapter_config.json" in paths and "config.json" not in paths,
        "has_safetensors": any(p.endswith(".safetensors") for p in paths),
        "has_gguf": any(p.endswith(".gguf") for p in paths),
    }
    if detail["is_adapter"]:
        adapter = await _raw_json(repo_id, revision, "adapter_config.json")
        detail["base_model"] = adapter.get("base_model_name_or_path") or ""
        detail["peft_type"] = adapter.get("peft_type") or ""
    detail["estimate"] = await asyncio.to_thread(_estimate, repo_id, files, detail["total_bytes"])
    detail["compatibility"] = _compatibility(detail, memory.available_bytes)
    return detail


# --- the local cache ----------------------------------------------------

# The datasets-server exposes the parsed shape of a dataset — configs, splits,
# column names and a few real rows. That is the dataset equivalent of a model's
# architecture and context length: it is what decides whether a fine-tune can
# use the thing at all.
DATASETS_SERVER = "https://datasets-server.huggingface.co"

# Column names the fine-tuning worker knows how to read.
TEXT_COLUMNS = ("text", "content", "output", "completion", "response")
CHAT_COLUMNS = ("messages", "conversations", "conversation", "chat")


async def _datasets_server(path: str, params: dict) -> dict:
    """Best-effort: not every dataset is auto-converted, and that is not an error."""
    try:
        response = await _get(f"{DATASETS_SERVER}{path}", params, label=params.get("dataset", ""))
        return response.json() or {}
    except HubError:
        return {}


def _feature_type(value: Any) -> str:
    """A readable name for a datasets-server feature type.

    A scalar column is a dict — {"dtype": "string", "_type": "Value"} — but a
    conversational column is a *list* of struct definitions, which is exactly
    the shape that matters most here: `messages` is the column a chat fine-tune
    trains on. Assuming a dict broke every conversational dataset.
    """
    if isinstance(value, dict):
        return str(value.get("dtype") or value.get("_type") or "")
    if isinstance(value, list):
        inner = _feature_type(value[0]) if value else ""
        if not inner and value and isinstance(value[0], dict):
            inner = "struct{" + ", ".join(sorted(value[0])) + "}"
        return f"list<{inner}>" if inner else "list"
    return str(value or "")


def _training_shape(columns: list[dict]) -> dict:
    """Whether the fine-tuning worker could read this, and via which column.

    Order matters. A pre-rendered `text` column beats the instruction/output
    pair that usually sits beside it, but where there is no `text` the pair is
    the real content — picking `output` alone would train on answers with the
    questions thrown away.
    """
    names = {column["name"] for column in columns}

    chat = next((name for name in CHAT_COLUMNS if name in names), "")
    if chat:
        return {"level": "ok", "format": "messages", "field": chat,
                "note": f"conversational rows in '{chat}'; the tokenizer's chat template applies."}

    if "text" in names:
        return {"level": "ok", "format": "text", "field": "text",
                "note": "a pre-rendered 'text' column — train on it directly."}

    if {"instruction", "output"} <= names:
        return {"level": "warn", "format": "instruction", "field": "instruction",
                "note": ("instruction/output columns and no pre-rendered 'text'. Render the pair "
                         "into one field before training, or only the answers are learned.")}

    other = next((name for name in TEXT_COLUMNS if name in names), "")
    if other:
        return {"level": "warn", "format": "text", "field": other,
                "note": f"'{other}' looks like the content column, but check it is the whole row."}

    return {"level": "warn", "format": "", "field": "",
            "note": "no column this recognises; choose the text field by hand after pulling."}


async def dataset_detail(repo_id: str, revision: str = "main") -> dict:
    """A dataset the way the model page describes a model: what it is, what it
    costs, and whether it can actually be trained on."""
    check_repo_id(repo_id)
    info, tree, estimate, splits = await asyncio.gather(
        _model_info(repo_id, revision, "dataset"),
        _tree(repo_id, revision, "dataset"),
        estimate_download(repo_id, revision, "dataset"),
        _datasets_server("/splits", {"dataset": repo_id}),
    )

    entries = [
        {"config": item.get("config"), "split": item.get("split")}
        for item in (splits.get("splits") or [])
    ]
    columns: list[dict] = []
    rows: list[dict] = []
    if entries:
        first = entries[0]
        preview = await _datasets_server(
            "/first-rows",
            {"dataset": repo_id, "config": first["config"], "split": first["split"]},
        )
        columns = [
            {"name": f.get("name"), "type": _feature_type(f.get("type"))}
            for f in (preview.get("features") or [])
        ]
        rows = [item.get("row") or {} for item in (preview.get("rows") or [])][:3]

    return {
        "id": info.get("id") or repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "sha": info.get("sha", ""),
        "gated": info.get("gated", False),
        "private": info.get("private", False),
        "tags": info.get("tags", []),
        "downloads": info.get("downloads", 0),
        "likes": info.get("likes", 0),
        "updated": info.get("lastModified", ""),
        "files": _files(tree),
        "total_bytes": estimate["total_bytes"],
        "estimate": estimate,
        "configs": sorted({e["config"] for e in entries if e["config"]}),
        "splits": entries,
        "columns": columns,
        "sample_rows": rows,
        "training": _training_shape(columns),
    }


def _snapshot_kind(path: Path) -> str:
    """A snapshot is only servable if it carries a real config.json."""
    try:
        if (path / "config.json").is_file():
            return "model"
        if (path / "adapter_config.json").is_file():
            return "adapter"
    except OSError:
        return "unknown"
    return "other"


def _scan() -> dict:
    hub = settings.hf_cache / "hub"
    try:
        info = scan_cache_dir(hub)
    except Exception as exc:  # an unreadable or half-written cache is not a 500
        return {
            "ok": False,
            "cache_dir": str(hub),
            "error": f"could not read the HuggingFace cache at {hub}: {exc}",
            "size_on_disk": 0,
            "repos": [],
        }

    repos = []
    for repo in sorted(info.repos, key=lambda r: -r.size_on_disk):
        revisions = [
            {
                "sha": rev.commit_hash,
                "refs": sorted(rev.refs),
                "path": str(rev.snapshot_path),
                "size_on_disk": rev.size_on_disk,
                "nb_files": rev.nb_files,
                "last_modified": rev.last_modified,
                "kind": _snapshot_kind(rev.snapshot_path),
            }
            # Every collection scan_cache_dir hands back is a frozenset, so
            # without an explicit sort the UI order jitters between calls.
            for rev in sorted(repo.revisions, key=lambda r: -r.last_modified)
        ]
        repos.append(
            {
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "size_on_disk": repo.size_on_disk,
                "nb_files": repo.nb_files,
                "last_modified": repo.last_modified,
                "path": str(repo.repo_path),
                "refs": {name: rev.commit_hash for name, rev in repo.refs.items()},
                "revisions": revisions,
                "kind": revisions[0]["kind"] if revisions else "unknown",
            }
        )
    return {
        "ok": True,
        "cache_dir": str(hub),
        "size_on_disk": info.size_on_disk,
        "incomplete_bytes": info.incomplete_size_on_disk,
        "incomplete_files": len(info.incomplete_files),
        "warnings": [str(warning) for warning in info.warnings],
        "repos": repos,
    }


async def local_models() -> dict:
    return await asyncio.to_thread(_scan)


def cache_dir_for(repo_id: str, repo_type: str = "model") -> Path:
    """Resolve a repo id to its cache directory, or refuse.

    Whatever this returns is handed to `rm -rf` in a root container, so it has
    to be a real directory whose resolved parent is the hub root — a symlink or
    a traversal attempt must never survive this function.
    """
    check_repo_id(repo_id)
    _segment, prefix = _repo_type(repo_type)
    hub = (settings.hf_cache / "hub").resolve()
    name = f"{prefix}{repo_id.replace('/', '--')}"
    if not _CACHE_DIR.fullmatch(name):
        raise HubError(f"'{repo_id}' does not name a cache directory.", 400)
    directory = (hub / name).resolve()
    if directory.parent != hub or directory.name != name:
        raise HubError(f"'{repo_id}' does not resolve inside {hub}.", 400)
    if not directory.is_dir():
        raise HubError(f"'{repo_id}' is not in the local cache.", 404)
    return directory


def _node_host(node: str) -> str | None:
    """Where this cache operation runs. The worker container and the cache mount
    are identical on every node, so the only difference is which daemon."""
    from app import nodes

    return nodes.by_name(node).docker_host if node else None


async def delete_local(repo_id: str, repo_type: str = "model", node: str = "") -> dict:
    host = _node_host(node)
    if host:
        # The peer's cache is its own; check and remove it there.
        return await _delete_remote(repo_id, repo_type, node)
    directory = await asyncio.to_thread(cache_dir_for, repo_id, repo_type)
    freed = await asyncio.to_thread(_dir_size, directory)
    code, out, err = await docker_ctl.run_capture(
        image=settings.vllm_image,
        command=["-rf", f"{CONTAINER_CACHE}/hub/{directory.name}"],
        entrypoint="rm",
        mounts=[docker_ctl.Mount(settings.hf_cache, CONTAINER_CACHE)],
        gpu=False,
        network="none",
        timeout=600.0,
    )
    if code != 0:
        raise HubError(f"delete failed: {(err or out).strip()[-400:]}", 502)
    return {"deleted": repo_id, "freed_bytes": freed, "path": str(directory)}


def _dir_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


# --- download jobs ------------------------------------------------------

@jobs.register_parser("download")
def parse_download(line: str, progress: dict) -> dict | None:
    """Turn the worker's marker lines into the job's progress dict."""
    final = RESULT_MARKER in line
    marker = RESULT_MARKER if final else PROGRESS_MARKER
    if marker not in line:
        return None
    try:
        payload = json.loads(line.split(marker, 1)[1])
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    if final:
        # The worker's closing line names the snapshot it produced; that belongs
        # on the job's result, not buried in a progress reading.
        return {"phase": "done", "percent": 100.0, **payload, jobs.RESULT_KEY: payload}
    return payload


def _running_download(repo_id: str) -> str | None:
    for job in jobs.manager.list("download", limit=50):
        if job["status"] in jobs.TERMINAL:
            continue
        if ((job.get("spec") or {}).get("meta") or {}).get("repo_id") == repo_id:
            return job["id"]
    return None


async def submit_download(
    repo_id: str,
    *,
    revision: str = "main",
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    repo_type: str = "model",
    node: str = "",
) -> str:
    check_repo_id(repo_id)
    _repo_type(repo_type)
    existing = await asyncio.to_thread(_running_download, repo_id)
    if existing:
        # Two containers writing one repo's blobs interleave badly, and the
        # cache lock is inside the container, not across containers.
        raise HubError(f"'{repo_id}' is already downloading in job {existing}.", 409)

    command = ["-u", "/worker/hf_download.py", "--repo-id", repo_id, "--revision", revision,
               "--repo-type", repo_type]
    for pattern in allow_patterns or []:
        command += ["--allow", pattern]
    for pattern in ignore_patterns or []:
        command += ["--ignore", pattern]

    env = {
        "HF_HOME": CONTAINER_CACHE,
        # hf_transfer is a deprecated no-op now; Xet is what actually speeds
        # transfers up in huggingface_hub 1.x.
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    token = await stored_token()
    if token:
        env["HF_TOKEN"] = token

    from app import nodes as node_registry

    target = node_registry.by_name(node)
    # The worker is bind-mounted, and a bind mount resolves on the daemon's own
    # filesystem — so a peer needs its own copy of the script.
    worker_dir = WORKER_DIR if target.is_local else await node_registry.ensure_workers(target)

    spec = jobs.JobSpec(
        kind="download",
        title=f"Download {repo_id}" + (" (dataset)" if repo_type == "dataset" else ""),
        image=settings.vllm_image,
        command=command,
        env=env,
        mounts=[
            docker_ctl.Mount(settings.hf_cache, CONTAINER_CACHE),
            docker_ctl.Mount(worker_dir, "/worker", read_only=True),
        ],
        gpu=False,
        entrypoint="python",
        host=_node_host(node),
        meta={
            "repo_id": repo_id,
            "repo_type": repo_type,
            "node": node or "local",
            "revision": revision,
            "allow_patterns": allow_patterns or [],
            "ignore_patterns": ignore_patterns or [],
        },
    )
    return await jobs.manager.submit(spec)


async def _delete_remote(repo_id: str, repo_type: str, node_name: str) -> dict:
    """Remove a cached repo from a peer.

    The same guard as the local path — a validated directory name, never a
    user string — because this still ends in `rm -rf` run as root, just on
    another machine.
    """
    from app import nodes

    check_repo_id(repo_id)
    _segment, prefix = _repo_type(repo_type)
    name = f"{prefix}{repo_id.replace('/', '--')}"
    if not _CACHE_DIR.fullmatch(name):
        raise HubError(f"'{repo_id}' does not name a cache directory.", 400)

    node = nodes.by_name(node_name)
    if node.is_local:
        raise HubError("that is this machine; use the local delete", 400)

    hub = f"{settings.hf_cache}/hub"
    code, out = await nodes._ssh(
        node.name or node.address,
        f"test -d {hub}/{name} && du -sb {hub}/{name} | cut -f1 || echo missing",
    )
    if code != 0:
        raise HubError(f"could not reach {node.name}: {out.strip()[:200]}", 502)
    if "missing" in out:
        raise HubError(f"'{repo_id}' is not in {node.name}'s cache.", 404)
    freed = int(out.split()[0]) if out.split() and out.split()[0].isdigit() else 0

    code, out, err = await docker_ctl.run_capture(
        image=settings.vllm_image,
        command=["-rf", f"{CONTAINER_CACHE}/hub/{name}"],
        entrypoint="rm",
        mounts=[docker_ctl.Mount(settings.hf_cache, CONTAINER_CACHE)],
        gpu=False,
        network="none",
        timeout=600.0,
        host=node.docker_host,
    )
    if code != 0:
        raise HubError(f"delete on {node.name} failed: {(err or out).strip()[-400:]}", 502)
    return {"deleted": repo_id, "freed_bytes": freed, "node": node.name}


async def node_cache(node_name: str) -> dict:
    """What a peer holds, with sizes, so the Models page can manage it."""
    from app import nodes

    node = nodes.by_name(node_name)
    if node.is_local:
        return await local_models()

    hub = f"{settings.hf_cache}/hub"
    code, out = await nodes._ssh(
        node.name or node.address,
        f"cd {hub} 2>/dev/null && du -sb models--* datasets--* 2>/dev/null || true",
    )
    if code != 0:
        return {"ok": False, "node": node.name, "error": out.strip()[:300], "repos": []}

    repos = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        size, directory = int(parts[0]), parts[1].strip()
        kind, _, rest = directory.partition("--")
        repos.append({
            "repo_id": rest.replace("--", "/", 1),
            "repo_type": "dataset" if kind == "datasets" else "model",
            "size_on_disk": size,
            "directory": directory,
            "node": node.name,
        })
    repos.sort(key=lambda r: -r["size_on_disk"])
    return {
        "ok": True,
        "node": node.name,
        "cache_dir": hub,
        "size_on_disk": sum(r["size_on_disk"] for r in repos),
        "repos": repos,
    }
