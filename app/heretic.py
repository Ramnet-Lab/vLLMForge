"""Heretic — automated censorship removal.

A job is one long-lived container running the upstream `heretic` CLI headlessly:
an Optuna search over per-layer ablation weights, then a merged model written to
disk. Three facts about this box shape the integration.

The shared HF cache is root-owned, so the container runs as root in order to be
able to download a model into it — which means it also has to hand its outputs
back to the dashboard user before it exits, or nobody can delete their own
results. Heretic additionally writes model.safetensors mode 0600, which a vLLM
container running as another uid cannot read.

Heretic takes no memory-utilisation fraction the way vLLM does; it simply loads
the model with device_map="auto" and, on a unified-memory part, sees all 121 GiB
as VRAM. The only pre-flight available is an estimate against free host memory,
which is why `preflight` returns a `safety.Verdict` rather than calling
`safety.check_launch`.

Configuration reaches the container through a generated config.toml in the job
directory, which is the *lowest*-priority source in Heretic's settings stack.
Nothing else in the job sets those keys, and a user who wants to re-run a job by
hand can edit the file in place and re-launch the same directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, model_validator

from app import docker_ctl, events, hf, jobs, safety
from app.config import settings

# Container-side layout. The host job directory is bind-mounted at /job, which
# is also the working directory Heretic reads config.toml from.
JOB_PATH = "/job"
OUT_PATH = "/job/out"
CHECKPOINT_PATH = "/job/checkpoints"
NV_PATH = "/root/.nv"

# Shared across jobs: the first GB10 run JIT-compiles ~195 MB of PTX because the
# image's CUDA 13.3 torch runs under forward compatibility on a 580.x driver.
NV_CACHE_DIR = "_heretic_nvcache"

IMAGE_LABEL_REF = "org.llmd.heretic.ref"
DOCKERFILE = "heretic.Dockerfile"

# The two default prompt corpora, and how many rows each actually has. Heretic
# slices them with `split = "train[:N]"`, so asking for more than exists fails
# at load time rather than in the UI.
GOOD_DATASET = "mlabonne/harmless_alpaca"
BAD_DATASET = "mlabonne/harmful_behaviors"
MAX_DIRECTION_PROMPTS = 400
MAX_EVAL_PROMPTS = 100

# Upstream's rule of thumb, in GiB of resident memory per billion parameters.
GIB_PER_B = 2.5
GIB_PER_B_4BIT = 0.75
# Merging the LoRA back into the base model materialises a second copy of the
# weights; that moment, not the search, is the job's peak.
MERGE_GIB_PER_B = 2.0

HUB_TIMEOUT = 15.0


class HereticSettings(BaseModel):
    """The knobs the dashboard exposes, defaulted for a 121 GiB GB10."""

    model: str = Field(
        min_length=1,
        description="Hub id or a local model directory to abliterate.",
    )
    n_trials: int = Field(
        default=100,
        ge=1,
        le=2000,
        description="Optimisation trials. Upstream defaults to 200; each one is a full "
        "evaluation pass, so this is the main quality/wall-clock trade.",
    )
    n_startup_trials: int = Field(
        default=30,
        ge=1,
        le=2000,
        description="Random trials before the sampler starts steering. Must not exceed n_trials.",
    )
    batch_size: int = Field(
        default=0,
        ge=0,
        le=1024,
        description="Generation batch size. 0 auto-tunes with a throughput ladder at startup; "
        "pin a known-good value to skip it.",
    )
    max_batch_size: int = Field(
        default=64,
        ge=1,
        le=1024,
        description="Ceiling for the auto-tune ladder. 64 was the value GB10 chose unaided.",
    )
    quantization: Literal["none", "bnb_4bit"] = Field(
        default="none",
        description="bnb_4bit cuts memory by roughly 70% but reloads the base model on CPU at "
        "merge time, and its kernels are untested on this GB10.",
    )
    export_strategy: Literal["merge", "adapter"] = Field(
        default="merge",
        description="merge writes a full standalone model directory; adapter writes just the "
        "LoRA, which vLLM has to be told to load on top of the base model.",
    )
    trial_index: int = Field(
        default=0,
        ge=0,
        description="Which point on the sorted Pareto front to export. 0 is the best trade-off "
        "Heretic found between refusal rate and KL divergence.",
    )
    checkpoint_action: Literal["continue", "restart"] = Field(
        default="continue",
        description="What to do when this model already has a study checkpoint. restart deletes "
        "every completed trial from the previous run.",
    )
    seed: int = Field(
        default=12345,
        ge=0,
        description="Sampler seed. Fixed by default so a repeated run reproduces the same trials.",
    )
    max_shard_size: str = Field(
        default="5GB",
        pattern=r"^\d+(\.\d+)?(KB|MB|GB|TB)?$",
        description="Largest individual safetensors file in the exported model.",
    )
    direction_prompts: int = Field(
        default=MAX_DIRECTION_PROMPTS,
        ge=8,
        le=MAX_DIRECTION_PROMPTS,
        description="Harmless/harmful prompt pairs used to compute the refusal direction. "
        "Fewer is faster and noisier.",
    )
    eval_prompts: int = Field(
        default=MAX_EVAL_PROMPTS,
        ge=8,
        le=MAX_EVAL_PROMPTS,
        description="Held-out prompts scored after every trial. This count multiplies directly "
        "into per-trial time.",
    )

    @model_validator(mode="after")
    def _startup_fits(self) -> HereticSettings:
        if self.n_startup_trials > self.n_trials:
            raise ValueError("n_startup_trials cannot exceed n_trials")
        return self


def defaults() -> HereticSettings:
    return HereticSettings(model="Qwen/Qwen2.5-0.5B-Instruct")


def ui_model() -> dict[str, Any]:
    """The settings surface as the form generator wants it: ordered, with help."""
    schema = HereticSettings.model_json_schema()
    fields = []
    for name, spec in schema["properties"].items():
        entry: dict[str, Any] = {
            "name": name,
            "help": spec.get("description", ""),
            "default": spec.get("default"),
            "required": name in schema.get("required", []),
            "type": spec.get("type", "string"),
        }
        # Literal[...] lands in the schema as an enum, sometimes behind an
        # allOf/$ref when it also carries a default.
        options = spec.get("enum") or next(
            (sub["enum"] for sub in spec.get("allOf", []) if "enum" in sub), None
        )
        if options:
            entry["options"] = list(options)
            entry["type"] = "enum"
        for bound in ("minimum", "maximum", "pattern"):
            if bound in spec:
                entry[bound] = spec[bound]
        fields.append(entry)
    return {
        "fields": fields,
        "defaults": defaults().model_dump(),
        "datasets": {"good": GOOD_DATASET, "bad": BAD_DATASET},
        "image": settings.heretic_image,
    }


# --- config.toml generation ---------------------------------------------

def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def config_toml(cfg: HereticSettings) -> str:
    """The config.toml written into a job directory.

    Prompt counts have to go through TOML tables: on master the scorer prompt
    sets moved into the plugin system and have no CLI or environment form at all.
    """
    lines = [
        "# Generated by llm-dashboard. Editing this file and re-launching the same",
        "# job directory resumes the study from its checkpoint.",
        f"model = {_toml_str(cfg.model)}",
        f"seed = {cfg.seed}",
        f"batch_size = {cfg.batch_size}",
        f"max_batch_size = {cfg.max_batch_size}",
        f"quantization = {_toml_str(cfg.quantization)}",
        f"n_trials = {cfg.n_trials}",
        f"n_startup_trials = {cfg.n_startup_trials}",
        f"study_checkpoint_dir = {_toml_str(CHECKPOINT_PATH)}",
        f"max_shard_size = {_toml_str(cfg.max_shard_size)}",
        "",
        "# Without all five of these Heretic opens a questionary menu after the",
        "# search and waits forever on a stdin that is not a terminal.",
        f"checkpoint_action = {_toml_str(cfg.checkpoint_action)}",
        f"trial_index = {cfg.trial_index}",
        'model_action = "save"',
        f"save_directory = {_toml_str(OUT_PATH)}",
        f"export_strategy = {_toml_str(cfg.export_strategy)}",
        "",
        "[good_prompts]",
        f"dataset = {_toml_str(GOOD_DATASET)}",
        f'split = "train[:{cfg.direction_prompts}]"',
        'column = "text"',
        "",
        "[bad_prompts]",
        f"dataset = {_toml_str(BAD_DATASET)}",
        f'split = "train[:{cfg.direction_prompts}]"',
        'column = "text"',
        "",
        "[scorer.KeywordRate.prompts]",
        f"dataset = {_toml_str(BAD_DATASET)}",
        f'split = "test[:{cfg.eval_prompts}]"',
        'column = "text"',
        "",
        "[scorer.KLDivergence.prompts]",
        f"dataset = {_toml_str(GOOD_DATASET)}",
        f'split = "test[:{cfg.eval_prompts}]"',
        'column = "text"',
        "",
    ]
    return "\n".join(lines)


def checkpoint_filename(model: str) -> str:
    """Optuna's journal file, named the way main.py names it."""
    stem = "".join(c if c.isalnum() or c in ("_", "-") else "--" for c in model)
    return f"{stem}.jsonl"


# --- job construction ----------------------------------------------------

def build_job(cfg: HereticSettings) -> jobs.JobSpec:
    run_id = uuid.uuid4().hex[:12]
    job_dir = jobs.output_dir(run_id, f"heretic-{cfg.model}")
    (job_dir / "checkpoints").mkdir(exist_ok=True)
    (job_dir / "out").mkdir(exist_ok=True)
    (job_dir / "config.toml").write_text(config_toml(cfg), encoding="utf-8")

    nv_cache = settings.output_dir / NV_CACHE_DIR
    nv_cache.mkdir(parents=True, exist_ok=True)

    env = {"HF_HOME": "/hf"}
    hf_token = hf.token()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    return jobs.JobSpec(
        kind="heretic",
        title=f"Heretic · {cfg.model}",
        image=settings.heretic_image,
        entrypoint="/bin/bash",
        command=["-c", _container_script(cfg)],
        env=env,
        mounts=[
            docker_ctl.Mount(settings.hf_cache, "/hf"),
            docker_ctl.Mount(job_dir, JOB_PATH),
            docker_ctl.Mount(nv_cache, NV_PATH),
        ],
        gpu=True,
        workdir=JOB_PATH,
        meta={
            "run_id": run_id,
            "model": cfg.model,
            "job_dir": str(job_dir),
            "output_dir": str(job_dir / "out"),
            "config_path": str(job_dir / "config.toml"),
            "checkpoint": str(job_dir / "checkpoints" / checkpoint_filename(cfg.model)),
            "settings": cfg.model_dump(),
        },
    )


def _container_script(cfg: HereticSettings) -> str:
    """Run Heretic as root, then give the results back to the dashboard user.

    Root is not a choice: the shared HF cache is root-owned, so anything else
    cannot download a model it does not already have. The exit code has to
    survive the clean-up, otherwise every job looks like it succeeded.
    """
    uid, gid = os.getuid(), os.getgid()
    return "; ".join(
        [
            f"heretic --model {shlex.quote(cfg.model)}",
            "rc=$?",
            # 0600 is Heretic's own umask for the weights, and a vLLM container
            # running as another uid cannot read that.
            f"find {OUT_PATH} -name '*.safetensors' -exec chmod 0644 {{}} + 2>/dev/null",
            f"chown -R {uid}:{gid} {JOB_PATH} {NV_PATH} 2>/dev/null",
            "exit $rc",
        ]
    )


# --- progress parsing ----------------------------------------------------

_LOADING = re.compile(r"^Loading model (.+)\.\.\.$")
_BATCH_TUNE = re.compile(r"^Determining optimal batch size\.\.\.$")
_BATCH_TRY = re.compile(r"^\* Trying batch size (\d+)\.\.\. (.+)$")
_BATCH_CHOSEN = re.compile(r"^\* Chosen batch size: (\d+)$")
_DIRECTIONS = re.compile(r"^Calculating per-layer residual directions\.\.\.$")
_BASELINE_RATE = re.compile(r"^\*\s+Baseline (?P<name>Keywords|Refusals): (\d+)/(\d+)$")
_BASELINE_KL = re.compile(r"^\*\s+Baseline KL divergence: (\S+)")
_TRIAL = re.compile(r"^Running trial (\d+) of (\d+)\.\.\.$")
_RATE = re.compile(r"^\s+\* (?:Keywords|Refusals): (\d+)/(\d+)$")
_KL = re.compile(r"^\s+\* KL divergence: ([\d.eE+-]+)$")
_PARAM = re.compile(r"^\s+\* ([a-z_][\w.]*) = (.+)$")
_ELAPSED = re.compile(r"^Elapsed time: (.+)$")
_ETA = re.compile(r"^Estimated remaining time: (.+)$")
_ALLOCATED = re.compile(r"^Allocated GPU VRAM: ([\d.]+) GB$")
_RESERVED = re.compile(r"^Reserved GPU VRAM: ([\d.]+) GB$")
_RSS = re.compile(r"^Resident system RAM: ([\d.]+) GB$")
_FINISHED = re.compile(r"^Optimization finished!$")
_RESTORING = re.compile(r"^Restoring model from trial (\d+)\.\.\.$")
_SAVING = re.compile(r"^Saving (merged model|adapter)\.\.\.$")
_SAVED = re.compile(r"^Model saved to (.+)\.$")


@jobs.register_parser("heretic")
def parse_progress(line: str, progress: dict) -> dict | None:
    """Turn Heretic's rich-console log into structured progress.

    The log is a human-readable contract, not an API — the metric line already
    changed from "* Refusals: n/100" to a "* Metrics:" block between 1.4.0 and
    master — so everything here is optional and nothing raises. The Optuna
    journal under checkpoints/ is the authoritative record of the trials.
    """
    text = jobs.split_carriage(line).rstrip()

    if match := _LOADING.match(text):
        return {"phase": "loading-model", "model": match.group(1)}
    if _BATCH_TUNE.match(text):
        return {"phase": "batch-size"}
    if match := _BATCH_TRY.match(text):
        return {"phase": "batch-size", "batch_probe": f"{match.group(1)}: {match.group(2)}"}
    if match := _BATCH_CHOSEN.match(text):
        return {"phase": "batch-size", "batch_size": int(match.group(1))}
    if _DIRECTIONS.match(text):
        return {"phase": "directions"}

    if match := _BASELINE_RATE.match(text):
        baseline = dict(progress.get("baseline") or {})
        baseline.update(
            {
                "metric": match.group("name").lower(),
                "refusals": int(match.group(2)),
                "total": int(match.group(3)),
            }
        )
        return {"phase": "baseline", "baseline": baseline}
    if match := _BASELINE_KL.match(text):
        baseline = dict(progress.get("baseline") or {})
        baseline["kl_divergence"] = _number(match.group(1))
        return {"baseline": baseline}

    if match := _TRIAL.match(text):
        index, total = int(match.group(1)), int(match.group(2))
        trials = list(progress.get("trials") or [])
        trials.append({"trial": index, "params": {}})
        return {
            "phase": "trials",
            "step": index,
            "total": total,
            "percent": round(100.0 * (index - 1) / total, 1),
            "trials": trials,
            "restoring": False,
        }

    if match := _PARAM.match(text):
        return _record_param(progress, match.group(1), match.group(2).strip())
    if match := _RATE.match(text):
        return _record_metric(progress, refusals=int(match.group(1)), total=int(match.group(2)))
    if match := _KL.match(text):
        return _record_metric(progress, kl_divergence=_number(match.group(1)))

    if match := _ELAPSED.match(text):
        # An elapsed line closes a trial, so the bar can advance a whole step.
        step, total = progress.get("step"), progress.get("total")
        update: dict[str, Any] = {"elapsed": match.group(1)}
        if step and total:
            update["percent"] = round(100.0 * step / total, 1)
        return update
    if match := _ETA.match(text):
        return {"eta": match.group(1)}
    if match := _ALLOCATED.match(text):
        return {"allocated_gb": _number(match.group(1))}
    if match := _RESERVED.match(text):
        return {"reserved_gb": _number(match.group(1))}
    if match := _RSS.match(text):
        return {"rss_gb": _number(match.group(1))}

    if _FINISHED.match(text):
        return {"phase": "optimized", "percent": 100.0, "eta": ""}
    if match := _RESTORING.match(text):
        return {"phase": "restoring", "chosen_trial": int(match.group(1)), "restoring": True}
    if match := _SAVING.match(text):
        return {"phase": "saving", "export": match.group(1)}
    if match := _SAVED.match(text):
        return {"phase": "saved", "percent": 100.0, "saved_to": match.group(1)}
    return None


def _number(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def _record_param(progress: dict, name: str, value: str) -> dict | None:
    """Ablation parameters print both per trial and again for the chosen one."""
    parsed: float | str = value if _number(value) is None else float(value)
    if progress.get("restoring"):
        chosen = dict(progress.get("chosen_params") or {})
        chosen[name] = parsed
        return {"chosen_params": chosen}
    trials = list(progress.get("trials") or [])
    if not trials:
        return None
    trials[-1] = {**trials[-1], "params": {**trials[-1].get("params", {}), name: parsed}}
    return {"trials": trials}


def _record_metric(progress: dict, **values: Any) -> dict | None:
    trials = list(progress.get("trials") or [])
    if not trials:
        return None
    trials[-1] = {**trials[-1], **values}
    return {"trials": trials, "last_metrics": trials[-1]}


# --- memory pre-flight ---------------------------------------------------

_SIZE_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z0-9])")


async def inspect_model(model: str) -> dict[str, Any]:
    """Parameter count and trust_remote_code status, best effort.

    Models needing trust_remote_code cannot be abliterated at all: transformers
    asks on stdin, and a detached container has no terminal to answer with.
    """
    local = Path(model)
    if local.is_dir():
        return await asyncio.to_thread(_inspect_local, local)

    info: dict[str, Any] = {"source": "", "params": None, "trust_remote_code": False, "error": ""}
    headers = {"Authorization": f"Bearer {settings.hf_token}"} if settings.hf_token else {}
    try:
        async with httpx.AsyncClient(timeout=HUB_TIMEOUT, headers=headers) as client:
            meta, config = await asyncio.gather(
                client.get(f"https://huggingface.co/api/models/{model}"),
                client.get(f"https://huggingface.co/{model}/resolve/main/config.json"),
                return_exceptions=True,
            )
        if isinstance(meta, httpx.Response) and meta.status_code == 200:
            total = (meta.json().get("safetensors") or {}).get("total")
            if total:
                info.update(params=int(total), source="hub safetensors metadata")
        if isinstance(config, httpx.Response) and config.status_code == 200:
            info["trust_remote_code"] = "auto_map" in config.json()
    except (httpx.HTTPError, ValueError, OSError) as exc:
        info["error"] = f"could not reach the Hub: {exc}"

    if info["params"] is None:
        match = _SIZE_IN_NAME.search(model)
        if match:
            info.update(params=int(float(match.group(1)) * 1e9), source="model name")
    return info


def _inspect_local(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"source": "", "params": None, "trust_remote_code": False, "error": ""}
    weights = sum(f.stat().st_size for f in path.glob("**/*.safetensors"))
    if weights:
        # No dtype information without opening the shards; 16-bit is the only
        # sane guess and errs towards over-counting for fp32 checkpoints.
        info.update(params=weights // 2, source="local safetensors size")
    config = path / "config.json"
    if config.is_file():
        with contextlib.suppress(ValueError, OSError):
            info["trust_remote_code"] = "auto_map" in json.loads(config.read_text())
    else:
        info["error"] = f"{path} has no config.json — Heretic will not be able to load it"
    return info


def estimate_bytes(params: int, cfg: HereticSettings) -> dict[str, int]:
    billions = params / 1e9
    per_b = GIB_PER_B_4BIT if cfg.quantization == "bnb_4bit" else GIB_PER_B
    load = int(billions * per_b * safety.GIB)
    # bnb_4bit dequantises the base model into system RAM to merge, so the spike
    # is against the unquantised size either way.
    merge = int(billions * MERGE_GIB_PER_B * safety.GIB) if cfg.export_strategy == "merge" else 0
    return {"load_bytes": load, "merge_bytes": merge, "peak_bytes": load + merge}


async def preflight(cfg: HereticSettings) -> dict[str, Any]:
    """Would this job fit in what is left of the host right now?"""
    info = await inspect_model(cfg.model)
    budget = await safety.current_budget()
    payload = budget.as_dict()
    notes: list[str] = []
    if info["error"]:
        notes.append(info["error"])
    if info["trust_remote_code"]:
        notes.append(
            "This model sets auto_map, so transformers demands trust_remote_code. Heretic asks "
            "for that on stdin, which a detached container cannot answer — the job will fail."
        )
    if cfg.quantization == "bnb_4bit":
        notes.append(
            "bitsandbytes ships aarch64 wheels but its kernels are unverified on this GB10; "
            "quantization=none is the mode that has actually run here."
        )

    if not info["params"]:
        return {
            **safety.Verdict(
                ok=True,
                level="warn",
                message=(
                    f"Cannot size {cfg.model}: no parameter count from the Hub, the local "
                    "directory or the model name. Heretic will load it with device_map=auto and "
                    f"take whatever it needs out of the {_gib(budget.available_bytes)} free."
                ),
                budget=payload,
            ).as_dict(),
            "estimate": None,
            "model": info,
            "notes": notes,
        }

    estimate = estimate_bytes(info["params"], cfg)
    peak = estimate["peak_bytes"]
    billions = info["params"] / 1e9
    headroom = budget.available_bytes - peak
    util = peak / budget.total_bytes if budget.total_bytes else 0.0
    tenants = ", ".join(f"{t.name}={t.util:g}" for t in budget.tenants) or "none"
    sized = (
        f"{cfg.model} is ~{billions:.1f}B parameters, so expect ~{_gib(estimate['load_bytes'])} "
        f"resident during the search and ~{_gib(peak)} at merge time"
    )

    if peak > budget.available_bytes:
        verdict = safety.Verdict(
            ok=False,
            level="block",
            message=(
                f"{sized}, but only {_gib(budget.available_bytes)} is free right now "
                f"(vLLM tenants: {tenants}). Stop something, drop export_strategy to adapter, "
                "or switch quantization to bnb_4bit."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=peak,
        )
    elif headroom < budget.reserve_bytes:
        verdict = safety.Verdict(
            ok=True,
            level="warn",
            message=(
                f"{sized}, leaving only {_gib(headroom)} for the OS. Heretic does not cap itself "
                "the way vLLM does, and the merge spike is not gradual — this may take the box "
                "down. Serialise it against the running servers."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=peak,
        )
    else:
        verdict = safety.Verdict(
            ok=True,
            level="ok",
            message=f"{sized}, leaving {_gib(headroom)} free (vLLM tenants: {tenants}).",
            budget=payload,
            requested_util=util,
            requested_bytes=peak,
        )
    return {**verdict.as_dict(), "estimate": estimate, "model": info, "notes": notes}


def _gib(value: float) -> str:
    return f"{value / safety.GIB:.1f} GiB"


# --- image ---------------------------------------------------------------

async def image_status() -> dict[str, Any]:
    """Is the image built, and from which Heretic commit?"""
    tag = settings.heretic_image
    info = await docker_ctl.inspect(tag)
    if info is None:
        return {"image": tag, "present": False, "ref": "", "commit": "", "created": ""}
    labels = (info.get("Config", {}) or {}).get("Labels") or {}
    return {
        "image": tag,
        "present": True,
        "id": info.get("Id", "")[:19],
        "created": info.get("Created", ""),
        "ref": labels.get(IMAGE_LABEL_REF, ""),
        "commit": await _installed_commit(tag),
    }


_commit_cache: dict[str, str] = {}


async def _installed_commit(tag: str) -> str:
    """The commit pip actually resolved, from the PEP 610 record in the image."""
    if tag in _commit_cache:
        return _commit_cache[tag]
    code, out, _ = await docker_ctl.run_capture(
        image=tag,
        command=["/opt/heretic/build-ref.json"],
        entrypoint="cat",
        gpu=False,
        network="none",
        timeout=30.0,
    )
    commit = ""
    if code == 0:
        try:
            commit = ((json.loads(out) or {}).get("vcs_info") or {}).get("commit_id", "")
        except ValueError:
            commit = ""
    _commit_cache[tag] = commit
    return commit


async def submit_build(ref: str = "master") -> str:
    """Build the image as a job so the UI can stream it like any other run.

    `docker build` is not a container run, so the job manager cannot drive it;
    this owns the log and the status transitions instead.
    """
    spec = jobs.JobSpec(
        kind="build",
        title=f"Build {settings.heretic_image} (heretic@{ref})",
        image=settings.heretic_image,
        command=[],
        env={},
        mounts=[],
        gpu=False,
        meta={"dockerfile": DOCKERFILE, "heretic_ref": ref},
    )
    job_id = await asyncio.to_thread(jobs.manager.create, spec)
    jobs.manager.adopt(job_id, _drive_build(job_id, ref))
    return job_id


async def _drive_build(job_id: str, ref: str) -> None:
    tag = settings.heretic_image
    dockerfile = settings.docker_dir / DOCKERFILE
    jobs.manager.begin(job_id)
    jobs.manager.log(
        job_id, f"$ docker build -t {tag} -f {dockerfile} --build-arg HERETIC_REF={ref}"
    )
    try:
        async for line in docker_ctl.build_image(
            tag, dockerfile, settings.docker_dir, build_args={"HERETIC_REF": ref}
        ):
            if line.strip():
                jobs.manager.log(job_id, line)
    except docker_ctl.DockerError as exc:
        jobs.manager.finish(job_id, ok=False, error=str(exc))
        return
    except Exception as exc:  # a build failure must land on the job, not the event loop
        jobs.manager.finish(job_id, ok=False, error=repr(exc))
        return
    _commit_cache.pop(tag, None)
    jobs.manager.finish(job_id, ok=True)
    await events.broker.publish(events.JOBS, {"type": "image", "image": tag})
