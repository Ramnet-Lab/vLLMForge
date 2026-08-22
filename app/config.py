"""Runtime configuration.

Everything here is overridable through the environment so the dashboard can be
pointed at a different box, image or cache without editing code. Defaults are
tuned for the DGX Spark this was written on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- web server -------------------------------------------------------
    host: str = field(default_factory=lambda: _env("LLMD_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("LLMD_PORT", 8700))

    # --- storage ----------------------------------------------------------
    state_dir: Path = field(
        default_factory=lambda: _env_path("LLMD_STATE_DIR", "~/.local/share/llm-dashboard")
    )
    hf_cache: Path = field(
        default_factory=lambda: _env_path("LLMD_HF_CACHE", "~/models/hf-cache")
    )
    output_dir: Path = field(
        default_factory=lambda: _env_path("LLMD_OUTPUT_DIR", "~/models/outputs")
    )
    dataset_dir: Path = field(
        default_factory=lambda: _env_path("LLMD_DATASET_DIR", "~/models/datasets")
    )

    # --- container images -------------------------------------------------
    vllm_image: str = field(
        default_factory=lambda: _env("LLMD_VLLM_IMAGE", "nvcr.io/nvidia/vllm:26.07-py3")
    )
    heretic_image: str = field(
        default_factory=lambda: _env("LLMD_HERETIC_IMAGE", "llmd/heretic:latest")
    )
    finetune_image: str = field(
        default_factory=lambda: _env("LLMD_FINETUNE_IMAGE", "llmd/finetune:latest")
    )
    # vLLM plus ray, for an engine pooled across machines. The NGC image has no
    # ray, so multi-node pipeline parallelism needs this one.
    ray_image: str = field(
        default_factory=lambda: _env("LLMD_RAY_IMAGE", "llmd/vllm-ray:latest")
    )
    container_prefix: str = field(default_factory=lambda: _env("LLMD_CONTAINER_PREFIX", "llmd-"))

    # --- credentials ------------------------------------------------------
    hf_token: str = field(
        default_factory=lambda: _env("LLMD_HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    )

    # --- host memory safety ----------------------------------------------
    # GPU memory IS host memory on GB10. A gpu-memory-utilization that looks
    # innocuous can take the whole machine down during CUDA graph capture, so
    # every launch is checked against these reserves. See docs/MEMORY.md.
    mem_reserve_gib: float = field(default_factory=lambda: _env_float("LLMD_MEM_RESERVE_GIB", 32.0))
    mem_warn_reserve_gib: float = field(
        default_factory=lambda: _env_float("LLMD_MEM_WARN_RESERVE_GIB", 38.0)
    )
    # The watchdog kills vLLM containers before the kernel OOM killer gets a
    # chance to freeze an interactive desktop.
    memguard_threshold_mib: int = field(
        default_factory=lambda: _env_int("LLMD_MEMGUARD_THRESHOLD_MIB", 10240)
    )
    memguard_enabled: bool = field(
        default_factory=lambda: _env("LLMD_MEMGUARD_ENABLED", "1") not in ("0", "false", "no")
    )

    # --- cluster fabric (NCCL over RoCE) ----------------------------------
    roce_interface: str = field(default_factory=lambda: _env("LLMD_ROCE_IF", "enp1s0f0np0"))
    roce_hca: str = field(default_factory=lambda: _env("LLMD_ROCE_HCA", "rocep1s0f0"))

    # --- polling ----------------------------------------------------------
    telemetry_interval: float = field(
        default_factory=lambda: _env_float("LLMD_TELEMETRY_INTERVAL", 2.0)
    )

    @property
    def db_path(self) -> Path:
        return self.state_dir / "dashboard.db"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def upload_dir(self) -> Path:
        return self.state_dir / "uploads"

    @property
    def web_dir(self) -> Path:
        return REPO_ROOT / "web"

    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / "app" / "data"

    @property
    def docker_dir(self) -> Path:
        return REPO_ROOT / "docker"

    def container_name(self, kind: str, ident: str | int) -> str:
        return f"{self.container_prefix}{kind}-{ident}"

    def nccl_env(self) -> dict[str, str]:
        """NCCL/Gloo settings matching this cluster's RoCE fabric."""
        return {
            "NCCL_SOCKET_IFNAME": self.roce_interface,
            "GLOO_SOCKET_IFNAME": self.roce_interface,
            "NCCL_IB_HCA": self.roce_hca,
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_GID_INDEX": "3",
        }

    def ensure_dirs(self) -> None:
        for path in (
            self.state_dir, self.log_dir, self.upload_dir, self.output_dir, self.dataset_dir
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
