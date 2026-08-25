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
    # The NGC image plus the xgrammar its own vLLM imports; docker/vllm.Dockerfile
    # says why. Set LLMD_VLLM_IMAGE back to nvcr.io/nvidia/vllm:26.07-py3 to run
    # the base image, and accept that every request carrying tools 500s.
    vllm_image: str = field(
        default_factory=lambda: _env("LLMD_VLLM_IMAGE", "llmd/vllm:latest")
    )
    # "auto" asks the hardware (app/images.py). Set a concrete image to pin it;
    # the old behaviour is LLMD_VLLM_BASE_IMAGE=nvcr.io/nvidia/vllm:26.07-py3.
    vllm_base_image: str = field(
        default_factory=lambda: _env("LLMD_VLLM_BASE_IMAGE", "auto")
    )
    heretic_image: str = field(
        default_factory=lambda: _env("LLMD_HERETIC_IMAGE", "llmd/heretic:latest")
    )
    finetune_image: str = field(
        default_factory=lambda: _env("LLMD_FINETUNE_IMAGE", "llmd/finetune:latest")
    )
    # The second serving engine. Built from docker/llamacpp.Dockerfile rather
    # than pulled, for the same reason the vLLM image is: the upstream server
    # image puts the binary in its ENTRYPOINT, which would swallow the argv the
    # dashboard passes as the container command.
    llamacpp_image: str = field(
        default_factory=lambda: _env("LLMD_LLAMACPP_IMAGE", "llmd/llamacpp:latest")
    )
    # What docker/llamacpp.Dockerfile builds on. Separate from vllm_base_image
    # because it is a different upstream entirely — a llama.cpp server build, not
    # an NGC vLLM image — and choosing it by accelerator kind would be wrong.
    llamacpp_base_image: str = field(
        default_factory=lambda: _env("LLMD_LLAMACPP_BASE_IMAGE",
                                     "ghcr.io/ggml-org/llama.cpp:server-cuda")
    )
    # A container to run root filesystem work in: deleting a repo out of the
    # root-owned cache, and running the HuggingFace download worker. The vLLM
    # image has always been doing this second job, which quietly makes a
    # multi-gigabyte serving image a prerequisite for downloading anything — on
    # every node. Empty keeps exactly that behaviour; set it to something small
    # on a box that does not serve with vLLM.
    utility_image: str = field(default_factory=lambda: _env("LLMD_UTILITY_IMAGE", ""))
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
    # On a discrete GPU the watchdog's trigger — host MemAvailable — has nothing
    # to do with the memory the engines are holding, and the kernel OOM killer
    # is a working backstop there because the desktop is not in the framebuffer.
    # `auto` warns on discrete and kills on unified; `kill` restores the old
    # behaviour everywhere.
    memguard_host_action: str = field(
        default_factory=lambda: _env("LLMD_MEMGUARD_HOST_ACTION", "auto")
    )
    # The same watchdog, on the memory that actually runs out on a discrete box.
    # Host MemAvailable can sit at 90 GiB while every framebuffer is full, so on
    # discrete the trigger is device free memory and a kill there is not a
    # self-inflicted outage — it is the only signal that saw the problem. Lower
    # than the host threshold because a framebuffer is smaller than host RAM and
    # nothing but engines lives in it.
    memguard_device_threshold_mib: int = field(
        default_factory=lambda: _env_int("LLMD_MEMGUARD_DEVICE_THRESHOLD_MIB", 2048)
    )

    # --- accelerator memory -----------------------------------------------
    # Empty means detect; see app/accel.py. Set to unified/discrete/none only to
    # override a detection that is wrong, and understand which way is safe: a
    # machine wrongly called discrete can be talked into claiming memory the OS
    # is living in.
    accel_mode: str = field(default_factory=lambda: _env("LLMD_ACCEL_MODE", ""))

    # --- what the Playground may connect to -------------------------------
    # The dashboard will proxy a chat request to a URL the operator types, so
    # something has to stop it being pointed at the public internet. Private
    # address space is the honest expression of "engines on my own network":
    # it covers any LAN and any cluster fabric without naming one, where a list
    # of this operator's own subnets both leaked their layout and silently
    # blocked the feature for everybody else. Comma-separated CIDRs to override.
    allowed_networks: tuple[str, ...] = field(default_factory=lambda: tuple(
        n.strip() for n in _env(
            "LLMD_ALLOWED_NETWORKS",
            "127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16",
        ).split(",") if n.strip()
    ))
    # What to hold back on a machine whose GPU has its own memory. Small,
    # because the OS is not a tenant of a framebuffer — this is the driver's
    # own overhead and allocator slack, not room for a desktop.
    gpu_reserve_gib: float = field(default_factory=lambda: _env_float("LLMD_GPU_RESERVE_GIB", 2.0))
    # Host RAM held back on a discrete machine, where it is a separate pool that
    # only --cpu-offload-gb and the loading process draw on.
    host_reserve_gib: float = field(
        default_factory=lambda: _env_float("LLMD_HOST_RESERVE_GIB", 8.0)
    )

    # --- cluster fabric ---------------------------------------------------
    # Both default to empty, and empty means "work it out from this machine".
    # A NIC name that does not exist on the box is worse than no name at all:
    # NCCL refuses to initialise on an unknown NCCL_SOCKET_IFNAME even at world
    # size 1 (DistBackendError, invalid usage), so naming one machine's NIC here
    # made every launch fail on every other machine. nodes.cluster_interface()
    # finds the real one; set this only to override that.
    roce_interface: str = field(default_factory=lambda: _env("LLMD_ROCE_IF", ""))
    # RDMA is opt-in for the same reason — NCCL_IB_HCA naming an absent device
    # is the same mistake, and the pooled path disables IB anyway.
    roce_hca: str = field(default_factory=lambda: _env("LLMD_ROCE_HCA", ""))

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
    def compile_cache(self) -> Path:
        """Where an engine's compiled kernels survive their container.

        vLLM writes torch.compile output, Inductor's autotune results and
        Triton's kernel cache into three directories inside the container. The
        launcher removes the container before every start, so all three were
        discarded every time and each launch recompiled from cold — minutes on
        an ordinary model, and on a model whose sampling step is separately
        compiled it is the difference between starting and appearing to hang.
        """
        return self.state_dir / "compile-cache"

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

    def fabric_env(self) -> dict[str, str]:
        """RDMA settings, for a container that talks to another machine.

        Only what the operator has explicitly configured. The interface names
        are NOT here: they are per-node and detected at launch, because the NIC
        carrying the cluster subnet has a different name on each box — and a
        single-machine engine gets none of this at all, since it has no peer to
        reach and an absent interface name breaks NCCL outright.
        """
        env: dict[str, str] = {}
        if self.roce_hca:
            env["NCCL_IB_HCA"] = self.roce_hca
            env["NCCL_IB_GID_INDEX"] = "3"
        return env

    def ensure_dirs(self) -> None:
        for path in (
            self.state_dir, self.log_dir, self.upload_dir, self.output_dir, self.dataset_dir,
            self.compile_cache,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
