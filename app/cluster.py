"""One vLLM engine whose memory is pooled across several machines.

A model larger than one box gets split by layer across the nodes — pipeline
parallel, not tensor parallel. Each Spark has a single GPU, so tensor
parallelism would all-reduce across the network on every layer; pipeline
parallelism hands activations over once per token per stage boundary, which the
link here comfortably carries.

The engine has a fixed world size. If a node leaves, vLLM aborts and has to be
relaunched at the new size — that is inherent to the executor, not a choice
made here, and the UI says so rather than pretending the pool is elastic.

Three things the hand-written cluster scripts got wrong and this does not:

  * ray is not in the NGC image at all, so `ray start` fails. A derived image
    (docker/vllm-ray.Dockerfile) adds it.
  * NCCL_SOCKET_IFNAME cannot be shared between nodes. The interface carrying
    the cluster subnet has a different name on each box here, and pointing a
    node's NCCL at an interface that is down on that node fails obscurely.
  * the model has to be in every node's cache, or each node tries to fetch it
    on its own at load time.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from app import docker_ctl, nodes, safety, vllm_spec
from app.config import settings

RAY_PORT = 6379
HEAD = "llmd-ray-head"
WORKER = "llmd-ray-worker"
READY_TIMEOUT = 120


@dataclass
class NodeWiring:
    """How one node reaches the cluster: which interface, at which address."""

    node: nodes.Node
    interface: str = ""
    address: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.interface and self.address)


def _subnet_prefix() -> str:
    """The cluster subnet, taken from this machine's own cluster interface."""
    import subprocess

    try:
        raw = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", settings.roce_interface],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"inet (\d+\.\d+\.\d+)\.\d+/", raw)
    return match.group(1) if match else ""


_IFACE = re.compile(r"^\d+:\s+(\S+?)\s+inet\s+(\d+\.\d+\.\d+\.\d+)/", re.M)


async def wiring(node: nodes.Node, prefix: str) -> NodeWiring:
    """Find the up interface carrying the cluster subnet ON THAT NODE.

    The name differs between machines — enp1s0f0np0 on one, enp1s0f1np1 on the
    next — so a shared NCCL_SOCKET_IFNAME points half the cluster at a dead
    port.
    """
    command = "ip -4 -o addr show | awk '{print $1, $2, $3, $4}'"
    if node.is_local:
        import subprocess

        out = subprocess.run(["bash", "-lc", command], capture_output=True,
                             text=True, timeout=15).stdout
    else:
        code, out = await nodes._ssh(node.name or node.address, command)
        if code != 0:
            return NodeWiring(node=node)

    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[3].startswith(prefix + "."):
            continue
        return NodeWiring(node=node, interface=parts[1], address=parts[3].split("/")[0])
    return NodeWiring(node=node)


async def plan(node_names: list[str], model: str = "",
               args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything that has to be true before a pooled engine can start.

    Given a model, this also reports which nodes are missing it — the answer the
    form needs *before* a launch, since each node loads its own shard from its
    own disk and a missing model there surfaces minutes in.

    Given the arguments too, it runs the memory guard on every node. A pooled
    engine declares the same utilisation fraction on each machine it spans, so
    "does this fit" has to be asked of all of them; asking only the head is how
    a config that cannot start anywhere gets accepted.
    """
    prefix = _subnet_prefix()
    if not prefix:
        return {"ok": False, "reason": f"no cluster subnet on {settings.roce_interface}"}

    resolved = [nodes.by_name(name) for name in node_names]
    if len(resolved) < 2:
        return {"ok": False, "reason": "pooling needs at least two nodes"}

    wirings = await asyncio.gather(*(wiring(n, prefix) for n in resolved))
    problems = [w.node.name for w in wirings if not w.ok]
    if problems:
        return {"ok": False, "reason": f"no interface on the {prefix}.0/24 subnet on: "
                                       f"{', '.join(problems)}"}

    images = await asyncio.gather(*(
        docker_ctl.image_exists(settings.ray_image, host=w.node.docker_host) for w in wirings
    ))
    missing_image = [w.node.name for w, has in zip(wirings, images, strict=True) if not has]

    budgets = await asyncio.gather(*(safety.current_budget(node=w.node) for w in wirings))
    pooled_bytes = sum(int(b.max_util * b.total_bytes) for b in budgets)

    missing_model_on = await missing_model(model, node_names) if model else []

    # The same fraction, judged against each machine's own free memory.
    util = vllm_spec.gpu_memory_utilization(args or {})
    verdicts = await asyncio.gather(*(
        safety.check_launch(util, params=args, node=w.node) for w in wirings
    )) if args is not None else [None] * len(wirings)
    refused = [(w.node.name, v) for w, v in zip(wirings, verdicts, strict=True)
               if v is not None and not v.ok]

    # A missing image is a blocker: nothing here can build it on a peer.
    # A missing model is not — the launch copies it over the cluster link
    # before starting the engine, so it is work to be done, not a refusal.
    reasons = []
    if missing_image:
        reasons.append(f"{settings.ray_image} is not built on: {', '.join(missing_image)}")
    for name, verdict in refused:
        reasons.append(f"{name}: {verdict.message}")

    will_sync = ""
    if missing_model_on:
        will_sync = (f"{model} will be copied to {', '.join(missing_model_on)} "
                     "over the cluster link before the engine starts")

    return {
        "ok": not reasons,
        "will_sync": will_sync,
        "reason": "; ".join(reasons),
        "missing_model_on": missing_model_on,
        "model": model,
        "head": wirings[0].node.name,
        "nodes": [
            {
                "name": w.node.name,
                "interface": w.interface,
                "address": w.address,
                "local": w.node.is_local,
                "free_util": round(b.free_util, 3),
                "total_bytes": b.total_bytes,
                "verdict": v.as_dict() if v is not None else None,
            }
            for w, b, v in zip(wirings, budgets, verdicts, strict=True)
        ],
        "pipeline_parallel_size": len(wirings),
        # A pooled engine can only ask for what the tightest node can give.
        "free_util": round(min((b.free_util for b in budgets), default=0.0), 3),
        "pooled_bytes": pooled_bytes,
        "single_node_bytes": int(budgets[0].max_util * budgets[0].total_bytes),
        "missing_image": missing_image,
    }


def _ray_env(w: NodeWiring) -> dict[str, str]:
    return {
        "NCCL_SOCKET_IFNAME": w.interface,
        "GLOO_SOCKET_IFNAME": w.interface,
        "NCCL_IB_DISABLE": "1",
        "HF_HOME": "/hf",
    }


def _mounts() -> list[docker_ctl.Mount]:
    return [
        docker_ctl.Mount(settings.hf_cache, "/hf"),
        docker_ctl.Mount(settings.output_dir, "/outputs"),
    ]


def head_command(head: NodeWiring, serve_argv: list[str]) -> list[str]:
    """Start the Ray head and then become the engine, in one container.

    The engine cannot be a separate container. A Ray driver needs the raylet's
    session directory and sockets, which live inside whichever container ran
    `ray start` — two containers on one host do not share /tmp/ray, so a
    separate driver fails with "No node info found matching attributes".
    Running the engine inside the head container is also what vLLM's own
    multi-node recipe does.
    """
    ray_start = (
        f"ray start --head --node-ip-address={head.address} --port={RAY_PORT} "
        "--dashboard-host=0.0.0.0"
    )
    quoted = " ".join(__import__("shlex").quote(part) for part in serve_argv)
    return ["-lc", f"{ray_start} && exec {quoted}"]


async def start_workers(wirings: list[NodeWiring]) -> None:
    """Join every other node to the head. The head must already be listening."""
    head, *workers = wirings
    for worker in workers:
        await docker_ctl.remove(WORKER, force=True, host=worker.node.docker_host)
        await docker_ctl.run_detached(
            name=WORKER,
            image=settings.ray_image,
            entrypoint="ray",
            command=["start", "--block", f"--address={head.address}:{RAY_PORT}",
                     f"--node-ip-address={worker.address}"],
            env=_ray_env(worker),
            mounts=_mounts(),
            gpu=True,
            network="host",
            host=worker.node.docker_host,
        )


async def await_cluster(head: NodeWiring, expected: int, container: str) -> str:
    """Wait for every node to register, so vLLM does not start against half a
    cluster and then size itself wrong."""
    deadline = asyncio.get_running_loop().time() + READY_TIMEOUT
    last = ""
    while asyncio.get_running_loop().time() < deadline:
        code, out, _ = await docker_ctl._run(
            docker_ctl.docker_argv(head.node.docker_host, "exec", container, "ray", "status"),
            check=False,
        )
        last = out
        if code == 0 and out.count("node_") >= expected:
            return out
        await asyncio.sleep(3)
    raise RuntimeError(f"only {last.count('node_')} of {expected} nodes joined within "
                       f"{READY_TIMEOUT}s:\n{last[-600:]}")


async def stop_workers(wirings: list[NodeWiring]) -> None:
    """The head goes away with the engine container; the workers are ours."""
    for worker in wirings[1:]:
        await docker_ctl.remove(WORKER, force=True, host=worker.node.docker_host)


async def status(node_names: list[str], container: str = "") -> dict:
    """Is this pool's Ray cluster up, and does it have everyone?

    The engine container IS the Ray head, so the head's health is the engine's:
    there is no separate head container to ask any more. Without a container
    name all that can be reported is which peers are carrying a worker.
    """
    if not node_names:
        return {"running": False, "nodes": 0, "expected": 0, "raw": ""}

    head = nodes.by_name(node_names[0])
    workers = []
    for name in node_names[1:]:
        peer = nodes.by_name(name)
        state = await docker_ctl.state(WORKER, peer.docker_host)
        workers.append({"node": peer.name, "running": state.running, "status": state.ui_status})

    if not container:
        return {
            "running": all(w["running"] for w in workers) and bool(workers),
            "head": head.name,
            "nodes": sum(1 for w in workers if w["running"]) + 1,
            "expected": len(node_names),
            "workers": workers,
            "raw": "",
        }

    state = await docker_ctl.state(container, head.docker_host)
    if not state.running:
        return {"running": False, "head": head.name, "nodes": 0,
                "expected": len(node_names), "workers": workers, "raw": ""}

    code, out, _ = await docker_ctl._run(
        docker_ctl.docker_argv(head.docker_host, "exec", container, "ray", "status"), check=False
    )
    return {
        "running": True,
        "head": head.name,
        "nodes": out.count("node_") if code == 0 else 0,
        "expected": len(node_names),
        "workers": workers,
        "raw": out[-2000:],
    }


async def missing_model(model: str, node_names: list[str]) -> list[str]:
    """Nodes whose cache does not have this model.

    Every node loads its own shard from its own disk, so a model present on only
    the head means the others each try to fetch it at load time.
    """
    if model.startswith("/"):
        return []
    from app import sync

    try:
        directory = sync.repo_dir_name(model)
    except Exception:
        return []

    async def has(name: str) -> tuple[str, bool]:
        node = nodes.by_name(name)
        path = f"{settings.hf_cache}/hub/{directory}"
        if node.is_local:
            from pathlib import Path

            return name, Path(path).is_dir()
        code, _ = await nodes._ssh(node.name or node.address, f"test -d {path}")
        return name, code == 0

    results = await asyncio.gather(*(has(n) for n in node_names))
    return [name for name, present in results if not present]
