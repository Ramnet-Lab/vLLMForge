"""One vLLM engine whose memory is pooled across several machines.

A model larger than one box gets split by layer across the nodes — pipeline
parallel, not tensor parallel. Each Spark has a single GPU, so tensor
parallelism would all-reduce across the network on every layer; pipeline
parallelism hands activations over once per token per stage boundary, which the
link here comfortably carries.

The engine has a fixed world size. If a node leaves, vLLM aborts and has to be
relaunched at the new size — that is inherent to the executor, not a choice
made here, and the UI says so rather than pretending the pool is elastic.

Ray is not involved, and that is the point. vLLM's own multi-node path is
torch.distributed: rank 0 serves HTTP and every other rank runs `--headless`,
and they meet at `--master-addr:--master-port`. That port is per engine, so N
pooled engines coexist. The Ray shape could only ever run one — its head bound
a fixed port on the host network and every peer ran its worker under one fixed
container name, so a second pooled launch took the first engine's far rank away
and then died on the port it could not have.

Two things the hand-written cluster scripts got wrong and this does not:

  * NCCL_SOCKET_IFNAME cannot be shared between nodes. The interface carrying
    the cluster subnet has a different name on each box here, and pointing a
    node's NCCL at an interface that is down on that node fails obscurely. It
    matters more now than it did under Ray: every rank is its own `vllm serve`
    and has to be told its own interface.
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

# Where the ranks of one engine meet. Unique per engine and only per master
# node, which is the whole reason several pooled engines can run at once.
MASTER_PORT_RANGE = range(29500, 29600)
MASTER_PORT_FLAG = re.compile(r"--master[-_]port(?:=(\S+))?")

# How long rank 0 gets to answer /health before a launch is called failed.
# Sized for reading a checkpoint on every node, not for a handshake.
READY_TIMEOUT = 600


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
            ["ip", "-4", "-o", "addr", "show", nodes.cluster_interface()],
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
               args: dict[str, Any] | None = None,
               replacing: str | None = None,
               image: str = "") -> dict[str, Any]:
    """Everything that has to be true before a pooled engine can start.

    Given a model, this also reports which nodes are missing it — the answer the
    form needs *before* a launch, since each node loads its own shard from its
    own disk and a missing model there surfaces minutes in.

    Given the arguments too, it runs the memory guard on every node. A pooled
    engine declares the same utilisation fraction on each machine it spans, so
    "does this fit" has to be asked of all of them; asking only the head is how
    a config that cannot start anywhere gets accepted.
    """
    resolved = [nodes.by_name(name) for name in node_names]
    if len(resolved) < 2:
        return {"ok": False, "reason": "pooling needs at least two nodes"}

    prefix = _subnet_prefix()
    if not prefix:
        iface = nodes.cluster_interface()
        return {"ok": False, "reason": (
            f"no cluster subnet on {iface}" if iface else
            "no interface on this machine shares a subnet with a registered peer, so there is "
            "no fabric to pool over — register the peer on the Nodes tab, or set LLMD_ROCE_IF "
            "if the right interface cannot be worked out")}

    # by_name answers "this machine" for a name it does not know, so a pool
    # naming a peer that has since been removed resolves to two ranks on one
    # box — which then rendezvous with themselves, both claim the utilisation,
    # and fail as a memory problem rather than as the configuration error it is.
    seen = [n.name for n in resolved]
    if len(set(seen)) != len(seen):
        duplicated = ", ".join(sorted({n for n in seen if seen.count(n) > 1}))
        unknown = [name for name in node_names if nodes.by_name(name).name != name]
        detail = (f" — {', '.join(unknown)} is not a registered node" if unknown else "")
        return {"ok": False,
                "reason": f"this pool puts more than one rank on {duplicated}{detail}"}

    wirings = await asyncio.gather(*(wiring(n, prefix) for n in resolved))
    problems = [w.node.name for w in wirings if not w.ok]
    if problems:
        return {"ok": False, "reason": f"no interface on the {prefix}.0/24 subnet on: "
                                       f"{', '.join(problems)}"}

    # The image the launch will actually use, not the configured default. A
    # server may carry its own, and checking the wrong tag either passes a plan
    # for an image no node has or refuses one every node does.
    image = image or settings.vllm_image
    images = await asyncio.gather(*(
        docker_ctl.image_exists(image, host=w.node.docker_host) for w in wirings
    ))
    missing_image = [w.node.name for w, has in zip(wirings, images, strict=True) if not has]

    budgets = await asyncio.gather(*(safety.current_budget(node=w.node) for w in wirings))
    # Byte twins, so the pooled total is not round-tripped through a fraction.
    pooled_bytes = sum(b.max_bytes for b in budgets)

    missing_model_on = await missing_model(model, node_names) if model else []

    # The same fraction, judged against each machine's own free memory.
    util = vllm_spec.gpu_memory_utilization(args or {})
    # `replacing` matters, and it is a DIFFERENT name on every node: rank r runs
    # as `<base>-r<r>` there. Under Ray the peer container ran `ray start`, which
    # the guard does not recognise as an engine, so one name for the whole pool
    # was harmless. A far rank runs `vllm serve --headless` and is a tenant like
    # any other, so passing rank 0's name to every node would count each peer's
    # own container against its own restart — the bug f3622ff fixed for a single
    # machine, re-introduced once per peer.
    verdicts = await asyncio.gather(*(
        safety.check_launch(util, replacing=rank_container(replacing, rank) if replacing else None,
                            params=args, node=w.node)
        for rank, w in enumerate(wirings)
    )) if args is not None else [None] * len(wirings)
    refused = [(w.node.name, v) for w, v in zip(wirings, verdicts, strict=True)
               if v is not None and not v.ok]

    # A missing image is a blocker: nothing here can build it on a peer.
    # A missing model is not — the launch copies it over the cluster link
    # before starting the engine, so it is work to be done, not a refusal.
    reasons = []
    if missing_image:
        reasons.append(f"{image} is not pulled on: {', '.join(missing_image)}")
    incompatible = _pipeline_incompatible(model)
    if incompatible:
        reasons.append(incompatible)
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
        "single_node_bytes": budgets[0].max_bytes,
        "missing_image": missing_image,
    }


def _pipeline_incompatible(model: str) -> str:
    """Why this model cannot be split by layer, when it cannot.

    Checked before anything is started, because the alternative is discovering
    it from an assertion in the far rank's warmup — after every shard has been
    read, the KV cache sized and the graphs captured.
    """
    if not model:
        return ""
    from app import model_profile

    profile = model_profile.read(model)
    if not profile.custom_sampler:
        return ""
    arch = (profile.architectures or [model])[0]
    return (
        f"{arch} supplies its own sampler, and vLLM's pipeline-parallel broadcast requires the "
        "standard one's output shape and dtype. It serves on a single machine; it cannot be "
        "split across them")



def _mounts() -> list[docker_ctl.Mount]:
    return [
        docker_ctl.Mount(settings.hf_cache, "/hf"),
        docker_ctl.Mount(settings.output_dir, "/outputs"),
    ]


def rank_container(base: str, rank: int) -> str:
    """What one rank's container is called.

    Rank 0 keeps the server's plain name, so everything that already looks a
    server up by container — logs, health, stop, the memory guard's exclusion,
    the foreign-container scan — keeps working without knowing pooling exists.
    The far ranks hang off it, which also makes them recognisable as belonging
    to this engine rather than to nobody.
    """
    return base if rank == 0 else f"{base}-r{rank}"


def rank_names(base: str, count: int) -> list[str]:
    return [rank_container(base, rank) for rank in range(max(1, count))]


def rank_env(w: NodeWiring, master: NodeWiring, base: dict[str, str] | None = None
             ) -> dict[str, str]:
    """One rank's environment, with THIS node's own interface in it.

    The only place a fabric setting is ever applied, and every value in it was
    measured on the machine the rank will run on: wiring() asked that node which
    of its interfaces carries the cluster subnet. Nothing is inherited from the
    dashboard host, because the NIC is called something different on every box
    and a name that is not present there fails NCCL initialisation rather than
    degrading — so a wrong name is worse than none.
    """
    env = dict(base or {})
    # Nothing node-specific survives from the caller's environment: a value that
    # was right for the dashboard host names a device the peer does not have.
    for stale in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
                  "NCCL_IB_HCA", "NCCL_IB_GID_INDEX"):
        env.pop(stale, None)
    env.update(settings.fabric_env())
    env.update({
        "NCCL_SOCKET_IFNAME": w.interface,
        "GLOO_SOCKET_IFNAME": w.interface,
        "NCCL_IB_DISABLE": "0" if settings.roce_hca else "1",
        "HF_HOME": "/hf",
    })
    return env


def rank_argv(serve_argv: list[str], rank: int, nnodes: int, master: NodeWiring,
              master_port: int) -> list[str]:
    """One rank's `vllm serve` command.

    Every rank is handed the *same* engine arguments and differs only in which
    rank it is, because nothing verifies that the ranks agree — a model or a
    parallel size that differs between them surfaces as a shape mismatch deep
    in the rendezvous, if it surfaces at all.

    Appended literally rather than routed through vllm_spec.build_argv, which
    drops any value equal to the schema default: --node-rank 0 and whichever
    master port matches the build's own default would silently not be emitted,
    and then nothing could read the engine's own wiring back out of its argv.
    """
    argv = list(serve_argv)
    if rank:
        # Ranks above zero run no API server. vLLM returns from the headless
        # branch before it binds anything, so they need no port of their own.
        argv.append("--headless")
    argv += [
        "--pipeline-parallel-size", str(nnodes),
        "--nnodes", str(nnodes),
        "--node-rank", str(rank),
        "--master-addr", master.address,
        "--master-port", str(master_port),
    ]
    return argv


def parse_master_port(command: list[str] | None) -> int | None:
    """The rendezvous port a running container is actually using.

    The LAST occurrence, because argparse takes the last and so must anything
    reading the value back. --master-port is a settable flag in the generated
    form, so a container's argv can carry the operator's value followed by the
    one rank_argv appended; believing the first would hand the allocator a port
    nobody is on and mark the live one free for the next engine to collide with.
    """
    argv = safety.argv_of(command)
    found = None
    for index, token in enumerate(argv):
        match = MASTER_PORT_FLAG.fullmatch(token)
        if not match:
            continue
        raw = match.group(1) if match.group(1) is not None else (
            argv[index + 1] if index + 1 < len(argv) else None)
        try:
            found = int(raw)
        except (TypeError, ValueError):
            continue
    return found


async def used_master_ports(targets: list[nodes.Node], exclude: set[str] | None = None
                            ) -> set[int]:
    """Rendezvous ports already spoken for on the machines a launch would touch.

    Read off running containers rather than out of the database, so an engine
    somebody started by hand is counted too. Two engines sharing a port do not
    fail cleanly — they rendezvous into each other, and the damage lands on
    whichever one was already serving.
    """
    exclude = exclude or set()
    ports: set[int] = set()
    for node in targets:
        try:
            rows = await docker_ctl.ps(all_containers=False, host=node.docker_host)
        except Exception:
            # An unreachable peer is the wiring check's business to report. What
            # matters here is that an unread node must not look empty, so its
            # whole range is treated as spoken for by returning what we have and
            # letting the caller's wiring check refuse the launch first.
            continue
        for row in rows:
            name = str(row.get("Names", ""))
            if not name or name in exclude:
                continue
            info = await docker_ctl.state(name, node.docker_host)
            port = parse_master_port(info.command)
            if port is not None:
                ports.add(port)
    return ports


async def allocate_master_port(targets: list[nodes.Node], exclude: set[str] | None = None
                               ) -> int:
    """The lowest rendezvous port free on every machine this engine will span.

    Callers must hold the launch lock: the answer is only true until somebody
    else takes it.
    """
    taken = await used_master_ports(targets, exclude)
    for candidate in MASTER_PORT_RANGE:
        if candidate not in taken:
            return candidate
    raise RuntimeError(
        f"no free rendezvous port in {MASTER_PORT_RANGE.start}-{MASTER_PORT_RANGE.stop - 1}; "
        f"{len(taken)} pooled engines are already running")


RANK_SUFFIX = re.compile(r"-r(\d+)$")


def is_rank_of(name: str, base: str) -> bool:
    """Whether a container is rank 0 or a far rank of the engine called `base`."""
    if name == base:
        return True
    return name.startswith(base) and bool(RANK_SUFFIX.fullmatch(name[len(base):]))


async def find_ranks(base: str, targets: list[nodes.Node]) -> list[tuple[str, str | None]]:
    """Where this engine's containers ACTUALLY are, as (name, docker host).

    Deliberately not derived from the server's pool_nodes: that row is editable
    while the engine runs, and the ranks do not move when it changes. Reordering
    the pool, or dropping a node from it, re-points every name at the wrong
    machine — so stopping the engine would stop nothing, and its far ranks would
    be left holding memory with no row left that names them.

    Stopped containers count. A rank that exited still owns its name, and the
    next launch has to remove it before it can reuse it.
    """
    found: list[tuple[str, str | None]] = []
    for node in targets:
        try:
            rows = await docker_ctl.ps(all_containers=True, host=node.docker_host)
        except Exception:
            continue
        for row in rows:
            name = str(row.get("Names", ""))
            if name and is_rank_of(name, base):
                found.append((name, node.docker_host))
    # Rank order, so a caller that acts locally first can say so.
    found.sort(key=lambda item: int(RANK_SUFFIX.search(item[0]).group(1))
               if RANK_SUFFIX.search(item[0]) else 0)
    return found


async def stop_ranks(base: str, wirings: list[NodeWiring], *, remove: bool = False) -> None:
    """Take down every rank of one engine, on the machines that hold them.

    All of them, always. A surviving rank is a process holding its full share
    of a machine for an engine that no longer exists, and nothing else will
    ever name it.
    """
    for rank, w in enumerate(wirings):
        name = rank_container(base, rank)
        if remove:
            await docker_ctl.remove(name, force=True, host=w.node.docker_host)
        else:
            await docker_ctl.stop(name, host=w.node.docker_host)


async def status(node_names: list[str], container: str = "") -> dict:
    """Is every rank of this engine up, and where is the one that is not?

    There is no cluster daemon to interrogate any more, and nothing to ask
    `ray status`. An mp engine is exactly its rank containers: if they are all
    running it is up, and if one is missing the engine is dead however healthy
    rank 0 looks — the world size is fixed, so a lost rank aborts the rest
    rather than degrading them.

    That makes the per-rank table the whole answer, and worth rendering: the
    failures unique to multi-node land in the rank that hit them, while rank 0
    shows a stall.
    """
    if not node_names:
        return {"running": False, "nodes": 0, "expected": 0, "raw": "", "ranks": []}

    head = nodes.by_name(node_names[0])
    ranks = []
    for rank, name in enumerate(node_names):
        peer = nodes.by_name(name)
        state = (await docker_ctl.state(rank_container(container, rank), peer.docker_host)
                 if container else None)
        ranks.append({
            "rank": rank,
            "node": peer.name,
            "container": rank_container(container, rank) if container else "",
            "running": bool(state and state.running),
            "status": state.ui_status if state else "not started",
        })

    up = sum(1 for r in ranks if r["running"])
    return {
        "running": bool(container) and up == len(ranks),
        "head": head.name,
        "nodes": up,
        "expected": len(node_names),
        # Kept under the old key so the pool panel keeps rendering something
        # useful; it is a rank table now rather than a dump of `ray status`.
        "workers": [r for r in ranks if r["rank"]],
        "ranks": ranks,
        "raw": "\n".join(
            f"rank {r['rank']} on {r['node']}: {r['status']}" for r in ranks),
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
