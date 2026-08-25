"""Launch-time memory guard.

The single most destructive mistake available on this box is starting a vLLM
server whose --gpu-memory-utilization does not fit. GPU memory is host memory:
overcommit does not degrade performance, it freezes the machine during CUDA
graph capture, and the kernel OOM killer reacts far too slowly to save an
interactive desktop.

Two independent checks run before any launch:

  1. Static budget. vLLM sizes its allocation as `util * torch_total`, and on a
     unified-memory part torch_total equals MemTotal. So the utils of every
     resident engine sum directly, and the sum must leave a reserve for the OS,
     torch.compile and transient activation spikes.

  2. Live headroom. Even a budget-legal launch is refused if MemAvailable right
     now cannot cover it, because something outside the dashboard may be holding
     memory.

Containers the dashboard did not start still count: their utils are parsed out
of their recorded command line, so a hand-launched `vllm serve` is included in
the budget exactly like a managed one.

**Two engines, one pool.** A vLLM server declares a fraction; a llama.cpp server
declares nothing and is priced from its own weights file. Both spend the same
memory, so both are tenants of the same budget — but the arithmetic below was
already denominated in bytes (`Tenant.bytes_committed` sums, not utils), and
that is what makes a second engine a routing change rather than a rewrite.

What is engine-specific — recognising a container, pricing it, and the words a
verdict is phrased in — lives in `app/engines/`. Three things about that seam
are deliberate and load-bearing:

  * `is_vllm_command` is NOT widened. It is vLLM's own predicate and stays so;
    `engines.recognise()` asks the general question. A container recognised
    before this file gained a second engine cannot change what it is.
  * Recognition and pricing are members of one object, because
    `vllm_spec.footprint_bytes` charges its default fraction — over 100 GiB
    here — to any argv with no utilisation flag, which is every llama.cpp argv.
  * `check_launch` branches to a bytes-native twin at its first line and is
    otherwise untouched, so every vLLM verdict is the string it has always been.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from app import accel, docker_ctl, engines, vllm_spec
from app.config import settings
from app.engines.vllm import KV_BYTES_FLAG, OFFLOAD_FLAG, UTIL_FLAG

GIB = 1024 ** 3

_VLLM = engines.get("vllm")

__all_flags__ = (UTIL_FLAG, KV_BYTES_FLAG, OFFLOAD_FLAG)


def flag_value(command: list[str] | None, pattern: re.Pattern[str]) -> str | None:
    """The value of a flag in a container's argv, whether joined by = or spaced.

    Kept here, and kept unwrapping the shell the vLLM way, because that is what
    its callers have always meant. The engine-neutral version — which takes an
    argv that is already tokens — is `engines.flag_value`, and each engine
    unwraps with its own sentinel before calling it.
    """
    return engines.flag_value(argv_of(command), pattern)


# How close to MemAvailable a request may sit before it is worth a warning.
# Recommendations keep the same headroom; see app/recommend.py UTIL_MARGIN.
DRIFT_MARGIN = 0.03


def default_util() -> float:
    """vLLM's own default, read from the image's schema so it tracks upgrades."""
    return _VLLM.implicit_util()


def parse_util(command: list[str] | None) -> float | None:
    """Pull --gpu-memory-utilization out of a container's argv.

    Deliberately vLLM's, and only vLLM's. A llama-server argv has no such flag,
    and routing one through here returns None — which `vllm_spec.footprint_bytes`
    then prices at the default fraction, over 100 GiB on this box. The general
    question is `engines.recognise(...).declared_util(...)`.
    """
    return _VLLM.parse_util(command)


def command_params(command: list[str] | None) -> dict[str, Any]:
    """The memory-relevant flags of a running vLLM container, shaped like stored
    args, so a hand-launched container is accounted exactly like a managed one."""
    return _VLLM.command_params(command)


def footprint(params: dict[str, Any], total_bytes: int, *, engine: str = "vllm",
              devices: int = 1) -> int:
    """A floor on the memory these params will take, priced by their own engine.

    The chokepoint: every byte figure in this module and in the watchdog flows
    through here. `engine` is a keyword with vLLM's default so every existing
    call site keeps its exact value, and it is what stops a llama.cpp params dict
    reaching vLLM's pricer — which would charge it the default fraction and
    refuse every launch on the machine afterwards.

    Synchronous on purpose: the watchdog calls it once per container per tick.
    An engine whose price needs a file read does that read in `Engine.resolve()`,
    which the async callers await before they get here.
    """
    return engines.get(engine).footprint_bytes(params, total_bytes, devices=devices)


def argv_of(command: list[str] | None) -> list[str]:
    """A vLLM container's command as tokens, unwrapping a shell if there is one.

    Named for what its callers want: `cluster.parse_master_port` reads a vLLM
    engine's own wiring back out of its argv, which is a vLLM question. Each
    engine has its own unwrapper because each has its own sentinel.
    """
    return _VLLM.argv_of(command)


def is_vllm_command(command: list[str] | None) -> bool:
    """Whether this argv is a vLLM engine.

    NOT widened to mean "is an engine", and that is the point. Widening it would
    admit a llama.cpp container to every caller at once — the budget survey, the
    watchdog's kill list, foreign discovery — while `footprint` still priced it
    with vLLM's arithmetic. `engines.recognise()` is the general question, and
    it routes recognition and pricing together.
    """
    return _VLLM.matches(command)


@dataclass
class Tenant:
    name: str
    util: float | None
    """The fraction the operator declared, for an engine that has such a thing.
    None for llama.cpp, and deliberately not a fraction derived from the bytes:
    a Util column carrying both would look summable, and it is not."""
    managed: bool
    bytes_committed: int = 0
    implicit: bool = False        # util came from the engine's default, not the argv
    note: str = ""
    engine: str = "vllm"
    label: str = ""
    """How this tenant reads in a verdict message: 'llmd-vllm-3=0.52' for an
    engine that declares a fraction, 'llmd-llamacpp-3=40.1 GiB' for one that does
    not. Derived when it is left empty, so a Tenant built the way it always was
    — four positional-ish fields and no label — still names itself correctly."""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = _tenant_label(self.name, self.util, self.implicit,
                                       self.bytes_committed)


@dataclass
class Budget:
    total_bytes: int = 0
    available_bytes: int = 0
    free_bytes: int = 0
    measured_gpu_bytes: int = 0
    excluded_bytes: int = 0
    """What the excluded container is measured to be holding. A restart gets its
    own memory back, and nvidia-smi cannot know that — it reports processes, not
    intentions — so it is subtracted here rather than left to inflate a budget
    that is about to change."""
    reserve_bytes: int = 0
    warn_reserve_bytes: int = 0
    devices: int = 1
    """How many cards `total_bytes` spans. 1 on a unified box and on any pool
    that could not be read, which is what keeps a per-device fraction meaning
    the whole pool exactly as it always did."""
    tenants: list[Tenant] = field(default_factory=list)
    pool: Any = None
    """The accel.Pool these figures came from: which memory they describe, how
    that was decided, and how sure it is. Additive — every field above keeps
    the meaning it had when they were always host memory."""

    @property
    def committed_bytes(self) -> int:
        # Summed in bytes, not fractions: a tenant using --kv-cache-memory or
        # --cpu-offload-gb takes memory its utilisation fraction never mentions.
        return sum(t.bytes_committed for t in self.tenants)

    @property
    def committed_util(self) -> float:
        return self.committed_bytes / self.total_bytes if self.total_bytes else 0.0

    @property
    def occupied_bytes(self) -> int:
        """What is really spoken for.

        The util sum misses anything that is not a vLLM engine — a fine-tuning
        job, a Heretic run, even a browser with a few GiB of GPU surfaces — so
        the measured per-process figure is used whenever it is larger.

        The container being replaced comes off the measured side only. The
        committed side never had it: current_budget drops it before it reaches
        the tenant list. Taking it off both was invisible while only the local
        node had tenants — measured is read from nvidia-smi and is 0 for a peer,
        so the two figures agreed — but a pooled engine puts a real tenant on
        every node it spans, and there the double subtraction cancelled a peer's
        genuine occupancy and reported a full machine as empty.
        """
        return max(0, max(self.committed_bytes, self.measured_gpu_bytes - self.excluded_bytes))

    @property
    def available_after_replacement(self) -> int:
        """MemAvailable as it will be once the container being replaced has gone.

        vLLM compares its request against free memory at the moment the engine
        starts, and by then the old container has been removed — so judging a
        restart against memory that container is still holding refuses launches
        that would have been fine. Same reasoning as excluding it from the
        budget; this is the live half of it, which was missed.
        """
        return min(self.total_bytes, self.available_bytes + self.excluded_bytes)

    @property
    def occupied_util(self) -> float:
        return self.occupied_bytes / self.total_bytes if self.total_bytes else 0.0

    @property
    def max_bytes(self) -> int:
        """Largest total commitment that still leaves the hard reserve.

        The byte twin of `max_util`, added rather than substituted. It would be
        tidier to define one in terms of the other, and it would also be a
        silent risk: `1 - r/t` and `(t - r)/t` can differ by one unit in the last
        place, `free_util` is floored to two decimals and handed to the operator
        as a value to retype, and a floor at an exact boundary is exactly where a
        one-ULP difference changes the answer. Two independent expressions of the
        same quantity cost nothing and cannot regress each other.
        """
        return max(0, self.total_bytes - self.reserve_bytes)

    @property
    def refusals_are_advice(self) -> bool:
        """Whether exhausting this pool costs a container or costs the machine.

        On a discrete GPU the framebuffer is not where the OS lives. An
        overcommit is a CUDA out-of-memory inside the process that asked for too
        much; that process dies and the desktop, the ssh session and every other
        engine on the box carry on. There is nothing here for a guard to save,
        so the capacity checks below say what they measured and let the operator
        through.

        On a unified part the same bytes ARE the OS's. An overcommit freezes the
        machine during graph capture and no OOM killer reacts in time, which is
        the whole reason this module exists — so there a refusal is a refusal.

        `accel` fails unified on any doubt, so an unmeasured, overridden or
        ambiguous box keeps the veto rather than inheriting this.
        """
        return getattr(self.pool, "kind", "") == accel.DISCRETE

    @property
    def free_bytes_to_commit(self) -> int:
        """What a new engine may take. The byte twin of `free_util`, and the only
        answer available to an engine that declares no fraction."""
        return max(0, self.max_bytes - self.occupied_bytes)

    @property
    def max_util(self) -> float:
        """Largest total utilisation that still leaves the hard reserve."""
        if not self.total_bytes:
            return 0.0
        return max(0.0, 1.0 - self.reserve_bytes / self.total_bytes)

    @property
    def warn_util(self) -> float:
        if not self.total_bytes:
            return 0.0
        return max(0.0, 1.0 - self.warn_reserve_bytes / self.total_bytes)

    @property
    def free_util(self) -> float:
        return max(0.0, self.max_util - self.occupied_util)

    def as_dict(self) -> dict:
        return {
            "total_bytes": self.total_bytes,
            "devices": self.devices,
            "available_bytes": self.available_bytes,
            "free_bytes": self.free_bytes,
            "measured_gpu_bytes": self.measured_gpu_bytes,
            "reserve_bytes": self.reserve_bytes,
            "warn_reserve_bytes": self.warn_reserve_bytes,
            "committed_util": round(self.committed_util, 4),
            "committed_bytes": self.committed_bytes,
            "occupied_bytes": self.occupied_bytes,
            "occupied_util": round(self.occupied_util, 4),
            "max_util": round(self.max_util, 4),
            "warn_util": round(self.warn_util, 4),
            "free_util": round(self.free_util, 4),
            "default_util": default_util(),
            # Byte twins of max_util/free_util, for an engine that has no
            # fraction to express them as. Additive: every key above keeps its
            # meaning and its value, so a vLLM-only box reports exactly what it
            # reported before there was a second engine.
            "max_bytes": self.max_bytes,
            "free_bytes_to_commit": self.free_bytes_to_commit,
            # Which memory every figure above describes, so a reader never has
            # to assume it is host RAM. Additive: no key above changed.
            "pool_kind": getattr(self.pool, "kind", "unified"),
            "pool_confidence": getattr(self.pool, "confidence", "assumed"),
            "pool_devices": getattr(self.pool, "device_count", 0),
            "pool_evidence": list(getattr(self.pool, "evidence", ()) or ()),
            "host_available_bytes": getattr(
                getattr(self.pool, "host", None), "available_bytes", self.available_bytes),
            "tenants": [
                {
                    "name": t.name,
                    "util": t.util,
                    "managed": t.managed,
                    "bytes_committed": t.bytes_committed,
                    "implicit": t.implicit,
                    "note": t.note,
                    "engine": t.engine,
                    "label": t.label,
                }
                for t in self.tenants
            ],
        }


@dataclass
class Verdict:
    ok: bool
    level: str            # ok | warn | block
    message: str
    budget: dict
    requested_util: float = 0.0
    requested_bytes: int = 0
    suggested_util: float | None = None
    suggested_bytes: int | None = None
    """What an engine with no utilisation fraction is told it may take. Left None
    on the vLLM path, where `suggested_util` is the actionable answer and the
    form labels it as the flag to type."""
    engine: str = "vllm"

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "level": self.level,
            "message": self.message,
            "requested_util": round(self.requested_util, 4),
            "requested_bytes": self.requested_bytes,
            "suggested_util": (
                round(self.suggested_util, 3) if self.suggested_util is not None else None
            ),
            "suggested_bytes": self.suggested_bytes,
            "engine": self.engine,
            "budget": self.budget,
        }


def _gib(value: float) -> str:
    return f"{value / GIB:.1f} GiB"


async def current_budget(exclude: str | None = None, node: Any = None) -> Budget:
    """Survey every running container and process that has taken GPU memory.

    `node` is an app.nodes.Node; omitted means this machine. A launch aimed at a
    peer has to be measured against that peer's memory, not against ours.
    """
    from app import accel

    host = getattr(node, "docker_host", None)
    # One probe decides what these numbers describe. On a unified machine it
    # answers with MemTotal and MemAvailable, byte for byte, and this is the
    # same budget it always was; on a machine whose GPU has its own memory it
    # answers with the framebuffer, which is what the utilisation fraction
    # actually multiplies there. Nothing below needs to know which it got.
    pool = await accel.pool_for(node)

    budget = Budget(
        total_bytes=pool.total_bytes,
        available_bytes=pool.available_bytes,
        free_bytes=pool.free_bytes,
        reserve_bytes=pool.reserve_bytes,
        warn_reserve_bytes=pool.warn_reserve_bytes,
        devices=max(1, pool.device_count),
        pool=pool,
    )

    # Per-process accounting comes from nvidia-smi, which reports on THIS
    # machine. Counting it against a peer made an idle peer look half full and
    # refused launches it had ample room for.
    if host is None:
        budget.measured_gpu_bytes = pool.measured_bytes

    for row in await docker_ctl.ps(all_containers=False, host=host):
        name = str(row.get("Names", ""))
        if not name:
            continue
        info = await docker_ctl.state(name, host)
        # The argv comes back with the engine: a container whose binary lives in
        # its image's ENTRYPOINT is recognised from entrypoint+cmd, and reading
        # its flags from `command` alone would find none of them and price a
        # resident engine at nothing.
        engine, argv = engines.identify(info)

        # The exclusion is tested BEFORE the engine gate, and the order is a fix
        # rather than a tidy-up. It used to sit below, so a container the gate
        # did not recognise could never set excluded_bytes — which meant a large
        # engine of any kind the gate did not know was guaranteed to block its
        # own restart, charged through measured_gpu_bytes for memory it was about
        # to release. For a vLLM container both orders reach the same branch, and
        # for an unrecognised one both leave excluded_bytes at zero.
        if exclude and name == exclude:
            # The container this survey is about to replace. It is not a tenant,
            # and what it holds is memory the replacement can have back — so it
            # comes off the measured total too, which otherwise counts the very
            # engine being restarted against its own restart.
            if engine is not None:
                params = await engine.resolve(engine.command_params(argv))
                credit = engine.footprint_bytes(params, pool.total_bytes,
                                                devices=budget.devices)
                # Capped by what the machine is actually measured to be holding.
                # `excluded_bytes` is subtracted from occupancy AND added to
                # available memory, so an estimate biased high — which llama.cpp's
                # deliberately is — would flip its own safe direction on a
                # restart and credit back more than the container can release.
                if engine.declared_util(params) is None and pool.measured_bytes:
                    credit = min(credit, pool.measured_bytes)
                budget.excluded_bytes = credit
            continue

        if engine is None:
            continue

        params = await engine.resolve(engine.command_params(argv))
        util = engine.declared_util(params)
        # "Implicit" means the engine has a default fraction and is applying it
        # because nothing was declared — a vLLM serve with no
        # --gpu-memory-utilization is not free, it takes over 100 GiB here. An
        # engine with no such concept is not implicit, it is simply unpriced by
        # fraction, and saying "(implied)" beside an empty column would be noise.
        implicit = util is None and engine.implicit_util() is not None
        if implicit:
            util = engine.implicit_util()
        committed = engine.footprint_bytes(params, budget.total_bytes,
                                           devices=budget.devices)

        budget.tenants.append(
            Tenant(
                name=name,
                util=util,
                managed=name.startswith(settings.container_prefix),
                bytes_committed=committed,
                implicit=implicit,
                note="; ".join(engine.notes(params, implicit=implicit)),
                engine=engine.name,
                label=_tenant_label(name, util, implicit, committed),
            )
        )
    return budget


def _tenant_label(name: str, util: float | None, implicit: bool, committed: int) -> str:
    """How one resident engine reads inside a verdict message.

    For anything declaring a fraction this is character-for-character the string
    the message used to build inline, so a vLLM-only box's verdicts are the ones
    it has always produced. An engine without a fraction reports what it is
    actually holding, and an engine whose weights could not be read says so
    rather than reporting a confident nothing.
    """
    if util is not None:
        return f"{name}={util:g}{' (implied)' if implicit else ''}"
    return f"{name}={_gib(committed)}" if committed else f"{name}=size unknown"


async def check_launch(
    util: float | None,
    *,
    replacing: str | None = None,
    params: dict[str, Any] | None = None,
    node: Any = None,
    engine: str = "vllm",
) -> Verdict:
    """Decide whether a launch is safe right now.

    Pass `params` — the whole stored argument dict — wherever it is available:
    the utilisation fraction alone does not describe what a config will take.

    `engine` is a keyword with vLLM's default, and everything below this first
    branch is the code it has always been. That is on purpose. The four verdict
    messages, the preface they are assembled from and the two-decimal floor on
    the suggestion are the observable contract — tests assert on their
    substrings and the form renders `suggested_util` as the flag to type — so
    the bytes-native path is a separate function that shares the *policy* rather
    than a template that has to be proved equal to these strings.

    It also has to be a branch and not a fall-through: `footprint` below would
    price a llama.cpp params dict with vLLM's arithmetic and charge it the
    default fraction, which on this box is over 100 GiB.
    """
    if engine != "vllm":
        return await _check_launch_bytes(
            engines.get(engine), params or {}, replacing=replacing, node=node)

    budget = await current_budget(exclude=replacing, node=node)
    payload = budget.as_dict()
    fallback = default_util()

    if util is None and params:
        util = vllm_spec.gpu_memory_utilization(params)
    if util is None:
        util = fallback
        preface = (
            f"No --gpu-memory-utilization set, so vLLM will apply its default of {fallback:g}. "
        )
    else:
        preface = ""

    requested_bytes = footprint({**(params or {}), "gpu_memory_utilization": util},
                                budget.total_bytes, devices=budget.devices)
    # What the fraction ALONE would cost, priced the same way — same pool, same
    # device count, same parallel spread — so the only difference left is the
    # two flags below. Comparing against a bare `util x total_bytes` stopped
    # being that baseline the moment the fraction became per-device, and would
    # have accused every multi-card launch of setting flags it had not set.
    bare = {k: v for k, v in (params or {}).items()
            if k not in ("kv_cache_memory_bytes", "kv_cache_memory", "cpu_offload_gb")}
    fraction_only = footprint({**bare, "gpu_memory_utilization": util},
                              budget.total_bytes, devices=budget.devices)
    if params and requested_bytes > fraction_only:
        preface += (
            "This config also sets "
            + " and ".join(
                flag for flag, key in (
                    ("--kv-cache-memory", "kv_cache_memory_bytes"),
                    ("--cpu-offload-gb", "cpu_offload_gb"),
                ) if key in params
            )
            + ", so its real footprint is larger than the fraction implies. "
        )
    projected_bytes = budget.occupied_bytes + requested_bytes
    projected = projected_bytes / budget.total_bytes if budget.total_bytes else 0.0
    headroom_after = budget.total_bytes - projected_bytes
    # `free_util` is a fraction of the whole pool, and --gpu-memory-utilization
    # is a fraction of one card charged once per card the ranks span. Those are
    # the same number only when the ranks span every card — the usual case, and
    # exactly 1.0 on a single-card or unified box, which is why this multiplier
    # changes no answer that box has ever been given. A tensor-parallel-1 engine
    # on a two-card machine may have twice the fraction its share of the pool
    # suggests, and telling it otherwise halves the card for no reason.
    spanned = vllm_spec.parallel_devices(params) if budget.devices > 1 else 1
    headroom_util = budget.free_util * budget.devices / max(1, spanned)
    # Floor, never round: suggesting 0.06 when 0.057 is the ceiling hands the
    # user a value the very next check refuses.
    safe_now = math.floor(max(0.0, min(1.0, headroom_util)) * 100) / 100
    suggested = safe_now or None

    # `label` is built at construction and, for a tenant declaring a fraction,
    # is exactly the string this line used to assemble inline.
    tenants = ", ".join(t.label for t in budget.tenants) if budget.tenants else "none"

    if projected_bytes > budget.total_bytes - budget.reserve_bytes:
        if budget.refusals_are_advice:
            # Discrete: this is video memory, and the sentence below about "the
            # OS and torch.compile" would be describing a pool the OS is not in.
            return Verdict(
                ok=True,
                level="warn",
                message=(
                    f"{preface}Over capacity: {util:g} on top of "
                    f"{_gib(budget.occupied_bytes)} already in use ({tenants}) wants "
                    f"{_gib(projected_bytes)} of {_gib(budget.total_bytes)} of video memory. "
                    f"Nothing is refused — the OS does not live in a framebuffer, so being wrong "
                    f"here costs this engine a CUDA out-of-memory and costs the rest of the box "
                    f"nothing. {budget.free_util:.2f} is what currently fits."
                ),
                budget=payload,
                requested_util=util,
                requested_bytes=requested_bytes,
                suggested_util=suggested,
            )
        return Verdict(
            ok=False,
            level="block",
            message=(
                f"{preface}Refusing to launch: {util:g} on top of {_gib(budget.occupied_bytes)} "
                f"already in use ({tenants}) would reserve {_gib(projected_bytes)} of "
                f"{_gib(budget.total_bytes)} host memory, leaving only {_gib(headroom_after)} for "
                f"the OS and torch.compile. This host needs at least {_gib(budget.reserve_bytes)} "
                f"free. Largest safe value right now is {budget.free_util:.2f}."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
            suggested_util=suggested,
        )

    live_available = budget.available_after_replacement
    if requested_bytes > live_available:
        # This is vLLM's own gate, not a policy of ours. On a unified-memory box
        # the engine reads psutil.virtual_memory().available — MemAvailable — and
        # refuses when ceil(total * util) exceeds it.
        return Verdict(
            ok=False,
            level="block",
            message=(
                f"{preface}Refusing to launch: {util:g} needs {_gib(requested_bytes)} but only "
                f"{_gib(live_available)} is available on the host right now — which is "
                "the number vLLM itself compares against, so it would refuse to start. "
                "Stop something first."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
            suggested_util=suggested,
        )

    # MemAvailable moves while an image is pulled and weights are read, so a
    # request that only just fits now can be refused by the time the engine
    # starts. That is worth a word; page cache is not — vLLM counts reclaimable
    # memory as free on this hardware, and telling the operator to drop caches
    # for a launch that would have succeeded is how an afternoon disappears.
    #
    #   vllm/utils/mem_utils.py: if current_platform.is_integrated_gpu(...):
    #       self.free_memory = psutil.virtual_memory().available
    #
    if requested_bytes > live_available * (1 - DRIFT_MARGIN):
        return Verdict(
            ok=True,
            level="warn",
            message=(
                f"{preface}{util:g} needs {_gib(requested_bytes)} of the "
                f"{_gib(live_available)} available — close enough that a little more "
                "activity before the engine starts would push it over vLLM's own check. "
                f"{max(0.0, suggested or 0):.2f} leaves room for that."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
            suggested_util=suggested,
        )

    if projected_bytes > budget.total_bytes - budget.warn_reserve_bytes:
        return Verdict(
            ok=True,
            level="warn",
            message=(
                f"{preface}Tight: total utilisation would be {projected:.2f} "
                f"({_gib(projected_bytes)}), leaving {_gib(headroom_after)}. "
                "Keep --max-num-seqs low; concurrency spikes are what actually kill this box."
            ),
            budget=payload,
            requested_util=util,
            requested_bytes=requested_bytes,
        )

    return Verdict(
        ok=True,
        level="ok",
        message=(
            f"{preface}Fits: {util:g} reserves {_gib(requested_bytes)}; total {projected:.2f} "
            f"leaves {_gib(headroom_after)} for the host."
        ),
        budget=payload,
        requested_util=util,
        requested_bytes=requested_bytes,
    )


async def _check_launch_bytes(
    engine: Any,
    params: dict[str, Any],
    *,
    replacing: str | None = None,
    node: Any = None,
) -> Verdict:
    """The same decision for an engine that states its appetite in bytes.

    The *policy* is shared with `check_launch` above and must stay so: the same
    four `Budget` properties, read in the same order, against the same
    thresholds. What differs is only how the request was arrived at and how the
    answer is phrased — an operator running llama.cpp is going to act on a layer
    count and a context length, and being told the largest safe
    `--gpu-memory-utilization` would be advice about a flag their engine has
    never heard of.

    One asymmetry is worth stating because it inverts a comment above. The
    available-memory check on the vLLM path merely *predicts* vLLM's own refusal:
    the engine compares its request against free memory and declines to start.
    Recent llama.cpp does something similar when `-ngl` is left on `auto`, but it
    aims to leave 1 GiB free where this host wants tens, and an explicit `-ngl`
    turns the fitting off entirely. So here the check is not a prediction of
    somebody else's politeness — it is the only thing standing between an
    explicit layer count and a machine that stops responding.
    """
    budget = await current_budget(exclude=replacing, node=node)
    payload = budget.as_dict()

    resolved = await engine.resolve(params)
    requested_bytes = engine.footprint_bytes(resolved, budget.total_bytes,
                                             devices=budget.devices)
    unsized = str(resolved.get("_sizing") or "")

    projected_bytes = budget.occupied_bytes + requested_bytes
    headroom_after = budget.total_bytes - projected_bytes
    live_available = budget.available_after_replacement
    free = budget.free_bytes_to_commit
    tenants = ", ".join(t.label for t in budget.tenants) if budget.tenants else "none"

    def verdict(ok: bool, level: str, message: str) -> Verdict:
        return Verdict(
            ok=ok,
            level=level,
            message=message,
            budget=payload,
            # A display figure only, exactly as app/heretic.py and
            # app/finetune.py already compute one for workloads that take no
            # fraction. Never handed back as advice.
            requested_util=(requested_bytes / budget.total_bytes) if budget.total_bytes else 0.0,
            requested_bytes=requested_bytes,
            suggested_util=None,
            suggested_bytes=free,
            engine=engine.name,
        )

    advisory = budget.refusals_are_advice

    if projected_bytes > budget.total_bytes - budget.reserve_bytes:
        if advisory:
            return verdict(True, "warn", (
                f"Over capacity: this configuration needs {_gib(requested_bytes)} on top of "
                f"{_gib(budget.occupied_bytes)} already in use ({tenants}), against "
                f"{_gib(budget.total_bytes)} of video memory — about {_gib(free)} of it free. "
                "Nothing is refused: overcommitting a framebuffer costs this engine a CUDA "
                "out-of-memory while the machine carries on. To make it fit, lower "
                "--n-gpu-layers, shorten --ctx-size, or quantise the cache with -ctk q8_0."))
        return verdict(False, "block", (
            f"Refusing to launch: this configuration needs {_gib(requested_bytes)} on top of "
            f"{_gib(budget.occupied_bytes)} already in use ({tenants}), which would leave only "
            f"{_gib(headroom_after)} of {_gib(budget.total_bytes)}. This host needs at least "
            f"{_gib(budget.reserve_bytes)} free. There is room for {_gib(free)} right now — "
            "lower --n-gpu-layers, shorten --ctx-size, or quantise the cache with -ctk q8_0."))

    if requested_bytes > live_available:
        if advisory:
            # The refusal below is a claim about a unified part, where the
            # allocation llama-server does not check first is the machine's own
            # memory. On a discrete card it is the card's, and the failure is a
            # container that exits rather than a host that stops responding.
            return verdict(True, "warn", (
                f"Tight: this configuration needs {_gib(requested_bytes)} and only "
                f"{_gib(live_available)} of video memory is free right now. llama-server does "
                "not check before it allocates, so if this is wrong it will die partway through "
                "loading — which costs the load and nothing else."))
        return verdict(False, "block", (
            f"Refusing to launch: this configuration needs {_gib(requested_bytes)} but only "
            f"{_gib(live_available)} is free right now. Unlike vLLM, llama-server does not "
            "check before it allocates — with an explicit --n-gpu-layers it simply tries, and "
            "on this host that is what takes the machine down. Stop something first."))

    if unsized:
        # Nothing about the request could be measured, so nothing above tested
        # it. Saying "fits" here would be a claim the guard has no basis for.
        return verdict(True, "warn", (
            f"This launch cannot be sized: {unsized}. Nothing is refused, but the memory guard "
            f"is not protecting you here — there is {_gib(free)} free, and llama.cpp will take "
            "what it takes. Point the Model field at a .gguf already on this machine to get a "
            "real answer."))

    if requested_bytes > live_available * (1 - DRIFT_MARGIN):
        return verdict(True, "warn", (
            f"Tight: {_gib(requested_bytes)} of the {_gib(live_available)} free — close enough "
            "that a little more activity before the weights finish loading would push it over. "
            f"{_gib(free)} is the comfortable ceiling."))

    if projected_bytes > budget.total_bytes - budget.warn_reserve_bytes:
        return verdict(True, "warn", (
            f"Tight: {_gib(projected_bytes)} of {_gib(budget.total_bytes)} would be spoken for, "
            f"leaving {_gib(headroom_after)}. Keep --parallel and --ubatch-size low; the compute "
            "buffer grows with both, and it is what runs out during a long prompt rather than "
            "during loading."))

    return verdict(True, "ok", (
        f"Fits: {_gib(requested_bytes)} of {_gib(budget.total_bytes)}, leaving "
        f"{_gib(headroom_after)}. This is an estimate from the GGUF's own header — llama.cpp "
        "measures the real buffers at startup and prints them."))
