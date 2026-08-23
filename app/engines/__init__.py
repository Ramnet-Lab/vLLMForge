"""Which inference engine a server runs, and everything that differs because of it.

The dashboard used to have one engine, and "the engine" was spelled `vllm` in
about forty places: the argv builder, the container's name, the image default,
the flag schema, the Prometheus prefix, the memory guard's recogniser and its
pricer. A second engine is not a second copy of any of that — it is the same
lifecycle asking a different object the same questions.

Three rules shape what is on the protocol below, and each of them is a bug that
would otherwise be easy to write:

* **The vLLM answer never changes.** `engines/vllm.py` is a pure adapter: every
  member either delegates to `app/vllm_spec.py` or is a function moved verbatim
  out of `app/safety.py`. `is_vllm_command` in particular is NOT widened — it
  becomes vLLM's `matches()` unchanged, and the generic question is asked by
  `recognise()` instead. A container that is recognised today therefore cannot
  change engine, price or tenant identity because this module now exists.

* **Recognition and pricing move together.** `vllm_spec.footprint_bytes` charges
  `default_util x total` — over 100 GiB on a unified box — to any argv with no
  utilisation flag, which is every llama.cpp argv there is. So a recogniser that
  admits a second engine without routing the footprint at the same moment does
  not merely misreport: it refuses every subsequent launch on the machine.
  `matches()` and `footprint_bytes()` are members of one object for that reason.

* **The expensive part is separated from the cheap part.** `footprint_bytes()`
  is synchronous, because the memory watchdog calls it once per container per
  tick and the launch guard calls it while assembling a message. llama.cpp's
  footprint needs a file's size and its GGUF header, which is I/O — so that
  happens in `resolve()`, which is async, is awaited once by the three callers
  that already are, and hands back a params dict the sync pricer can read.
  vLLM's `resolve()` returns its argument.

Import discipline: nothing under `app/engines/` may import `app.safety`,
`app.servers` or `app.docker_ctl` at runtime. The chain is
safety → engines → {vllm_spec, gguf} → config, and it is acyclic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


# --- shared argv reading -------------------------------------------------
#
# Engine-neutral by construction: the caller supplies the pattern, so the same
# code reads `--gpu-memory-utilization=0.52` and `-ngl 40`. It deliberately does
# NOT unwrap a shell command — each engine's sentinel differs, so unwrapping is
# the engine's job and this is handed an argv that is already tokens.

def flag_value(argv: list[str] | None, pattern: re.Pattern[str]) -> str | None:
    """The value of a flag in an argv, whether joined by `=` or spaced."""
    if not argv:
        return None
    for index, token in enumerate(argv):
        match = pattern.fullmatch(token)
        if not match:
            continue
        if match.group(1) is not None:
            return match.group(1)
        if index + 1 < len(argv):
            return argv[index + 1]
        return None
    return None


def flag_present(argv: list[str] | None, *names: str) -> bool:
    """Whether a bare switch appears, in any of its spellings."""
    wanted = set(names)
    return any(token in wanted or token.split("=", 1)[0] in wanted
               for token in (argv or []))


def as_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# --- the protocol --------------------------------------------------------

@runtime_checkable
class Engine(Protocol):
    """One inference engine, as the rest of the dashboard needs to see it.

    Every member below exists because a specific call site asks for it; the
    comment names that call site, so a member with no caller is a member to
    delete rather than a member to implement.
    """

    # --- identity ---
    name: str
    """The `engine` column's value AND the container kind — servers.container_name
    passes it to settings.container_name(kind, id), so an existing vLLM row keeps
    the `llmd-vllm-<id>` name it already has."""

    label: str
    """'vLLM' / 'llama.cpp'. The editor's panel subtitle and the engine badge."""

    binary: str
    """The program name the argv starts with, for the command preview's title."""

    entrypoint: str | None
    """Passed to docker_ctl.build_run_argv. None means the image's own ENTRYPOINT
    is correct, which is true of every image this repo builds."""

    gpu: bool
    """Whether the container gets `--runtime nvidia --gpus all`."""

    supports_pooling: bool
    """Whether one engine may be split across machines. vLLM pools with
    torch.distributed pipeline parallel; llama.cpp's `--rpc` is a different
    topology with different failure modes, so it is refused rather than faked."""

    served_name_dest: str
    """The flag dest the `served_name` column becomes."""

    interesting_metrics: tuple[str, ...]
    """The Prometheus series the UI promotes out of a /metrics scrape."""

    # --- schema and command assembly ---
    def ui_model(self) -> dict[str, Any]:
        """The form model. GET /api/servers/schema?engine=…"""

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Problems with a stored args dict. ServerIn/ServerPatch validation."""

    def managed_flags(self) -> frozenset[str]:
        """Flags the dashboard sets itself and the form must not offer."""

    def path_kinds(self) -> dict[str, str]:
        """dest → the kind of on-disk thing it names, for the form's pickers."""

    def build_argv(self, model: str, params: dict[str, Any], *,
                   host: str = "0.0.0.0", port: int | None = None) -> list[str]:
        """The container's command. servers.build_command."""

    def env_overlay(self, params: dict[str, Any]) -> dict[str, str]:
        """Environment this engine needs for these args. servers.build_env."""

    # --- memory ---
    def matches(self, command: list[str] | None) -> bool:
        """Whether this argv is this engine. The budget's admission gate."""

    def argv_of(self, command: list[str] | None) -> list[str]:
        """Tokens, unwrapping a `bash -lc "… && exec <engine> …"` if there is one."""

    def command_params(self, command: list[str] | None) -> dict[str, Any]:
        """A running container's memory-relevant flags, shaped like stored args,
        so a hand-launched engine is accounted exactly like a managed one."""

    async def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        """Whatever pricing needs that costs I/O, folded into the params dict."""

    def footprint_bytes(self, params: dict[str, Any], total_bytes: int) -> int:
        """A floor on the accelerator memory these params will take."""

    def declared_util(self, params: dict[str, Any]) -> float | None:
        """The utilisation fraction the operator actually wrote, or None when the
        engine has no such concept. Never a fraction derived from bytes: the Util
        column would then look summable across engines when it is not."""

    def implicit_util(self) -> float | None:
        """What the engine takes when nothing was declared, as a fraction."""

    def notes(self, params: dict[str, Any], *, implicit: bool) -> list[str]:
        """Per-tenant prose for the memory panel."""

    # --- discovery and status ---
    def model_from_argv(self, argv: list[str] | None) -> str:
        """The model a foreign container is serving."""

    def is_loading(self, *, reachable: bool, healthy: bool, **_extra: Any) -> bool:
        """Whether a running-but-not-healthy container is still coming up.

        The two engines are exact inverses here: vLLM binds no port until the
        weights are read and the graphs captured, so unreachable means loading;
        llama-server binds first and answers /health with 503 while it loads, so
        reachable-but-unhealthy is what loading looks like."""


# --- registry ------------------------------------------------------------

DEFAULT = "vllm"


def _load() -> tuple[dict[str, Engine], tuple[Engine, ...]]:
    from app.engines import llamacpp as _llamacpp
    from app.engines import vllm as _vllm

    # vLLM first, and the order is load-bearing rather than alphabetical:
    # recognise() returns the first engine that claims an argv, so anything
    # already recognised keeps the answer it has always had.
    order = (_vllm.ENGINE, _llamacpp.ENGINE)
    return {engine.name: engine for engine in order}, order


_REGISTRY: dict[str, Engine] | None = None
_ORDER: tuple[Engine, ...] = ()


def registry() -> dict[str, Engine]:
    global _REGISTRY, _ORDER
    if _REGISTRY is None:
        _REGISTRY, _ORDER = _load()
    return _REGISTRY


def order() -> tuple[Engine, ...]:
    registry()
    return _ORDER


def names() -> list[str]:
    return [engine.name for engine in order()]


def get(name: str | None) -> Engine:
    """The engine by name, falling back to vLLM.

    Never raises. A row carrying an engine this build does not have — a
    downgrade, a hand-edited database — must not 500 the Serve page; it renders
    as a vLLM server, which is what it was before the column existed.
    """
    return registry().get(str(name or "").strip().lower()) or registry()[DEFAULT]


def known(name: str | None) -> bool:
    return str(name or "").strip().lower() in registry()


def of(server: dict[str, Any] | None) -> Engine:
    """The engine of a server row."""
    return get((server or {}).get("engine"))


def recognise_argv(argv: list[str] | None) -> Engine | None:
    """Which engine, if any, this argv is running.

    Asked instead of widening any single engine's `matches()`. Each engine
    unwraps the argv its own way first, because a shell-wrapped launch is one
    opaque token until somebody who knows the sentinel splits it.
    """
    if not argv:
        return None
    for engine in order():
        if engine.matches(argv):
            return engine
    return None


def recognise(state: Any) -> Engine | None:
    """Which engine a container is running, from its inspected state."""
    engine, _argv = identify(state)
    return engine


def identify(state: Any) -> tuple[Engine | None, list[str]]:
    """Which engine a container is running, AND the argv that says so.

    Both, together, and that is not convenience. `state.command` is consulted
    before `state.argv` on purpose: `command` is docker's `Config.Cmd` and is
    what every parser in this codebase has always read, while `argv` prepends
    `Config.Entrypoint` and exists only so an image that puts the binary in its
    ENTRYPOINT — which the upstream llama.cpp images do, leaving bare flags in
    Cmd — is visible at all. Trying `command` first means the fallback can only
    ever add a container to the picture, never move one.

    But a caller that then reads the flags out of `command` alone would find
    nothing for exactly the containers the fallback exists for, and price them
    at zero. So the matching argv comes back with the engine, and every caller
    reads its flags from that.
    """
    if state is None:
        return None, []
    command = getattr(state, "command", None)
    found = recognise_argv(command)
    if found is not None:
        return found, list(command or [])
    argv = getattr(state, "argv", None)
    found = recognise_argv(argv)
    return (found, list(argv or [])) if found is not None else (None, [])


__all__ = [
    "DEFAULT",
    "Engine",
    "as_float",
    "as_int",
    "flag_present",
    "flag_value",
    "get",
    "identify",
    "known",
    "names",
    "of",
    "order",
    "recognise",
    "recognise_argv",
    "registry",
]
