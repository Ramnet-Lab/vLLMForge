"""What a model on this disk actually is, read from the files beside its weights.

A pull already fetches every file in the repo — config.json, tokenizer_config.json,
generation_config.json, chat_template.jinja and the rest all land in the cache
next to the safetensors. Nothing read them, so the Serve page asked the operator
to supply from memory what was sitting on disk the whole time.

This module is the reading half only: it turns a model reference into a flat,
boring record of facts. Deciding what those facts imply for vLLM's flags belongs
somewhere else, because a fact stays true and a recommendation does not.

Two shapes of model exist here:

* a Hub repo in the shared cache, `<hub>/models--org--repo/snapshots/<sha>/`,
  where the snapshot directory is a tree of symlinks into `blobs/`;
* a plain directory, which is what the Fine-tune and Heretic tabs write under
  `outputs/` — config.json sits at the top and there is no snapshot indirection.

Both are read the same way once the directory is resolved.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import shlex
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import settings

log = logging.getLogger("llmd.profile")

# Read caps. A config.json is a few KB; anything wildly larger is not a config
# and must not be pulled into memory just because it has the right name.
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_TEMPLATE_BYTES = 1 * 1024 * 1024

# Files that carry a chat template outside tokenizer_config.json. Order matters:
# it is the order vLLM and transformers look in.
TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json")

# Markers that identify which parser a template expects. vLLM picks its tool
# and reasoning parsers by name, and the name is not written down anywhere in
# the repo — but the tokens the template emits are, and each parser keys on its
# own. Only the presence of these is recorded; the template body never leaves
# the disk read.
TEMPLATE_MARKERS = (
    "<|tool_call>",              # gemma4 tool calls
    "<tool_call>",               # hermes / qwen3 family
    "<function=",                # qwen3 xml
    "<parameter=",               # qwen3 xml
    "[TOOL_CALLS]",              # mistral
    "<|python_tag|>",            # llama3 json
    "<|tool_call|>",             # granite
    "<tool_calls>",              # jamba
    "<|tool_calls_section_begin|>",  # kimi k2
    "<|channel>",                # gemma4 reasoning
    "<think>",
    "</think>",
    "<seed:think>",
    "<mm:think>",
    "<|START_THINKING|>",
    "if not enable_thinking",    # the template hides reasoning unless asked
)

# A repo with any of these is doing more than text.
MULTIMODAL_FILES = ("preprocessor_config.json", "processor_config.json",
                    "image_processor_config.json", "video_preprocessor_config.json")

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")

# Layer kinds, by what they cost to remember a context.
FULL_ATTENTION = "full_attention"
# A recurrent layer keeps a fixed-size state per sequence instead of a cache
# that grows with the context, so it contributes nothing per token. Its state
# is real memory, but it is sized by max_num_seqs and not by context length.
RECURRENT_KINDS = ("linear_attention", "mamba", "mamba2", "recurrent", "ssm")

# What a cache directory may be called. The name is interpolated into a remote
# shell command, so it is matched rather than escaped: nothing but this shape
# reaches the peer.
_CACHE_NAME = re.compile(r"models--[A-Za-z0-9._-]+(--[A-Za-z0-9._-]+)*")

# Older architectures spell the same shape differently — GPT-2 and its
# descendants use n_layer where Llama-style configs use num_hidden_layers. The
# fallbacks are tried in order, after the canonical name.
CONFIG_ALIASES = {
    "max_position_embeddings": ("n_positions", "n_ctx", "seq_length", "max_seq_len"),
    "num_hidden_layers": ("n_layer", "n_layers", "num_layers"),
    "hidden_size": ("n_embd", "d_model"),
    "num_attention_heads": ("n_head", "n_heads"),
    "num_key_value_heads": ("n_head_kv", "num_kv_heads", "multi_query_group_num"),
    "head_dim": ("attention_head_size", "kv_channels"),
    "sliding_window": ("sliding_window_size", "attention_window_size"),
}


@dataclass
class Profile:
    """Everything readable about a model without loading it."""

    reference: str
    """What the caller asked for — a Hub id or a path."""

    found: bool = False
    path: str = ""
    source: str = "missing"
    """Where this came from: `cache`, `directory`, or `missing`."""

    # --- config.json ------------------------------------------------------
    architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    dtype: str = ""
    max_position_embeddings: int | None = None
    rope_scaling: dict[str, Any] | None = None
    """The rope block, under whichever of the two names the repo used."""
    rope_kind: str = ""
    """A one-word summary — `default`, `yarn`, `proportional`, `mixed`."""
    rope_theta: float | None = None
    quantization: dict[str, Any] | None = None
    quant_method: str = ""
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int | None = None
    sliding_window: int | None = None
    layer_types: dict[str, int] = field(default_factory=dict)
    """How many layers of each attention kind, when the config says. A hybrid
    model's sliding layers cost a fixed amount of KV rather than growing with
    the context, and getting that wrong overstates a Gemma-class model by 10x."""
    global_kv_heads: int | None = None
    global_head_dim: int | None = None
    """Full-attention layers can have their own head geometry."""
    tie_word_embeddings: bool | None = None
    num_experts: int | None = None
    requires_remote_code: bool = False
    """config.json carries an auto_map, so the modelling code lives in the repo."""

    is_multimodal: bool = False
    is_adapter: bool = False
    base_model: str = ""

    supported: bool | None = None
    """Whether this build of vLLM registers the architecture. None when there is
    no architecture to check, or no generated registry to check it against."""

    runner: str = "generate"
    """How vLLM will run this: `generate` or `pooling`. Not a free choice — it
    is resolved from the repo, and a model that resolves to `pooling` serves
    /v1/embeddings and refuses /v1/chat/completions no matter what is asked."""
    runner_reason: str = ""

    # --- tokenizer --------------------------------------------------------
    tokenizer_class: str = ""
    model_max_length: int | None = None
    chat_template: bool = False
    chat_template_source: str = ""
    """`tokenizer_config.json`, a file name, or empty when there is none."""
    template_markers: list[str] = field(default_factory=list)
    """Which of TEMPLATE_MARKERS the template contains. What a parser keys on."""
    eos_token: str = ""
    bos_token: str = ""

    # --- generation_config.json -------------------------------------------
    generation: dict[str, Any] = field(default_factory=dict)

    # --- the files themselves ---------------------------------------------
    files: list[str] = field(default_factory=list)
    weight_bytes: int = 0
    """What the engine will actually resident. The shard index is authoritative
    where it exists: a repo can ship tensors vLLM skips — speculative-decoding
    heads, for instance — which count on disk and never reach memory."""
    disk_bytes: int = 0
    parameters: int | None = None
    shard_count: int = 0
    has_safetensors: bool = False
    has_gguf: bool = False

    notes: list[str] = field(default_factory=list)
    """Anything read that a caller should be told about, in plain words."""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        # The two numbers every caller derives anyway, computed once here so a
        # browser never has to reimplement the hybrid-layer arithmetic.
        context = self.max_position_embeddings or 0
        payload["kv_bytes_full"] = self.kv_bytes(context) if context else None
        payload["kv_bytes_per_token"] = self.kv_bytes_per_token()
        return payload

    # --- derived arithmetic ------------------------------------------------

    def effective_head_dim(self) -> int | None:
        """head_dim, explicit or implied. Absent when the config says too little."""
        if self.head_dim:
            return self.head_dim
        if self.hidden_size and self.num_attention_heads:
            return self.hidden_size // self.num_attention_heads
        return None

    def _per_layer_bytes(self, kind: str, kv_dtype_bytes: int) -> int | None:
        """KV bytes one token costs in one layer of this kind.

        Two tensors, keys and values, one entry per KV head. Grouped query
        attention is exactly why this reads num_key_value_heads and not
        num_attention_heads: the cache is sized by the smaller number. A
        full-attention layer in a hybrid model may carry its own geometry.
        """
        if kind == "full_attention" and self.global_kv_heads and self.global_head_dim:
            heads, dim = self.global_kv_heads, self.global_head_dim
        else:
            heads = self.num_key_value_heads or self.num_attention_heads
            dim = self.effective_head_dim()
        if not (heads and dim):
            return None
        return 2 * heads * dim * kv_dtype_bytes

    def kv_bytes(self, tokens: int, kv_dtype_bytes: int = 2) -> int | None:
        """Total KV cache for one sequence of `tokens`, across every layer.

        A sliding-window layer never holds more than its window, so on a model
        that is mostly sliding layers — every Gemma-class model here — the cost
        of a long context is carried by the handful of full-attention layers
        alone. Treating all sixty layers as full is how a 20 GiB cache gets
        reported as 240 GiB.
        """
        if not self.num_hidden_layers or tokens <= 0:
            return None

        kinds = self.layer_types or {FULL_ATTENTION: self.num_hidden_layers}
        total = 0
        for kind, count in kinds.items():
            if kind in RECURRENT_KINDS:
                continue
            per_token = self._per_layer_bytes(kind, kv_dtype_bytes)
            if per_token is None:
                return None
            held = tokens
            if kind != FULL_ATTENTION and self.sliding_window:
                held = min(tokens, self.sliding_window)
            total += count * per_token * held
        return total

    def kv_bytes_per_token(self, kv_dtype_bytes: int = 2) -> int | None:
        """What one more token of context costs once the windows are full.

        The marginal rate, not the average: sliding layers stop growing, so only
        the full-attention layers keep charging per token.
        """
        kinds = self.layer_types or {FULL_ATTENTION: self.num_hidden_layers or 0}
        total = 0
        for kind, count in kinds.items():
            if kind in RECURRENT_KINDS:
                continue
            if kind != FULL_ATTENTION and self.sliding_window:
                continue
            per_token = self._per_layer_bytes(kind, kv_dtype_bytes)
            if per_token is None:
                return None
            total += count * per_token
        return total or None


# The small files worth carrying across an ssh connection. Everything else in a
# repo is weights, and none of it changes what the flags should be.
READABLE = ("config.json", "tokenizer_config.json", "generation_config.json",
            "adapter_config.json", "modules.json", "model.safetensors.index.json",
            *TEMPLATE_FILES)


@dataclass
class Snapshot:
    """A model directory reduced to what profiling needs: what is in it, how big
    each file is, and the text of the few small files that describe it.

    Having this in between means the parsing below never touches a filesystem,
    so a peer's cache — which can only be read a command at a time over ssh —
    profiles through exactly the same code as this machine's."""

    path: str = ""
    files: dict[str, int] = field(default_factory=dict)
    """name -> size in bytes. A directory is recorded as -1."""
    texts: dict[str, str] = field(default_factory=dict)

    def json(self, name: str) -> dict[str, Any]:
        body = self.texts.get(name)
        if not body:
            return {}
        try:
            value = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            log.debug("malformed %s in %s", name, self.path)
            return {}
        return value if isinstance(value, dict) else {}


# --- resolving a reference to a directory ---------------------------------


def _hub_root() -> Path:
    return (settings.hf_cache / "hub").resolve()


def resolve(reference: str) -> tuple[Path | None, str]:
    """Find the directory holding a model's config, and say how it was found.

    Returns `(path, source)`; path is None when nothing on this box matches.
    A reference is either a Hub id (`org/repo`) or an absolute path. Anything
    that escapes the cache root or the configured roots is refused rather than
    followed — this walks directories the operator did not name.
    """
    text = str(reference or "").strip().rstrip("/")
    if not text:
        return None, "missing"

    if text.startswith("/"):
        directory = Path(text)
        try:
            resolved = directory.resolve()
        except OSError:
            return None, "missing"
        if resolved.is_dir() and (resolved / "config.json").is_file():
            return resolved, "directory"
        return None, "missing"

    if "/" not in text:
        return None, "missing"

    hub = _hub_root()
    name = f"models--{text.replace('/', '--')}"
    repo = hub / name
    try:
        repo = repo.resolve()
    except OSError:
        return None, "missing"
    if repo.parent != hub or not repo.is_dir():
        return None, "missing"

    snapshot = _snapshot_of(repo)
    return (snapshot, "cache") if snapshot else (None, "missing")


def _snapshot_of(repo: Path) -> Path | None:
    """The snapshot a bare model reference resolves to.

    `refs/main` holds the commit a plain `org/repo` means. Without it — a repo
    pulled at a pinned revision, or a partially written cache — the newest
    snapshot is the only honest guess, and one that at least has a config.json
    beats one that does not.
    """
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return None

    ref = repo / "refs" / "main"
    if ref.is_file():
        try:
            sha = ref.read_text().strip()
        except OSError:
            sha = ""
        candidate = snapshots / sha
        if sha and candidate.is_dir():
            return candidate

    dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    if not dirs:
        return None
    with_config = [d for d in dirs if (d / "config.json").is_file()]
    pool = with_config or dirs
    return max(pool, key=lambda d: d.stat().st_mtime)


# --- reading --------------------------------------------------------------


def _int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _token(value: Any) -> str:
    """tokenizer_config spells a special token as a string or as an AddedToken dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content") or "")
    return ""


def read(reference: str) -> Profile:
    """Profile a model reference against this machine's disk."""
    profile = Profile(reference=str(reference or ""))
    directory, source = resolve(reference)
    profile.source = source
    if directory is None:
        return profile
    return _profile(profile, _scan(directory))


def _scan(directory: Path) -> Snapshot:
    """Reduce a directory on this machine to a Snapshot.

    A cached snapshot is symlinks into blobs/, so sizes come from `stat`, which
    follows — `lstat` would report the length of the link text instead of the
    60 GiB it points at.
    """
    snapshot = Snapshot(path=str(directory))
    try:
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
    except OSError:
        return snapshot

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            snapshot.files[entry.name] = -1
            continue
        try:
            snapshot.files[entry.name] = entry.stat().st_size
        except OSError:
            snapshot.files[entry.name] = 0
        if entry.name in READABLE and 0 < snapshot.files[entry.name] <= MAX_CONFIG_BYTES:
            try:
                snapshot.texts[entry.name] = entry.read_text()
            except (OSError, UnicodeDecodeError):
                continue
    return snapshot


def _profile(profile: Profile, snapshot: Snapshot) -> Profile:
    """Turn a Snapshot into a Profile. No filesystem, no network."""
    profile.found = True
    profile.path = snapshot.path
    _read_files(profile, snapshot)
    _read_config(profile, snapshot)
    _read_support(profile)
    _read_runner(profile, snapshot)
    _read_tokenizer(profile, snapshot)
    profile.generation = snapshot.json("generation_config.json")
    return profile


def _read_files(profile: Profile, snapshot: Snapshot) -> None:
    for name, size in snapshot.files.items():
        profile.files.append(f"{name}/" if size < 0 else name)
        if size > 0 and name.endswith(WEIGHT_SUFFIXES):
            profile.disk_bytes += size
            if name.endswith(".safetensors"):
                profile.shard_count += 1
    profile.weight_bytes = profile.disk_bytes

    # The shard index knows exactly what the checkpoint weighs and, sometimes,
    # how many parameters it holds — better than adding up files, which counts
    # anything the loader will skip.
    index = snapshot.json("model.safetensors.index.json").get("metadata") or {}
    total = index.get("total_size")
    if isinstance(total, int) and total > 0:
        profile.weight_bytes = total
    count = index.get("total_parameters")
    if isinstance(count, int) and count > 0:
        profile.parameters = count

    names = set(snapshot.files)
    profile.has_safetensors = any(n.endswith(".safetensors") for n in names)
    profile.has_gguf = any(n.endswith(".gguf") for n in names)
    profile.is_multimodal = any(n in names for n in MULTIMODAL_FILES)
    profile.is_adapter = "adapter_config.json" in names and "config.json" not in names

    if profile.is_adapter:
        adapter = snapshot.json("adapter_config.json")
        profile.base_model = str(adapter.get("base_model_name_or_path") or "")


def _read_config(profile: Profile, snapshot: Snapshot) -> None:
    raw = snapshot.json("config.json")
    if not raw:
        if not profile.is_adapter:
            profile.notes.append("no readable config.json beside the weights")
        return

    # A multimodal repo keeps the language model's shape one level down, and it
    # is the language model that decides context length and KV cache size.
    text = raw.get("text_config") or {}
    if raw.get("vision_config") or raw.get("audio_config") or text:
        profile.is_multimodal = profile.is_multimodal or bool(
            raw.get("vision_config") or raw.get("audio_config"))

    def pick(key: str) -> Any:
        for name in (key, *CONFIG_ALIASES.get(key, ())):
            for level in (raw, text):
                value = level.get(name)
                if value not in (None, "", []):
                    return value
        return None

    profile.architectures = list(pick("architectures") or [])
    if not profile.architectures:
        # transformers infers the class from model_type when architectures is
        # absent; say so rather than reporting an unknown architecture.
        profile.notes.append("config.json names no architecture — inferred from model_type")
    profile.model_type = str(raw.get("model_type") or "")
    profile.dtype = str(pick("torch_dtype") or pick("dtype") or "")
    profile.max_position_embeddings = _int(pick("max_position_embeddings"))
    # Two spellings live side by side in one cache: the legacy flat
    # `rope_scaling`, and `rope_parameters`, which transformers v5 writes and
    # which a hybrid model nests one level deeper, keyed by layer type.
    rope = pick("rope_parameters") or pick("rope_scaling") or None
    profile.rope_scaling = rope if isinstance(rope, dict) else None
    profile.rope_kind = _rope_kind(profile.rope_scaling)
    theta = pick("rope_theta")
    if theta is None and isinstance(profile.rope_scaling, dict):
        theta = profile.rope_scaling.get("rope_theta")
    profile.rope_theta = float(theta) if isinstance(theta, (int, float)) else None
    profile.num_hidden_layers = _int(pick("num_hidden_layers"))
    profile.hidden_size = _int(pick("hidden_size"))
    profile.num_attention_heads = _int(pick("num_attention_heads"))
    # Absent means no grouping: every attention head keeps its own KV.
    profile.num_key_value_heads = (
        _int(pick("num_key_value_heads")) or _int(pick("num_attention_heads")))
    profile.head_dim = _int(pick("head_dim"))
    profile.vocab_size = _int(pick("vocab_size"))
    profile.sliding_window = _int(pick("sliding_window"))
    profile.global_kv_heads = _int(pick("num_global_key_value_heads"))
    profile.global_head_dim = _int(pick("global_head_dim"))
    kinds = pick("layer_types")
    if isinstance(kinds, list) and kinds:
        counts: dict[str, int] = {}
        for kind in kinds:
            counts[str(kind)] = counts.get(str(kind), 0) + 1
        profile.layer_types = counts
        recurrent = sum(n for kind, n in counts.items() if kind in RECURRENT_KINDS)
        if recurrent:
            profile.notes.append(
                f"{recurrent} of {sum(counts.values())} layers are recurrent — they hold a "
                "fixed state per sequence rather than a KV cache, so their memory is set by "
                "how many sequences run at once, not by the context length")
    tied = pick("tie_word_embeddings")
    profile.tie_word_embeddings = bool(tied) if isinstance(tied, bool) else None
    profile.num_experts = _int(
        pick("num_experts") or pick("num_local_experts") or pick("n_routed_experts"))

    quant = raw.get("quantization_config") or text.get("quantization_config")
    if isinstance(quant, dict) and quant:
        profile.quantization = quant
        profile.quant_method = str(
            quant.get("quant_method") or quant.get("format") or "").lower()

    # transformers loads modelling code from the repo when config.json maps it,
    # and vLLM will not do that without being told it may.
    profile.requires_remote_code = bool(raw.get("auto_map") or text.get("auto_map"))


def _rope_kind(rope: dict[str, Any] | None) -> str:
    """One word for what the rope block does.

    A hybrid model keys it by layer type, so several kinds can be in play at
    once and the honest single word for that is `mixed`.
    """
    if not rope:
        return ""
    named = rope.get("rope_type") or rope.get("type")
    if isinstance(named, str) and named:
        return named
    kinds = {
        str(value.get("rope_type") or value.get("type") or "")
        for value in rope.values() if isinstance(value, dict)
    } - {""}
    if not kinds:
        return "scaled"
    return kinds.pop() if len(kinds) == 1 else "mixed"


@lru_cache(maxsize=1)
def supported_architectures() -> frozenset[str]:
    """What this build of vLLM can load, generated from the image itself.

    Empty when the file has not been generated, in which case nothing is
    claimed either way — an unknown answer must not read as a refusal.
    """
    path = settings.data_dir / "vllm_archs.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        log.debug("no generated architecture list at %s", path)
        return frozenset()
    return frozenset(data.get("architectures") or ())


def _read_support(profile: Profile) -> None:
    known = supported_architectures()
    if not known or not profile.architectures:
        return
    profile.supported = any(name in known for name in profile.architectures)
    if not profile.supported:
        profile.notes.append(
            f"this vLLM build does not register {profile.architectures[0]} — it has "
            f"{len(known)} architectures and this is not one of them")


def _read_runner(profile: Profile, snapshot: Snapshot) -> None:
    """Whether vLLM will serve this as a generator or as an embedder.

    This is not an option the operator gets to pick, and it is the difference
    between a chat endpoint and a 400. Two ways a model that looks like a chat
    model comes up as an embedder:

    * sentence-transformers repos ship a modules.json naming a Pooling module,
      and vLLM checks for that BEFORE it looks at the architecture, so a repo
      whose architectures say ...ForCausalLM still resolves to pooling;
    * a config with no `architectures` at all has one synthesised from
      model_type, and the bare `...Model` suffix that produces is itself a
      pooling default.
    """
    modules = snapshot.texts.get("modules.json") or ""
    if '"Pooling"' in modules or "sentence_transformers.models.Pooling" in modules:
        profile.runner = "pooling"
        profile.runner_reason = (
            "modules.json declares a sentence-transformers Pooling module, which vLLM "
            "reads before it looks at the architecture")
        return

    if not profile.architectures and profile.model_type:
        profile.runner = "pooling"
        profile.runner_reason = (
            "config.json names no architecture, so vLLM synthesises a bare "
            f"'{profile.model_type.title().replace('_', '')}Model' name, and a plain "
            "Model suffix defaults to pooling")
        return

    if any(name.endswith("Model") for name in profile.architectures):
        profile.runner = "pooling"
        profile.runner_reason = "a bare ...Model architecture defaults to pooling"


def _read_tokenizer(profile: Profile, snapshot: Snapshot) -> None:
    raw = snapshot.json("tokenizer_config.json")
    if raw:
        profile.tokenizer_class = str(raw.get("tokenizer_class") or "")
        profile.model_max_length = _int(raw.get("model_max_length"))
        profile.eos_token = _token(raw.get("eos_token"))
        profile.bos_token = _token(raw.get("bos_token"))
        if raw.get("chat_template"):
            profile.chat_template = True
            profile.chat_template_source = "tokenizer_config.json"

    # A repo may ship the template as its own file instead; transformers 4.44+
    # writes it that way and vLLM reads it.
    if not profile.chat_template:
        for name in TEMPLATE_FILES:
            size = snapshot.files.get(name, 0)
            if 0 < size <= MAX_TEMPLATE_BYTES:
                profile.chat_template = True
                profile.chat_template_source = name
                break

    if profile.chat_template:
        body = ""
        if profile.chat_template_source == "tokenizer_config.json":
            body = str(raw.get("chat_template") or "")
        else:
            body = snapshot.texts.get(profile.chat_template_source, "")
        if body:
            profile.template_markers = [m for m in TEMPLATE_MARKERS if m in body]

    if not raw and not profile.is_adapter:
        profile.notes.append("no tokenizer_config.json — the tokenizer may not resolve")


# --- a peer's cache --------------------------------------------------------

# One round trip: resolve the snapshot, list it with sizes followed through the
# symlinks, then emit the few small files base64'd so a newline in a jinja
# template cannot be mistaken for the end of a record.
_REMOTE_SCRIPT = r"""
hub={hub}
d="$hub/{directory}"
[ -d "$d" ] || exit 3
snap=""
sha=$(cat "$d/refs/main" 2>/dev/null)
[ -n "$sha" ] && [ -d "$d/snapshots/$sha" ] && snap="$d/snapshots/$sha"
[ -n "$snap" ] || snap=$(ls -1dt "$d"/snapshots/*/ 2>/dev/null | head -1)
[ -n "$snap" ] || exit 3
snap=${{snap%/}}
echo "P|$snap"
find -L "$snap" -maxdepth 1 -mindepth 1 -printf 'F|%y|%s|%f\n' 2>/dev/null
for f in {readable}; do
  [ -f "$snap/$f" ] && echo "T|$f|$(base64 -w0 < "$snap/$f")"
done
exit 0
"""


async def read_remote(reference: str, node: str) -> Profile:
    """Profile a model held in a peer's cache.

    The peer's files are only reachable a command at a time, so the whole scan
    is one script and the result goes through the same parsers as a local read.
    A path reference is refused rather than guessed at: a path means "as the
    container sees it", and the peer's containers are not this machine's.
    """
    from app import nodes

    profile = Profile(reference=str(reference or ""))
    target = nodes.by_name(node)
    if target.is_local:
        return read(reference)

    text = str(reference or "").strip().rstrip("/")
    if not text or "/" not in text or text.startswith("/"):
        profile.notes.append(f"only a Hub id can be profiled on {target.name}")
        return profile

    directory = f"models--{text.replace('/', '--')}"
    if not _CACHE_NAME.fullmatch(directory):
        profile.notes.append("that is not a model id")
        return profile

    script = _REMOTE_SCRIPT.format(
        hub=shlex.quote(f"{settings.hf_cache}/hub"),
        directory=directory,
        readable=" ".join(shlex.quote(name) for name in READABLE),
    )
    code, out = await nodes._ssh(target.name or target.address, script)
    if code == 3:
        profile.notes.append(f"not in {target.name}'s cache")
        return profile
    if code != 0:
        profile.notes.append(f"{target.name} did not answer: {out.strip()[:200]}")
        return profile

    profile.source = "peer"
    return _profile(profile, _parse_remote(out))


def _parse_remote(out: str) -> Snapshot:
    snapshot = Snapshot()
    for line in out.splitlines():
        tag, _, rest = line.partition("|")
        if tag == "P":
            snapshot.path = rest.strip()
        elif tag == "F":
            kind, _, tail = rest.partition("|")
            size, _, name = tail.partition("|")
            if not name or name.startswith("."):
                continue
            snapshot.files[name] = -1 if kind == "d" else (int(size) if size.isdigit() else 0)
        elif tag == "T":
            name, _, body = rest.partition("|")
            if name not in READABLE:
                continue
            try:
                snapshot.texts[name] = base64.b64decode(body).decode("utf-8", "replace")
            except (ValueError, binascii.Error):
                continue
    return snapshot
