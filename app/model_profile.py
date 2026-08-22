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

import json
import logging
from dataclasses import dataclass, field
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

# A repo with any of these is doing more than text.
MULTIMODAL_FILES = ("preprocessor_config.json", "processor_config.json",
                    "image_processor_config.json", "video_preprocessor_config.json")

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")

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
    tie_word_embeddings: bool | None = None
    num_experts: int | None = None
    requires_remote_code: bool = False
    """config.json carries an auto_map, so the modelling code lives in the repo."""

    is_multimodal: bool = False
    is_adapter: bool = False
    base_model: str = ""

    # --- tokenizer --------------------------------------------------------
    tokenizer_class: str = ""
    model_max_length: int | None = None
    chat_template: bool = False
    chat_template_source: str = ""
    """`tokenizer_config.json`, a file name, or empty when there is none."""
    eos_token: str = ""
    bos_token: str = ""

    # --- generation_config.json -------------------------------------------
    generation: dict[str, Any] = field(default_factory=dict)

    # --- the files themselves ---------------------------------------------
    files: list[str] = field(default_factory=list)
    weight_bytes: int = 0
    shard_count: int = 0
    has_safetensors: bool = False
    has_gguf: bool = False

    notes: list[str] = field(default_factory=list)
    """Anything read that a caller should be told about, in plain words."""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    # --- derived arithmetic ------------------------------------------------

    def effective_head_dim(self) -> int | None:
        """head_dim, explicit or implied. Absent when the config says too little."""
        if self.head_dim:
            return self.head_dim
        if self.hidden_size and self.num_attention_heads:
            return self.hidden_size // self.num_attention_heads
        return None

    def kv_bytes_per_token(self, kv_dtype_bytes: int = 2) -> int | None:
        """How much KV cache one token of context costs, across all layers.

        Two tensors per layer (keys and values), one entry per KV head. Grouped
        query attention is exactly why this reads num_key_value_heads rather
        than num_attention_heads: the KV cache is sized by the smaller number.
        """
        heads = self.num_key_value_heads or self.num_attention_heads
        dim = self.effective_head_dim()
        if not (self.num_hidden_layers and heads and dim):
            return None
        return 2 * self.num_hidden_layers * heads * dim * kv_dtype_bytes


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


def _read_json(directory: Path, name: str) -> dict[str, Any]:
    path = directory / name
    try:
        if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
            return {}
        return json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        # A malformed config is worth knowing about but never worth raising
        # from a read: the caller wants the rest of the profile regardless.
        log.debug("could not read %s", path, exc_info=True)
        return {}


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

    profile.found = True
    profile.path = str(directory)
    _read_files(profile, directory)
    _read_config(profile, directory)
    _read_tokenizer(profile, directory)
    profile.generation = _read_json(directory, "generation_config.json")
    return profile


def _read_files(profile: Profile, directory: Path) -> None:
    """The repo's own file list, and what the weights weigh.

    A snapshot is symlinks into blobs/, so the size has to come from the target
    — `stat` follows, `lstat` would report the length of the link text.
    """
    try:
        entries = sorted(p for p in directory.iterdir() if not p.name.startswith("."))
    except OSError:
        return

    for entry in entries:
        if entry.is_dir():
            profile.files.append(f"{entry.name}/")
            continue
        profile.files.append(entry.name)
        if entry.name.endswith(WEIGHT_SUFFIXES):
            try:
                profile.weight_bytes += entry.stat().st_size
            except OSError:
                continue
            if entry.name.endswith(".safetensors"):
                profile.shard_count += 1

    names = set(profile.files)
    profile.has_safetensors = any(n.endswith(".safetensors") for n in names)
    profile.has_gguf = any(n.endswith(".gguf") for n in names)
    profile.is_multimodal = any(n in names for n in MULTIMODAL_FILES)
    profile.is_adapter = "adapter_config.json" in names and "config.json" not in names

    if profile.is_adapter:
        adapter = _read_json(directory, "adapter_config.json")
        profile.base_model = str(adapter.get("base_model_name_or_path") or "")


def _read_config(profile: Profile, directory: Path) -> None:
    raw = _read_json(directory, "config.json")
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
    profile.rope_scaling = pick("rope_scaling") or None
    theta = pick("rope_theta")
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


def _read_tokenizer(profile: Profile, directory: Path) -> None:
    raw = _read_json(directory, "tokenizer_config.json")
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
            path = directory / name
            try:
                if path.is_file() and 0 < path.stat().st_size <= MAX_TEMPLATE_BYTES:
                    profile.chat_template = True
                    profile.chat_template_source = name
                    break
            except OSError:
                continue

    if not raw and not profile.is_adapter:
        profile.notes.append("no tokenizer_config.json — the tokenizer may not resolve")
