#!/usr/bin/env bash
# One-shot setup for the LLM dashboard: virtualenv, directories, database, the
# vLLM parameter schema, and — only if you say yes — the two worker images.
#
# Safe to re-run: every step checks before it acts.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
PYTHON="${PYTHON:-python3}"
IMAGES=ask
DEV=0

usage() {
    cat <<'USAGE'
Usage: scripts/setup.sh [options]

  --with-images   build the Heretic and fine-tuning images without asking
  --no-images     never build them (the default when stdin is not a terminal)
  --dev           also install pytest and ruff
  -h, --help      this text

Environment:
  PYTHON          interpreter used to create the venv (default: python3)
  LLMD_*          read from .env if present; see .env.example
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-images) IMAGES=yes ;;
        --no-images) IMAGES=no ;;
        --dev) DEV=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
fail() { printf '\n\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# Anything slow or large is opt-in, and a non-interactive run opts out.
confirm() {
    [ -t 0 ] || return 1
    local reply
    read -r -p "    $1 [y/N] " reply
    [[ "$reply" =~ ^[Yy] ]]
}

# .env is a convenience for the scripts only — app/config.py reads the process
# environment and nothing else, so the values have to be exported here.
if [ -f "$REPO/.env" ]; then
    say "Reading $REPO/.env"
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi

say "Checking the interpreter"
command -v "$PYTHON" >/dev/null || fail "$PYTHON not found; set PYTHON=/path/to/python3.12"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "$($PYTHON -V) is too old; this needs 3.11 or newer"
note "$("$PYTHON" -V) at $(command -v "$PYTHON")"

say "Virtualenv"
if [ -x "$VENV/bin/python" ]; then
    note "reusing $VENV"
else
    "$PYTHON" -m venv "$VENV"
    note "created $VENV"
fi

# --only-binary=:all: is the whole point: this host has no compiler and no sudo
# to add one, so a missing aarch64 wheel must fail here rather than start a build.
say "Dependencies"
note "resolving wheels (a first run downloads about thirty of them)"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --only-binary=:all: -r "$REPO/requirements.txt"
if [ "$DEV" = 1 ]; then
    "$VENV/bin/pip" install --quiet --only-binary=:all: -r "$REPO/requirements-dev.txt"
fi
note "$("$VENV/bin/pip" list --format=freeze | wc -l) packages installed"

say "Directories and database"
(cd "$REPO" && "$VENV/bin/python" - <<'PYEOF'
from app import db
from app.config import settings

# ensure_dirs deliberately leaves the model cache alone: it is shared and
# root-owned, and creating it here would make a typo in LLMD_HF_CACHE look like
# a working install.
settings.ensure_dirs()
db.init_db()
for label, path in (
    ("state", settings.state_dir),
    ("outputs", settings.output_dir),
    ("datasets", settings.dataset_dir),
):
    print(f"    {label:9} {path}")
print(f"    database  {settings.db_path}")
missing = "" if settings.hf_cache.exists() else "   MISSING"
print(f"    cache     {settings.hf_cache}{missing}")
if missing:
    print()
    print("    The model cache does not exist. The dashboard reads it directly and writes")
    print("    to it only from inside containers, so create it before pulling a model —")
    print("    or point LLMD_HF_CACHE at the one you already have.")
PYEOF
)

say "vLLM parameter schema"
if ! command -v docker >/dev/null; then
    note "docker not on PATH — skipping. The checked-in schema still renders the form."
else
    SCHEMA_IMAGE="$(cd "$REPO" && "$VENV/bin/python" -c \
        'import json,pathlib; p=pathlib.Path("app/data/vllm_args.json");
print(json.loads(p.read_text())["image"] if p.exists() else "")')"
    WANT_IMAGE="$(cd "$REPO" && "$VENV/bin/python" -c \
        'from app.config import settings; print(settings.vllm_image)')"
    if [ "$SCHEMA_IMAGE" = "$WANT_IMAGE" ]; then
        note "app/data/vllm_args.json already describes $WANT_IMAGE"
    elif ! docker image inspect "$WANT_IMAGE" >/dev/null 2>&1; then
        note "$WANT_IMAGE is not pulled; keeping the checked-in schema for $SCHEMA_IMAGE."
        note "Pull it and re-run this script to regenerate the form."
    else
        note "schema describes '${SCHEMA_IMAGE:-nothing}', but the configured image is $WANT_IMAGE"
        if confirm "Regenerate it from $WANT_IMAGE now? (two container runs, ~15 seconds)"; then
            (cd "$REPO" && "$VENV/bin/python" tools/gen_vllm_schema.py \
                --image "$WANT_IMAGE" --out app/data/vllm_args.json)
        else
            note "skipped; the Serve form will show the flags of $SCHEMA_IMAGE"
        fi
    fi
fi

say "Worker images"
if ! command -v docker >/dev/null; then
    note "docker not on PATH — the Heretic and fine-tuning tabs will not work."
else
    build_image() {
        local role="$1" tag="$2" dockerfile="$3"
        if docker image inspect "$tag" >/dev/null 2>&1; then
            note "$tag is already built"
            return
        fi
        note "$tag is missing (adds ~0.5 GB on top of the vLLM base image)"
        if [ "$IMAGES" = yes ] || { [ "$IMAGES" = ask ] && confirm "Build $role now?"; }; then
            docker build -t "$tag" -f "$REPO/docker/$dockerfile" "$REPO/docker"
        else
            note "skipped; the $role tab offers a build button that does the same thing"
        fi
    }
    eval "$(cd "$REPO" && "$VENV/bin/python" -c \
        'from app.config import settings
print(f"HERETIC_TAG={settings.heretic_image!r}")
print(f"FINETUNE_TAG={settings.finetune_image!r}")')"
    build_image Heretic "$HERETIC_TAG" heretic.Dockerfile
    build_image fine-tuning "$FINETUNE_TAG" finetune.Dockerfile
fi

PORT="$(cd "$REPO" && "$VENV/bin/python" -c \
    'from app.config import settings; print(settings.port)')"
say "Done"
cat <<NEXT
    Start it:      $REPO/scripts/run.sh
    Then open:     http://localhost:$PORT/
    Watchdog only: $REPO/scripts/memguard.sh

    Before serving anything, read docs/MEMORY.md. GPU memory is host memory on
    this machine and a bad --gpu-memory-utilization locks it up.
NEXT
