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
  --no-images     skip the OPTIONAL images (default with no terminal). The vLLM
                  image is built regardless; nothing can be served without it.
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

say "vLLM parameter schema and model registry"
if ! command -v docker >/dev/null; then
    note "docker not on PATH — skipping. The checked-in schema still renders the form."
else
    SCHEMA_IMAGE="$(cd "$REPO" && "$VENV/bin/python" -c \
        'import json,pathlib; p=pathlib.Path("app/data/vllm_args.json");
print(json.loads(p.read_text())["image"] if p.exists() else "")')"
    WANT_IMAGE="$(cd "$REPO" && "$VENV/bin/python" -c \
        'from app.config import settings; print(settings.vllm_image)')"
    ARCH_IMAGE="$(cd "$REPO" && "$VENV/bin/python" -c \
        'import json,pathlib; p=pathlib.Path("app/data/vllm_archs.json");
print(json.loads(p.read_text())["image"] if p.exists() else "")')"
    if [ "$SCHEMA_IMAGE" = "$WANT_IMAGE" ] && [ "$ARCH_IMAGE" = "$WANT_IMAGE" ]; then
        note "app/data/vllm_args.json already describes $WANT_IMAGE"
    elif ! docker image inspect "$WANT_IMAGE" >/dev/null 2>&1; then
        note "$WANT_IMAGE does not exist yet; keeping the checked-in schema for $SCHEMA_IMAGE."
        note "It is built a few steps below; re-run this script afterwards to regenerate the form."
    else
        note "schema describes '${SCHEMA_IMAGE:-nothing}', but the configured image is $WANT_IMAGE"
        if confirm "Regenerate them from $WANT_IMAGE now? (three container runs, ~30 seconds)"; then
            (cd "$REPO" && "$VENV/bin/python" tools/gen_vllm_schema.py \
                --image "$WANT_IMAGE" --out app/data/vllm_args.json)
            # The architecture list is what tells the Serve page a model cannot
            # load before it spends four minutes finding out. It goes stale the
            # same way the flag list does, so it is regenerated with it.
            (cd "$REPO" && "$VENV/bin/python" tools/gen_vllm_archs.py \
                --image "$WANT_IMAGE" --out app/data/vllm_archs.json)
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
        local role="$1" tag="$2" dockerfile="$3" required="${4:-no}"
        # Which base to build ON. Defaults to the vLLM base, which is right for
        # three of the four images — they are all layers on top of the same vLLM
        # container. The llama.cpp image is not: its upstream is a llama.cpp
        # server build, and handing it an NGC vLLM base would produce an image
        # with no llama-server in it at all.
        local base="${5:-${VLLM_BASE:-}}"
        local base_arg=""
        [ -n "$base" ] && base_arg="--build-arg BASE_IMAGE=$base"
        if docker image inspect "$tag" >/dev/null 2>&1; then
            note "$tag is already built"
            return
        fi
        note "$tag is missing (adds ~0.5 GB on top of the vLLM base image)"
        # A required image is not a question. Skipping it leaves an install that
        # looks finished and cannot start a single container, and `confirm` says
        # no to everything when stdin is not a terminal — so CI, a piped run and
        # anyone pressing Enter all used to end up there.
        if [ "$required" = yes ] || [ "$IMAGES" = yes ] \
           || { [ "$IMAGES" = ask ] && confirm "Build $role now?"; }; then
            # shellcheck disable=SC2086 -- base_arg is two words or none
            docker build $base_arg -t "$tag" -f "$REPO/docker/$dockerfile" "$REPO/docker" || {
                if [ "$required" = yes ]; then
                    fail "could not build $tag, and nothing can be served without it. Retry with:
    docker build -t $tag -f docker/$dockerfile docker/"
                fi
                note "could not build $tag; the $role tab can retry it"
                return 1
            }
        else
            note "skipped; build it later with:"
            note "  docker build -t $tag -f docker/$dockerfile docker/"
        fi
    }
    eval "$(cd "$REPO" && "$VENV/bin/python" -c \
        'from app.config import settings
print(f"HERETIC_TAG={settings.heretic_image!r}")
print(f"FINETUNE_TAG={settings.finetune_image!r}")
print(f"VLLM_TAG={settings.vllm_image!r}")
print(f"LLAMACPP_TAG={settings.llamacpp_image!r}")
print(f"LLAMACPP_BASE={settings.llamacpp_base_image!r}")')"
    # Which vLLM the images are built on is a property of the machine, not a
    # constant: NGC is right on a Spark and cannot serve a tied-embedding model
    # under 4-bit quantisation anywhere else. app/images.py decides, and says why.
    eval "$(cd "$REPO" && "$VENV/bin/python" -c \
        'import asyncio
from app import images
from app.config import settings
image, why = asyncio.run(images.detect_base_image(settings.vllm_base_image))
print(f"VLLM_BASE={image!r}")
print(f"VLLM_BASE_WHY={why!r}")')"
    note "base image: $VLLM_BASE"
    note "  $VLLM_BASE_WHY"
    # Not optional the way the other two are, so it is not asked about: it is
    # the image every server runs, and without it the NGC base 500s on any
    # request carrying tools. Build it on every node in the cluster — each rank
    # of a pooled engine runs it on its own machine.
    build_image "the vLLM image" "$VLLM_TAG" vllm.Dockerfile yes
    # Optional in exactly the way Heretic and fine-tuning are: a box that only
    # ever serves with vLLM never needs it. Note the fifth argument — this one
    # does NOT sit on the vLLM base.
    build_image llama.cpp "$LLAMACPP_TAG" llamacpp.Dockerfile no "$LLAMACPP_BASE"
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
