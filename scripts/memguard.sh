#!/usr/bin/env bash
# The host-memory watchdog on its own, for when the dashboard is not running.
#
# It polls MemAvailable every two seconds and, below the threshold, kills the
# running vLLM container with the largest --gpu-memory-utilization and pins its
# restart policy to "no". Nothing else is ever touched. Run it in a second
# terminal while you launch a model by hand.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"

usage() {
    cat <<'USAGE'
Usage: scripts/memguard.sh [--threshold-mib N]

  --threshold-mib N   kill below this much MemAvailable (default: 10240)

The dashboard runs this same watchdog internally, so there is no reason to run
both. LLMD_MEMGUARD_ENABLED=0 disables the built-in one; it does not disable
this script, which only ever runs because you asked it to.
USAGE
}

# .env first, arguments second, so a threshold typed on the command line wins.
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --threshold-mib) export LLMD_MEMGUARD_THRESHOLD_MIB="$2"; shift 2 ;;
        --threshold-mib=*) export LLMD_MEMGUARD_THRESHOLD_MIB="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -x "$VENV/bin/python" ] || {
    echo "no virtualenv at $VENV — run scripts/setup.sh first" >&2
    exit 1
}

cd "$REPO"
exec "$VENV/bin/python" - <<'PY'
import asyncio
import logging

from app import memguard
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info(
    "watching MemAvailable, killing vLLM containers below %d MiB",
    settings.memguard_threshold_mib,
)
try:
    asyncio.run(memguard.watch())
except KeyboardInterrupt:
    logging.info("stopped; nothing is being watched now")
PY
