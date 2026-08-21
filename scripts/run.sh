#!/usr/bin/env bash
# Start the dashboard in the foreground. Ctrl-C stops it; containers it started
# keep running, which is the point — a model load outlives the web process.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"

[ -x "$VENV/bin/python" ] || {
    echo "no virtualenv at $VENV — run scripts/setup.sh first" >&2
    exit 1
}

# app/config.py reads the process environment, so .env has to be exported here.
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi

cd "$REPO"
"$VENV/bin/python" -c 'from app.config import settings
print(f"llm-dashboard on http://{settings.host}:{settings.port}/  (state: {settings.state_dir})")'

# One worker, no reloader: the telemetry poller, the log followers and the
# memory watchdog are in-process state that a second worker would duplicate.
exec "$VENV/bin/python" -m app.main
