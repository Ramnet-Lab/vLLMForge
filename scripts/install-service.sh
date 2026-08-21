#!/usr/bin/env bash
# Install the dashboard as a systemd *user* service, so it starts at boot and
# comes back after a crash.
#
#   scripts/install-service.sh            install, enable and start
#   scripts/install-service.sh --status   show what is installed and running
#   scripts/install-service.sh --uninstall  stop, disable and remove the unit

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_NAME=llm-dashboard.service
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/$UNIT_NAME"
PYTHON="$REPO/.venv/bin/python"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
fail() { printf '\n\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# The unit reads .env itself; the probe below has to agree with it about which
# port it just told systemd to bind.
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi

case "${1:-install}" in
    -h|--help)
        sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    --status)
        systemctl --user status "$UNIT_NAME" --no-pager || true
        exit 0
        ;;
    --uninstall)
        systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
        rm -f "$UNIT"
        systemctl --user daemon-reload
        echo "removed $UNIT"
        echo "note: containers the dashboard started are detached and are still running."
        exit 0
        ;;
    install) ;;
    *) fail "unknown option: $1" ;;
esac

[ -x "$PYTHON" ] || fail "no virtualenv at $REPO/.venv — run scripts/setup.sh first"
systemctl --user is-system-running >/dev/null 2>&1 || \
    fail "no systemd user session (is XDG_RUNTIME_DIR set? try: loginctl enable-linger $USER)"

say "Writing $UNIT"
mkdir -p "$UNIT_DIR"
sed -e "s|@REPO@|$REPO|g" -e "s|@PYTHON@|$PYTHON|g" \
    "$REPO/deploy/llm-dashboard.service.in" > "$UNIT"
note "from deploy/llm-dashboard.service.in"

say "Enabling and starting"
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT_NAME"

# Without lingering the unit stops when the last session for this user ends,
# and does not come back until someone logs in.
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    say "Lingering is off"
    note "The service will stop when you log out and will not start at boot."
    note "Turn it on with:  loginctl enable-linger $USER"
    note "(that may prompt for a password; it is the only privileged step)"
fi

say "Waiting for it to answer"
PORT="$(cd "$REPO" && "$PYTHON" -c 'from app.config import settings; print(settings.port)')"
for _ in $(seq 1 30); do
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
        note "up on http://127.0.0.1:$PORT/"
        systemctl --user --no-pager --lines=0 status "$UNIT_NAME" | head -4
        cat <<USAGE

    systemctl --user status llm-dashboard      what it is doing
    systemctl --user restart llm-dashboard     after changing code
    journalctl --user -u llm-dashboard -f      follow its log

Stopping the service does not stop the models it launched — those are detached
containers and keep serving. It does stop the memory watchdog.
USAGE
        exit 0
    fi
    sleep 1
done

fail "it did not answer on port $PORT — see: journalctl --user -u $UNIT_NAME -n 50"
