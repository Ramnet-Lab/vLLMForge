#!/usr/bin/env bash
# BUILD.sh — stand the dashboard up on a machine nobody has prepared.
#
#   ./BUILD.sh                 install everything, asking before the slow parts
#   ./BUILD.sh --yes           never ask
#   ./BUILD.sh --help          the full option list
#
# It installs what the application cannot install for itself — python, docker,
# the NVIDIA container toolkit — and then hands over to scripts/setup.sh and
# scripts/install-service.sh, which own everything above that line.
#
# Safe to re-run: every step checks before it acts, and a second run should
# report that there is nothing to do.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

ASSUME_YES=0
WITH_SERVICE=1
WITH_GPU=auto
WITH_SUDOERS=1
SETUP_ARGS=()

SUDOERS_FILE=/etc/sudoers.d/99-llmd-docker

# --- output --------------------------------------------------------------

if [ -t 1 ]; then
    B=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; O=$'\033[0m'
else
    B=""; DIM=""; RED=""; YEL=""; GRN=""; O=""
fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$O"; }
note() { printf '    %s\n' "$*"; }
good() { printf '    %s✓%s %s\n' "$GRN" "$O" "$*"; }
warn() { printf '    %s!%s %s\n' "$YEL" "$O" "$*"; }
fail() { printf '\n%serror:%s %s\n' "$RED" "$O" "$*" >&2; exit 1; }

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  -y, --yes         assume yes; never prompt (required when there is no terminal)
      --with-images also build the optional Heretic and fine-tuning images.
                    The vLLM image is built either way: nothing can be served
                    without it. All three sit on a ~22 GB base image.
      --no-service  do not install the systemd user service
      --no-images   skip the optional worker images (the vLLM image is still
                    built; without it no server can start)
      --no-gpu      skip the NVIDIA container toolkit even if a GPU is present
      --no-sudoers  do not write the passwordless-docker sudoers rule
      --dev         also install the test and lint dependencies
  -h, --help        this text

What it does, in order:
  1. checks it can become root, once
  2. installs python3 + venv, curl and ca-certificates if missing
  3. installs docker if missing, and starts it
  4. puts you in the docker group so docker needs no sudo
  5. installs the NVIDIA container toolkit, if this machine has an NVIDIA GPU
  6. runs scripts/setup.sh      (virtualenv, dependencies, directories, images)
  7. runs scripts/install-service.sh  (systemd user service)
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes)     ASSUME_YES=1 ;;
        --with-images) SETUP_ARGS+=(--with-images) ;;
        --no-service) WITH_SERVICE=0 ;;
        --no-images)  SETUP_ARGS+=(--no-images) ;;
        --no-gpu)     WITH_GPU=no ;;
        --no-sudoers) WITH_SUDOERS=0 ;;
        --dev)        SETUP_ARGS+=(--dev) ;;
        -h|--help)    usage; exit 0 ;;
        *) fail "unknown option: $1 (try --help)" ;;
    esac
    shift
done

# `test -r /dev/tty` is not a terminal test: /dev/tty is mode 0666 and always
# passes it. With no controlling terminal, opening it is what fails.
has_tty() { (exec 3<>/dev/tty) 2>/dev/null; }

confirm() {
    [ "$ASSUME_YES" = 1 ] && return 0
    has_tty || return 1
    local reply
    read -r -p "    $1 [y/N] " reply </dev/tty
    [[ "$reply" =~ ^[Yy] ]]
}

# --- 1. privilege --------------------------------------------------------

step "Checking privileges"

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
    TARGET_USER="${SUDO_USER:-root}"
    good "running as root"
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
    TARGET_USER="$(id -un)"
else
    fail "this needs root. Install sudo, or re-run as root:  su -c '$REPO/BUILD.sh'"
fi

if [ -n "$SUDO" ]; then
    if sudo -n true 2>/dev/null; then
        good "sudo already granted"
    elif has_tty; then
        note "asking for your password once, up front"
        sudo -v || fail "could not obtain sudo"
        # The credential is cached against this terminal. A background keep-alive
        # loop is the usual advice but it is useless without a tty — the ticket
        # is then keyed to the loop's own pid, not to this script — so it only
        # runs where it actually helps.
        ( while true; do sleep 50; sudo -n true 2>/dev/null || exit; done ) &
        SUDO_KEEPALIVE=$!
        trap 'kill "${SUDO_KEEPALIVE:-}" 2>/dev/null || true' EXIT
        good "sudo granted"
    else
        fail "no terminal to prompt on. Re-run with a tty, or give this user a NOPASSWD sudo rule."
    fi
fi

TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
[ -n "$TARGET_HOME" ] || TARGET_HOME="$HOME"

# apt on Ubuntu 24.04 ships needrestart, which opens a whiptail service-restart
# prompt that DEBIAN_FRONTEND=noninteractive does not suppress. Left alone it
# hangs an unattended build.
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

# --- 2. the platform -----------------------------------------------------

step "Identifying the machine"

OS_ID=""; OS_LIKE=""; CODENAME=""; PRETTY=""
if [ -r /etc/os-release ]; then
    . /etc/os-release
    OS_ID="${ID:-}"; OS_LIKE="${ID_LIKE:-}"; PRETTY="${PRETTY_NAME:-$OS_ID}"
    case "$OS_ID $OS_LIKE" in
        *ubuntu*) CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}" ;;
        *)        CODENAME="${VERSION_CODENAME:-}" ;;
    esac
fi

case "$OS_ID $OS_LIKE" in
    *debian*|*ubuntu*) PKG=apt ;;
    *rhel*|*fedora*|*centos*) PKG=dnf ;;
    *arch*) PKG=pacman ;;
    *) PKG="" ;;
esac

note "${PRETTY:-unknown OS} on $(uname -m)"
[ -n "$PKG" ] || warn "unrecognised distribution; package installs will be skipped"
if [ -d /run/systemd/system ]; then HAVE_SYSTEMD=1; else HAVE_SYSTEMD=0; warn "no systemd"; fi

pkg_install() {
    [ $# -gt 0 ] || return 0
    case "$PKG" in
        apt)    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "$@" ;;
        dnf)    $SUDO dnf install -y -q "$@" ;;
        pacman) $SUDO pacman -Sy --noconfirm --needed "$@" ;;
        *)      fail "cannot install $* on this distribution; install them by hand and re-run" ;;
    esac
}

# --- 3. base packages ----------------------------------------------------

step "Base packages"

missing=()
command -v curl >/dev/null 2>&1 || missing+=(curl)
[ -r /etc/ssl/certs/ca-certificates.crt ] || [ "$PKG" != apt ] || missing+=(ca-certificates)
command -v git >/dev/null 2>&1 || missing+=(git)
# gpg --dearmor is how both the docker and the NVIDIA keyrings are installed,
# and a base Ubuntu image does not have it.
command -v gpg >/dev/null 2>&1 || [ "$PKG" != apt ] || missing+=(gnupg)

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    case "$PKG" in
        apt) missing+=(python3 python3-venv) ;;
        dnf) missing+=(python3) ;;
        pacman) missing+=(python) ;;
    esac
elif ! "$PYTHON" -c 'import venv, ensurepip' 2>/dev/null; then
    # Debian and Ubuntu split venv out of the interpreter, and the error you get
    # without it points at ensurepip rather than at the missing package.
    case "$PKG" in
        apt) missing+=("python3-venv") ;;
        dnf) missing+=(python3-virtualenv) ;;
    esac
fi

if [ ${#missing[@]} -gt 0 ]; then
    note "installing: ${missing[*]}"
    pkg_install "${missing[@]}"
    good "installed ${missing[*]}"
else
    good "curl, git and a usable python3 are present"
fi

command -v "$PYTHON" >/dev/null 2>&1 || fail "python3 still not on PATH"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "$($PYTHON -V) is too old; this needs 3.11 or newer"
note "$("$PYTHON" -V)"

# --- 4. docker -----------------------------------------------------------

step "Docker"

# `docker info` exits 1 for both "daemon down" and "permission denied", so the
# exit code cannot tell them apart. The message can.
docker_state() {
    command -v docker >/dev/null 2>&1 || { echo MISSING; return; }
    local err
    if err=$(docker info 2>&1 >/dev/null); then echo OK; return; fi
    case "$err" in
        *"permission denied"*)                      echo NEEDS_GROUP ;;
        *"Is the docker daemon running"*|*"Cannot connect"*|\
        *"connection refused"*|*"no such file or directory"*) echo DAEMON_DOWN ;;
        *) echo OTHER ;;
    esac
}

install_docker() {
    case "$PKG" in
        apt)
            local distro=debian
            case "$OS_ID $OS_LIKE" in *ubuntu*) distro=ubuntu ;; esac
            [ -n "$CODENAME" ] || fail "cannot determine the release codename for the docker repo"
            $SUDO install -m 0755 -d /etc/apt/keyrings
            $SUDO curl -fsSL "https://download.docker.com/linux/$distro/gpg" \
                 -o /etc/apt/keyrings/docker.asc
            $SUDO chmod a+r /etc/apt/keyrings/docker.asc
            # The current docs use deb822; get.docker.com writes the old one-line
            # format. Having both makes apt complain about duplicate sources.
            $SUDO rm -f /etc/apt/sources.list.d/docker.list
            $SUDO tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/$distro
Suites: $CODENAME
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
            $SUDO apt-get update -qq
            $SUDO apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
                 docker-buildx-plugin docker-compose-plugin
            ;;
        dnf)
            local d=rhel; [ "$OS_ID" = fedora ] && d=fedora
            $SUDO dnf install -y -q dnf-plugins-core
            $SUDO dnf config-manager addrepo --overwrite \
                 --from-repofile="https://download.docker.com/linux/$d/docker-ce.repo" 2>/dev/null \
              || $SUDO dnf config-manager --add-repo \
                 "https://download.docker.com/linux/$d/docker-ce.repo"
            $SUDO dnf install -y -q docker-ce docker-ce-cli containerd.io \
                 docker-buildx-plugin docker-compose-plugin
            ;;
        pacman) $SUDO pacman -Sy --noconfirm --needed docker docker-buildx docker-compose ;;
        *) fail "install docker by hand on this distribution, then re-run" ;;
    esac
}

start_docker() {
    if [ "$HAVE_SYSTEMD" = 1 ]; then
        $SUDO systemctl enable --now docker.service 2>/dev/null || $SUDO systemctl start docker
    elif command -v service >/dev/null 2>&1 && $SUDO service docker start >/dev/null 2>&1; then
        :
    else
        $SUDO sh -c 'nohup dockerd >/var/log/dockerd.log 2>&1 &'
    fi
    for _ in $(seq 1 30); do
        $SUDO docker info >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

STATE="$(docker_state)"
case "$STATE" in
    OK)
        good "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null) is installed and usable"
        note "not touching it — on some systems (DGX included) docker comes from a"
        note "vendor repo that apt pins, and reinstalling is a confusing no-op at best"
        ;;
    MISSING)
        note "not installed"
        install_docker
        start_docker || fail "docker installed but the daemon would not start; see: journalctl -u docker"
        good "installed docker $($SUDO docker version --format '{{.Server.Version}}' 2>/dev/null)"
        ;;
    DAEMON_DOWN)
        note "installed but not running"
        start_docker || fail "could not start the docker daemon; see: journalctl -u docker"
        good "docker daemon started"
        ;;
    NEEDS_GROUP) note "installed and running, but this user cannot reach it yet" ;;
    *)           warn "docker is in an unexpected state; continuing" ;;
esac

# --- 5. group and sudoers ------------------------------------------------

step "Letting $TARGET_USER use docker"

if [ "$TARGET_USER" = root ]; then
    good "running as root; no group needed"
elif id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx docker; then
    good "$TARGET_USER is already in the docker group"
else
    $SUDO groupadd -f docker
    $SUDO usermod -aG docker "$TARGET_USER"
    good "added $TARGET_USER to the docker group"
    warn "group membership does not apply to shells that are already open"
    REGROUPED=1
fi

# The request was to "add docker to sudoers". Group membership above is what
# actually makes `docker` work without sudo; this drop-in additionally makes
# `sudo docker` not prompt. Both are root-equivalent — anyone who can run a
# container can bind-mount / — so this grants nothing the group did not already.
render_sudoers() {
    cat <<EOF
# Installed by $REPO/BUILD.sh
#
# Lets $TARGET_USER run docker without a password prompt. This is equivalent to
# passwordless root: a container can bind-mount the whole filesystem. Membership
# of the docker group already carries the same power; this only removes the
# prompt from 'sudo docker'.
$TARGET_USER ALL=(ALL) NOPASSWD: $(command -v docker || echo /usr/bin/docker)
EOF
}

if [ "$WITH_SUDOERS" = 1 ] && [ "$TARGET_USER" != root ]; then
    TMP_SUDOERS="$(mktemp)"
    render_sudoers > "$TMP_SUDOERS"
    # visudo -c needs no privilege, so the file is proved good before root is
    # ever asked to install it. A drop-in whose name contains a dot, or ends in
    # a tilde, is silently ignored — hence 99-llmd-docker with no extension.
    if ! visudo -c -f "$TMP_SUDOERS" >/dev/null 2>&1; then
        rm -f "$TMP_SUDOERS"
        warn "generated an invalid sudoers rule; skipping it"
    elif $SUDO test -f "$SUDOERS_FILE" \
         && [ "$($SUDO cat "$SUDOERS_FILE" | sha256sum)" = "$(sha256sum < "$TMP_SUDOERS")" ] \
         && [ "$($SUDO stat -c '%a %U %G' "$SUDOERS_FILE")" = "440 root root" ]; then
        rm -f "$TMP_SUDOERS"
        good "$SUDOERS_FILE is already correct"
    else
        $SUDO install -m 0440 -o root -g root "$TMP_SUDOERS" "$SUDOERS_FILE"
        rm -f "$TMP_SUDOERS"
        if $SUDO visudo -c >/dev/null 2>&1; then
            good "wrote $SUDOERS_FILE"
            note "this is root-equivalent, like the docker group itself — remove it with:"
            note "  sudo rm $SUDOERS_FILE"
        else
            $SUDO rm -f "$SUDOERS_FILE"
            warn "the sudo configuration did not validate afterwards; rolled the rule back"
        fi
    fi
fi

# The rest of this script uses docker. `newgrp` does nothing in a script — it
# starts an interactive shell that is never read — so re-exec under `sg`, which
# does work non-interactively because the membership is already in /etc/group.
if [ "${REGROUPED:-0}" = 1 ] && [ -z "${LLMD_BUILD_REGROUPED:-}" ] && command -v sg >/dev/null 2>&1; then
    step "Re-entering the build with the docker group applied"
    export LLMD_BUILD_REGROUPED=1
    exec sg docker -c "$(printf '%q ' "$REPO/BUILD.sh" "$@")"
fi

# --- 6. NVIDIA container toolkit -----------------------------------------

have_gpu() {
    grep -qs 0x10de /sys/bus/pci/devices/*/vendor && return 0
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}
have_driver() { [ -e /proc/driver/nvidia/version ] || [ -e /dev/nvidiactl ]; }
have_toolkit() { command -v nvidia-ctk >/dev/null 2>&1; }
# The {{if index .Runtimes "nvidia"}} form prints "yes" even when the daemon is
# unreachable. The json form prints null and exits non-zero, which is the truth.
docker_has_nvidia() { docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; }

setup_gpu() {
    if [ "$WITH_GPU" = no ]; then
        note "skipped (--no-gpu)"
    elif ! have_gpu; then
        good "no NVIDIA GPU here — skipping"
        note "the dashboard still runs: the web UI, model downloads and the cache all"
        note "work without a GPU. Only serving and training need one."
    elif ! have_driver; then
        warn "an NVIDIA GPU is present but no driver is loaded"
        note "install the driver for your distribution, reboot, and re-run this script"
    elif docker_has_nvidia; then
        good "docker already has the nvidia runtime registered"
        note "not restarting the daemon — that would kill any container currently running"
    else
        note "GPU and driver present, but docker cannot use them yet"
        if ! have_toolkit; then
            case "$PKG" in
                apt)
                    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                      | $SUDO gpg --batch --yes --dearmor \
                          -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                      | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
                    $SUDO apt-get update -qq
                    $SUDO apt-get install -y -qq nvidia-container-toolkit
                    ;;
                dnf)
                    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
                      | $SUDO tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
                    $SUDO dnf install -y -q nvidia-container-toolkit
                    ;;
                *) warn "install nvidia-container-toolkit by hand on this distribution"; return 1 ;;
            esac
            good "installed the NVIDIA container toolkit"
        fi
        $SUDO nvidia-ctk runtime configure --runtime=docker
        # Docker 29 resolves --gpus through CDI. The toolkit's systemd path unit
        # normally generates the spec at boot; where there is no systemd it has
        # to be done here or --gpus all fails with "no known GPU vendor found".
        $SUDO nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >/dev/null 2>&1 || true
        if [ "$HAVE_SYSTEMD" = 1 ]; then $SUDO systemctl restart docker; else start_docker; fi
        docker_has_nvidia || return 1
        good "docker can now use the GPU"
    fi
}

step "GPU support"
# Never fatal. A machine whose GPU plumbing needs attention should still end up
# with a working dashboard — downloads, the cache and the whole UI need no GPU.
setup_gpu || {
    warn "could not finish GPU setup; continuing without it"
    note "serving and training will not work until docker can reach the GPU."
    note "re-run this script once the driver and toolkit are sorted."
}

# --- 7. the application --------------------------------------------------

step "Handing over to scripts/setup.sh"
note "virtualenv, dependencies, directories, database, worker images"
"$REPO/scripts/setup.sh" ${SETUP_ARGS[@]+"${SETUP_ARGS[@]}"}

if [ "$WITH_SERVICE" = 1 ]; then
    step "Installing the service"
    if [ "$HAVE_SYSTEMD" = 1 ] && [ "$TARGET_USER" != root ]; then
        "$REPO/scripts/install-service.sh" || warn "the service did not come up; see: journalctl --user -u llm-dashboard"
    else
        warn "skipping: a systemd *user* service needs systemd and a non-root user"
        note "start it in the foreground instead:  $REPO/scripts/run.sh"
    fi
fi

# --- 8. done -------------------------------------------------------------

PORT="$(cd "$REPO" && ./.venv/bin/python -c 'from app.config import settings; print(settings.port)' 2>/dev/null || echo 8700)"

step "Done"
cat <<DONE
    Open:     http://localhost:$PORT/
    Service:  systemctl --user status llm-dashboard
    Logs:     journalctl --user -u llm-dashboard -f
    Manually: $REPO/scripts/run.sh

    Read docs/MEMORY.md before serving a model. On a unified-memory machine the
    GPU allocation comes out of host RAM, and getting it wrong locks the box.
DONE

if [ "${REGROUPED:-0}" = 1 ]; then
    echo
    warn "log out and back in (or run: newgrp docker) before using docker in your own shell"
fi
