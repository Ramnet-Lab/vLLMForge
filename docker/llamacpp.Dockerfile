# llama.cpp's server, with the two things the dashboard needs it to stop doing.
#
# The upstream image (ghcr.io/ggml-org/llama.cpp:server-cuda) is a good build and
# is used as the base rather than compiled from source — a CUDA build of ggml is
# twenty minutes of nvcc for a binary somebody already published. What it is not
# is a container the dashboard can drive, for two reasons that are invisible
# until a server has already started:
#
# 1. ENTRYPOINT is ["/app/llama-server"]. The dashboard builds a complete argv
#    and passes it as the container COMMAND, exactly as it does for vLLM, so an
#    entrypoint that prepends a program of its own turns
#      llama-server --host 0.0.0.0 --port 8010 -m /hf/x.gguf
#    into
#      llama-server llama-server --host 0.0.0.0 ...
#    which llama.cpp rejects as an unknown positional. The same shim the vLLM
#    image uses fixes it: docker/entrypoint.sh execs "$@" verbatim.
#
#    Reusing the shim rather than `ENTRYPOINT []` also has a second effect worth
#    having. It keeps the binary's name in `Config.Cmd[0]`, which is where the
#    memory guard and the foreign-container scan look first — so a container
#    this image starts is recognised from its command alone, with no fallback.
#
# 2. HEALTHCHECK curls http://localhost:8080/health. Servers here are launched
#    with --network host on whatever port the operator chose, so on any port but
#    8080 that check can never pass. Docker then reports the container as
#    `unhealthy` forever, and the dashboard renders exactly what docker says —
#    so a perfectly healthy engine shows a red badge that no amount of restarting
#    clears. The dashboard probes /health on the real port itself, which is the
#    only place the real port is known.
#
# The binary is also not on PATH: upstream's Dockerfile sets no ENV PATH entry
# and relies on the absolute path in its entrypoint. Since the argv now starts
# with the program name, it has to be findable.
#
# One thing upstream sets that is deliberately LEFT alone: ENV LLAMA_ARG_HOST=
# 0.0.0.0. The dashboard passes --host explicitly and an explicit flag wins over
# the environment, so it changes nothing — and clearing it would make a bare
# `docker run` of this image bind loopback inside its own namespace, which is
# the less useful default for anyone poking at the image by hand.

ARG BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
LABEL org.llmd.llamacpp.base="${BASE_IMAGE}"

# Not derived from the tag, because a tag moves. `llama-server --version` prints
# the build number and commit the binary was actually made from, and that is the
# only honest answer to "which llama.cpp is this" — which is precisely the
# question app/data/llamacpp_args.json's accuracy depends on.
RUN set -eu; \
    if ! command -v llama-server >/dev/null 2>&1; then \
        found=""; \
        for candidate in /app/llama-server /usr/local/bin/llama-server \
                         /usr/bin/llama-server /llama.cpp/build/bin/llama-server; do \
            [ -x "$candidate" ] && { found="$candidate"; break; }; \
        done; \
        [ -n "$found" ] || { \
            echo "no llama-server binary found in ${BASE_IMAGE}" >&2; exit 1; }; \
        ln -sf "$found" /usr/local/bin/llama-server; \
    fi; \
    mkdir -p /opt/llamacpp; \
    llama-server --version > /opt/llamacpp/build-ref.txt 2>&1 || true; \
    cat /opt/llamacpp/build-ref.txt

# A functional probe, not a version comparison — the same idiom the vLLM image
# uses for xgrammar. llama.cpp does not treat its CLI as a stable API (the
# speculative-decoding flags were renamed wholesale in one release), and the
# dashboard renders a form from a checked-in schema of those flags. If the base
# has drifted past the flags that schema promises, this build fails here rather
# than shipping an image whose form silently emits arguments the binary rejects.
#
# Only the load-bearing few are checked: the ones the launcher itself emits, and
# the ones the memory estimate is computed from. A drift in a sampling default
# is a wrong number in a help string; a drift in -ngl is a wrong launch.
RUN set -eu; \
    help="$(llama-server --help 2>&1)"; \
    for flag in --host --port --metrics --alias --model --hf-repo \
                --n-gpu-layers --ctx-size --cache-type-k --cache-type-v \
                --ubatch-size --parallel --flash-attn --jinja; do \
        printf '%s' "$help" | grep -q -- "$flag" \
            || { echo "this llama.cpp build has no $flag; app/data/llamacpp_args.json \
describes a different one — regenerate it with tools/gen_llamacpp_schema.py" >&2; exit 1; }; \
    done; \
    echo "llama-server accepts every flag the dashboard's schema promises"

# No `python` shim here, unlike the other three images, and the omission is
# deliberate rather than forgotten: nothing runs a worker script in this
# container. Downloads and cache deletions run in LLMD_UTILITY_IMAGE (which
# defaults to the vLLM image), and llama.cpp fetches its own `-hf` references
# with the C++ binary. Adding python would be ~50 MB for nothing.

# Where llama.cpp keeps what it downloads for itself through -hf. Inside the
# mounted cache, so a `-hf` pull survives the container it was made in instead
# of being fetched again on every restart.
ENV LLAMA_CACHE=/hf/llamacpp \
    HF_HOME=/hf

# See the header: upstream's healthcheck is hard-wired to port 8080 and marks
# every server on any other port permanently unhealthy.
HEALTHCHECK NONE

WORKDIR /app
COPY entrypoint.sh /usr/local/bin/llmd-entrypoint
ENTRYPOINT ["/usr/local/bin/llmd-entrypoint"]
# A bare `docker run <image>` prints what this build is, which is how a human
# checks the image without knowing the dashboard's argv.
CMD ["llama-server", "--version"]
