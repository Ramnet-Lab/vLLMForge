# Unsloth QLoRA on aarch64 / GB10.
#
# The NGC vLLM image already carries working aarch64 builds of xformers, triton
# and flash-attn, which is the only reason nothing here compiles. Unsloth's own
# Dockerfile_DGX_Spark builds both from source against a pytorch base and takes
# ten minutes to fail in interesting ways. The base image is therefore a hard
# dependency of this recipe, not an implementation detail.
ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.07-py3
FROM ${BASE_IMAGE}

# Empty means "whatever pip resolves today". Unsloth ships roughly weekly, so
# pin a known-good set here once you have one and rebuild deliberately;
# 2026.8.19 / unsloth_zoo 2026.8.13 / bitsandbytes 0.50.1 / trl 1.10.0 against
# this base are verified working on GB10.
ARG PEFT_VERSION=
ARG TRL_VERSION=
ARG ACCELERATE_VERSION=
ARG DATASETS_VERSION=
ARG BITSANDBYTES_VERSION=
ARG UNSLOTH_VERSION=
ARG UNSLOTH_ZOO_VERSION=

# --no-deps on unsloth/unsloth_zoo is load-bearing: their pyproject gates
# bitsandbytes and xformers behind x86_64 platform markers and pins xformers to
# x86_64-only wheel URLs, so a resolved install produces a broken aarch64 env.
# NGC's base provides `python`; the official vLLM image ships only `python3`.
# The dashboard and its workers invoke `python` — the download worker runs
# `python -u /worker/hf_download.py`, and a missing binary there surfaces as an
# opaque docker exit 127, "executable file not found in $PATH", with nothing
# about downloads in it. Rather than teach every caller two names, the image is
# made to satisfy the one they use. A no-op wherever `python` already exists.
RUN command -v python >/dev/null || ln -sf "$(command -v python3)" /usr/local/bin/python

RUN pip install --no-cache-dir \
        "peft${PEFT_VERSION:+==${PEFT_VERSION}}" \
        "trl${TRL_VERSION:+==${TRL_VERSION}}" \
        "accelerate${ACCELERATE_VERSION:+==${ACCELERATE_VERSION}}" \
        "datasets${DATASETS_VERSION:+==${DATASETS_VERSION}}" \
        "bitsandbytes${BITSANDBYTES_VERSION:+==${BITSANDBYTES_VERSION}}" \
 && pip install --no-cache-dir --no-deps \
        "unsloth${UNSLOTH_VERSION:+==${UNSLOTH_VERSION}}" \
        "unsloth_zoo${UNSLOTH_ZOO_VERSION:+==${UNSLOTH_ZOO_VERSION}}"

ENV HF_HOME=/hf \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Unsloth compiles kernels into ./unsloth_compiled_cache; keep that in the image
# rather than in the mounted run directory, where it would be mistaken for an
# artefact and would be written as root.
WORKDIR /work

# The base image's entrypoint is the vLLM launcher, which would swallow argv.
ENTRYPOINT []
CMD ["python", "-c", "import unsloth; print(unsloth.__version__)"]
