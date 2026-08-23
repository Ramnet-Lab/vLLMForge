ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.07-py3
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
# Headless operation needs git master. The PyPI release (heretic-llm 1.4.0) has
# no --trial-index/--model-action/--save-directory, so it finishes the search
# and then blocks on a questionary menu holding a model it can never save.
# Pin this to a commit SHA when you want a byte-reproducible rebuild.
ARG HERETIC_REF=master

LABEL org.llmd.heretic.ref="${HERETIC_REF}" \
      org.llmd.heretic.base="${BASE_IMAGE}"

# The NVIDIA image ships PIP_CONSTRAINT=/etc/pip/constraint.txt pinning
# numpy<=2.1, which heretic's numpy~=2.2 cannot satisfy. Left in place, pip
# does not fail — it backtracks to heretic-llm 1.2.0 and drags transformers and
# huggingface_hub down with it.
ENV PIP_CONSTRAINT=""

# An overlay venv with --system-site-packages reuses the image's aarch64 CUDA 13
# torch/torchvision/transformers instead of resolving x86-flavoured wheels, and
# keeps heretic's dependency set out of the site-packages vLLM runs from.
# build-ref.json is pip's PEP 610 record of which commit actually got installed,
# which is the only honest answer to "what is in this image".
# NGC's base provides `python`; the official vLLM image ships only `python3`.
# The dashboard and its workers invoke `python` — the download worker runs
# `python -u /worker/hf_download.py`, and a missing binary there surfaces as an
# opaque docker exit 127, "executable file not found in $PATH", with nothing
# about downloads in it. Rather than teach every caller two names, the image is
# made to satisfy the one they use. A no-op wherever `python` already exists.
RUN command -v python >/dev/null || ln -sf "$(command -v python3)" /usr/local/bin/python

# NGC's base carries git; the official vLLM image does not, and Heretic is
# installed from a git URL — pip fails there with "Cannot find command 'git'",
# which says nothing about the base image that lacks it. Installed only when
# missing, so the Spark's build is untouched and pays no apt round trip.
RUN command -v git >/dev/null || { \
        apt-get update && apt-get install -y --no-install-recommends git \
        && rm -rf /var/lib/apt/lists/*; \
    }

RUN python -m venv --system-site-packages /opt/heretic \
 && /opt/heretic/bin/pip install --no-cache-dir \
      "git+https://github.com/p-e-w/heretic@${HERETIC_REF}" \
 && /opt/heretic/bin/heretic --help > /dev/null \
 && { cat /opt/heretic/lib/python*/site-packages/heretic_llm-*.dist-info/direct_url.json \
      > /opt/heretic/build-ref.json || echo '{}' > /opt/heretic/build-ref.json ; }

ENV PATH=/opt/heretic/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    COLUMNS=200 \
    HF_HOME=/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /job
ENTRYPOINT ["heretic"]
