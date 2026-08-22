# vLLM plus Ray, for one engine whose memory is pooled across several machines.
#
# The NGC image has no ray at all — the cluster scripts that assumed it would
# have failed on the first `ray start`. vLLM's "mp" executor backend is
# single-host by definition, so multi-node pipeline parallelism needs this.
ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.07-py3
FROM ${BASE_IMAGE}

ARG RAY_VERSION=

# PIP_CONSTRAINT in the NGC image pins numpy and friends; clearing it for this
# install is the same dance the Heretic image does. --no-deps is deliberately
# NOT used: ray[default] needs its own dashboard and gRPC stack.
ENV PIP_CONSTRAINT=""
RUN pip install --no-cache-dir "ray[default]${RAY_VERSION:+==${RAY_VERSION}}" \
 && python -c "import ray, vllm; print('ray', ray.__version__, '/ vllm', vllm.__version__)"

ENV PIP_CONSTRAINT=/etc/pip/constraint.txt
