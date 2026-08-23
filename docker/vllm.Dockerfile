# The NGC vLLM image, with the xgrammar it actually needs.
#
# nvcr.io/nvidia/vllm:26.07-py3 ships vLLM 0.24.0 alongside xgrammar 0.2.0, and
# they do not agree: vllm/tool_parsers/structural_tag_registry.py imports
# `normalize_tool_choice`, which xgrammar only grew in 0.2.4. The import is
# lazy, so the engine starts, loads a checkpoint and reports itself healthy —
# and then every request carrying `tools` or `tool_choice` comes back as
#
#   500 cannot import name 'normalize_tool_choice' from 'xgrammar'
#
# which is what any OpenAI-compatible client that supports function calling
# sends. The engine looks fine; only the requests fail.
#
# Nothing in a flag or a setting fixes it, so it is fixed in the image.
ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.07-py3
FROM ${BASE_IMAGE}

# The version that has all three symbols this vLLM imports. Pinned rather than
# floated: xgrammar sits on the request path for every structured output, and a
# silent upgrade there is not something to discover from a 500.
ARG XGRAMMAR_VERSION=0.2.4

# --no-deps is load-bearing, not caution. xgrammar 0.2.4 declares an upper bound
# on transformers, and resolving it downgrades the image's 5.6.1 to 4.57.6 and
# huggingface_hub 1.24 to 0.36 — which breaks vLLM itself, so the build fails on
# the very import it was added to fix. Only the compiled xgrammar wheel is
# wanted here; everything it depends on is already in this image at the version
# vLLM was built against.
#
# PIP_CONSTRAINT in the NGC image pins numpy and friends; clearing it for this
# install is the same dance the Heretic image does.
ENV PIP_CONSTRAINT=""
RUN pip install --no-cache-dir --no-deps "xgrammar==${XGRAMMAR_VERSION}" \
 && python -c "\
from xgrammar import StructuralTag, normalize_tool_choice, get_model_structural_tag; \
from vllm.tool_parsers import structural_tag_registry; \
import importlib.metadata as m; \
print('xgrammar', m.version('xgrammar'), '- tool calling imports cleanly')"
ENV PIP_CONSTRAINT=/etc/pip/constraint.txt
