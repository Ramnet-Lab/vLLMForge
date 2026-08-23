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

# A RANGE, not a pin, and the reason is a broken upstream release rather than
# taste. 0.2.4 shipped only 13 files where its neighbours shipped 35: for
# CPython 3.12 it has macOS and linux-aarch64 wheels and NO linux-x86_64 wheel,
# and no sdist to fall back on. So pinning it built fine on aarch64 and failed
# on every x86_64 machine with
#
#   ERROR: Could not find a version that satisfies the requirement
#          xgrammar==0.2.4 (from versions: 0.2.2, 0.2.3, 0.2.6rc1)
#
# which reads like "0.2.4 was never published" and is not: pip's "from versions"
# list only shows releases with an artifact THIS interpreter and platform can
# use. 0.2.4 is on PyPI; it just has no wheel for that box.
#
# The floor is what matters and the floor is 0.2.2 — verified by reading
# xgrammar/__init__.py out of the wheels, every release from 0.2.2 up exports
# all three symbols this vLLM imports, and the 0.2.0 the NGC image ships
# exports none of them. Each platform resolves to the newest wheel it has:
# 0.2.4 on aarch64, 0.2.3 on x86_64. The upper bound keeps a future major out,
# and the import check below is what makes a range safe — a resolution without
# the symbols fails the build instead of shipping an image that 500s.
ARG XGRAMMAR_SPEC=">=0.2.2,<0.3"

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
RUN pip install --no-cache-dir --no-deps "xgrammar${XGRAMMAR_SPEC}" \
 && python3 -c "\
from xgrammar import StructuralTag, normalize_tool_choice, get_model_structural_tag; \
from vllm.tool_parsers import structural_tag_registry; \
import importlib.metadata as m; \
print('xgrammar', m.version('xgrammar'), '- tool calling imports cleanly')"
# NGC ships this file and its contents matter; the official vLLM image has no
# such path, and pointing PIP_CONSTRAINT at a missing file makes every later pip
# in the image fail with "Could not open constraint file". An empty one is a
# no-op constraint, so both bases end up with a valid setting.
RUN [ -f /etc/pip/constraint.txt ] || { mkdir -p /etc/pip && : > /etc/pip/constraint.txt; }
ENV PIP_CONSTRAINT=/etc/pip/constraint.txt

# The last transformers that keeps Gemma 4's config flat. 5.15.0 moved
# global_head_dim / num_global_key_value_heads into a per-layer structure and
# made head_dim itself raise on global access, which vLLM reads directly:
#   AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
# The engine dies in config parsing, before a single weight is read.
ARG TRANSFORMERS_VERSION=5.14.1

# Conditional for the same reason the xgrammar layer is: NGC's base ships 5.6.1,
# which predates the restructure and is already correct, and reinstalling over it
# would be a change made to fix a bug that is not there. The probe is functional
# rather than a version comparison — it asks the config class the actual question
# vLLM will ask it, so a future release that restores the flat keys needs no edit
# here.
RUN set -eu; \
    if python3 -c "\
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig; \
c = Gemma4TextConfig(global_head_dim=512, num_global_key_value_heads=4); \
assert getattr(c, 'global_head_dim', None) == 512; \
assert c.head_dim" >/dev/null 2>&1; then \
        echo "base image's transformers keeps Gemma 4's config flat; not touching it"; \
    else \
        echo "base image's transformers hides Gemma 4's flat config; pinning ${TRANSFORMERS_VERSION}"; \
        PIP_CONSTRAINT="" pip install --no-cache-dir "transformers==${TRANSFORMERS_VERSION}"; \
    fi; \
    python3 -c "\
import transformers, vllm; \
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig; \
c = Gemma4TextConfig(global_head_dim=512, num_global_key_value_heads=4); \
assert getattr(c, 'global_head_dim', None) == 512, 'global_head_dim still hidden'; \
print('transformers', transformers.__version__, '- Gemma 4 config reads flat, vLLM', vllm.__version__)"

# The dashboard sends a complete `vllm serve ...` argv as the container command.
# The two supported bases disagree about entrypoints — NGC's runs NVIDIA setup
# and then execs what it was given, the official image's IS `vllm serve` — so a
# shim keeps the first and drops the second. See docker/entrypoint.sh.
COPY entrypoint.sh /usr/local/bin/llmd-entrypoint
RUN chmod +x /usr/local/bin/llmd-entrypoint
ENTRYPOINT ["/usr/local/bin/llmd-entrypoint"]
