#!/bin/sh
# Normalise two bases that disagree about what an image's entrypoint is for.
#
# The dashboard builds a complete `vllm serve ...` argv and passes it as the
# container command, so an entrypoint that prepends a command of its own turns
# it into nonsense. The official vLLM image's is ["vllm","serve"], which yields
#   vllm serve vllm serve <model>   ->   error: unrecognized arguments
#
# NGC's entrypoint is not that kind: /opt/nvidia/nvidia_entrypoint.sh does the
# container's NVIDIA setup and then execs whatever it was handed. Clearing it
# would silently drop that setup on the machine this repo was built for, so it
# is kept wherever it exists and only the command-prepending kind is dropped.
set -e
if [ -x /opt/nvidia/nvidia_entrypoint.sh ]; then
    exec /opt/nvidia/nvidia_entrypoint.sh "$@"
fi
exec "$@"
