#!/usr/bin/env python3
"""Generate the list of model architectures a vLLM image can actually load.

The Serve page reads a model's `architectures` off its config.json. Whether that
string is one this build knows is a yes/no question the image can answer in a
second, and the alternative is finding out four minutes into a load, from a
traceback. Run it whenever IMAGE changes, beside the flag schema:

    python3 tools/gen_vllm_archs.py --image nvcr.io/nvidia/vllm:26.07-py3 \
        --out app/data/vllm_archs.json

Only the names are extractable without loading a model: every richer predicate
on the registry — is_pooling_model, is_pp_supported_model, is_multimodal_model —
takes a built ModelConfig, which means an engine process and a minute. So this
answers "can this build load that architecture at all", and no more.

It also records which architectures ship a *custom sampler*. That is not a
curiosity. vLLM's pipeline-parallel path has the last rank broadcast its sampled
tokens back to the earlier ranks into a buffer it allocates itself —
`torch.empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64)` — and
asserts the sender matches. Every model that goes through the standard sampler
does. A model that supplies its own is under no such discipline, and
DiffusionGemma's returns int32, `canvas_length` wide, which fails the assert
during warmup after the whole checkpoint has been read. The class still declares
SupportsPP, so vLLM's own check passes and nothing warns.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

INTROSPECT = r'''
import inspect
import json
import pathlib
import vllm
from vllm import ModelRegistry

archs = sorted(ModelRegistry.get_supported_archs())

# Which modules define a custom sampler, read from source rather than by
# importing 359 model classes.
root = pathlib.Path(vllm.__file__).parent / "model_executor" / "models"
custom = {
    f"vllm.model_executor.models.{path.stem}"
    for path in root.glob("*.py")
    if "def custom_sampler" in path.read_text(errors="replace")
}

custom_sampler = sorted(
    name for name in archs
    if getattr(ModelRegistry.models.get(name), "module_name", None) in custom
)

print("@@JSON@@" + json.dumps({
    "vllm_version": vllm.__version__,
    "architectures": archs,
    "custom_sampler": custom_sampler,
}))
'''


def introspect(image: str) -> dict:
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
         image, "-c", INTROSPECT],
        capture_output=True, text=True, timeout=900,
    )
    for line in result.stdout.splitlines():
        # vLLM writes a great deal to stdout before anything we asked for.
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    sys.exit(f"no result from {image}:\n{result.stderr[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="nvcr.io/nvidia/vllm:26.07-py3")
    parser.add_argument("--out", default="app/data/vllm_archs.json")
    args = parser.parse_args()

    if not shutil.which("docker"):
        sys.exit("docker not found on PATH")

    print(f"introspecting {args.image} ...", file=sys.stderr)
    data = introspect(args.image)
    data["image"] = args.image
    print(f"  {len(data['architectures'])} architectures, "
          f"{len(data.get('custom_sampler') or [])} with a custom sampler, "
          f"vLLM {data['vllm_version']}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
