"""Which vLLM base image to build the worker images on.

The dashboard was built on a DGX Spark, where the right base is NVIDIA's NGC
build: it is the one that exists for that part, on that architecture, with that
driver. Everywhere else it is the wrong choice for a reason that does not show
up until a model is loaded — NGC 26.07 ships vLLM 0.24, and 0.24 cannot serve a
checkpoint whose lm_head is tied to its embedding table under a 4-bit quant
method:

    gemma4.py: self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
      -> base_config.py: raise NotImplementedError

The engine starts, pulls the weights, and dies at model construction. Nothing in
a flag fixes it and no amount of reading the image's flag schema predicts it,
because the schema is complete and correct — the gap is in what the version can
construct, not in what it will accept.

So the base is chosen from the hardware, by the same detector the rest of the
app trusts for memory (app/accel.py), and the rule is deliberately conservative:
only a positively-measured discrete GPU moves off NGC. Anything unproven —
detection failed, no accelerator, a unified part, an aarch64 machine — keeps the
image the Spark has always used. Being wrong in that direction costs nothing
that was working before; being wrong the other way replaces a working Spark
image with one that has no build for its architecture.
"""

from __future__ import annotations

import platform

# Pinned, not floated. The base decides which vLLM every server on this machine
# runs, and "whatever was latest the day someone rebuilt" is not a thing to
# discover from a model that will not load. This is the version this repo has
# actually served Gemma 4 and Qwen3 embeddings on.
NGC_BASE = "nvcr.io/nvidia/vllm:26.07-py3"
DISCRETE_BASE = "vllm/vllm-openai:v0.27.1"

AUTO = "auto"


def choose_base_image(kind: str, machine: str = "", override: str = "") -> tuple[str, str]:
    """(image, why). Pure, so the rule can be tested without a GPU.

    `kind` is an app.accel pool kind: "discrete", "unified", "none", "unknown".
    """
    override = (override or "").strip()
    if override and override.lower() != AUTO:
        return override, f"LLMD_VLLM_BASE_IMAGE={override}"

    machine = (machine or platform.machine() or "").lower()
    if machine in ("aarch64", "arm64"):
        # The Spark and every Jetson. The official x86 image has no build here,
        # and NGC is the one vendor image that does.
        return NGC_BASE, f"{machine}: NVIDIA's NGC build is the one that exists for this arch"

    if kind == "discrete":
        return DISCRETE_BASE, (
            "a discrete GPU was measured, and NGC's vLLM 0.24 cannot construct a "
            "tied-embedding model under 4-bit quantisation")

    return NGC_BASE, f"pool kind {kind or 'unknown'}: keeping the base a unified machine needs"


async def detect_base_image(override: str = "") -> tuple[str, str]:
    """The same call the memory panel makes, asked for a different reason."""
    if (override or "").strip() and (override or "").strip().lower() != AUTO:
        return choose_base_image("", "", override)
    try:
        from app import accel

        pool = await accel.local_pool()
        kind = pool.kind
    except Exception as exc:  # nvidia-smi absent, driver wedged, anything
        image, _why = choose_base_image("unknown", "", "")
        return image, f"detection failed ({exc}); keeping the base a unified machine needs"
    return choose_base_image(kind, "", "")
