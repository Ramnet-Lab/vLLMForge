"""Which base image a machine builds on.

The rule is asymmetric on purpose, so the tests are too: moving off NGC needs
positive proof of a discrete GPU, while everything unproven stays where the
Spark has always been.
"""

from __future__ import annotations

import pytest

from app import images


def test_a_measured_discrete_gpu_moves_off_the_ngc_base():
    """NGC ships vLLM 0.24, which cannot construct a tied-embedding model under
    4-bit quantisation — it dies in tie_weights with NotImplementedError after
    the weights are already read."""
    image, why = images.choose_base_image("discrete", "x86_64")
    assert image == images.DISCRETE_BASE
    assert "discrete" in why


@pytest.mark.parametrize("kind", ["unified", "none", "unknown", ""])
def test_anything_unproven_keeps_the_base_a_spark_needs(kind):
    image, _why = images.choose_base_image(kind, "x86_64")
    assert image == images.NGC_BASE


def test_aarch64_stays_on_ngc_even_with_a_discrete_card():
    """The official x86 image has no build for this architecture, so 'discrete'
    is not enough to send an aarch64 machine to an image it cannot run."""
    image, why = images.choose_base_image("discrete", "aarch64")
    assert image == images.NGC_BASE
    assert "arch" in why


def test_an_explicit_setting_wins_over_the_detector():
    image, why = images.choose_base_image("discrete", "x86_64", override="my/own:tag")
    assert image == "my/own:tag"
    assert "LLMD_VLLM_BASE_IMAGE" in why


def test_auto_is_not_treated_as_an_image_name():
    """The default is the string 'auto'; taking it literally would try to pull
    docker.io/library/auto:latest."""
    image, _why = images.choose_base_image("discrete", "x86_64", override="auto")
    assert image == images.DISCRETE_BASE


@pytest.mark.asyncio
async def test_detection_failure_falls_back_rather_than_raising(monkeypatch):
    """A wedged driver must not stop an install: the fallback is the image that
    was always used, not an exception out of the build script."""
    async def boom():
        raise RuntimeError("nvidia-smi went away")

    monkeypatch.setattr(images, "detect_base_image", images.detect_base_image)
    from app import accel

    monkeypatch.setattr(accel, "local_pool", boom)
    image, why = await images.detect_base_image()
    assert image == images.NGC_BASE
    assert "detection failed" in why
