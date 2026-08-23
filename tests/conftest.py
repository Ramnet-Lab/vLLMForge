"""Point the dashboard at a throwaway state directory for the whole test run."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = tempfile.mkdtemp(prefix="llmd-tests-")
os.environ.setdefault("LLMD_STATE_DIR", _TMP)
os.environ.setdefault("LLMD_OUTPUT_DIR", str(Path(_TMP) / "outputs"))
os.environ.setdefault("LLMD_DATASET_DIR", str(Path(_TMP) / "datasets"))
# Pin the accelerator pool, because otherwise the suite measures whatever card
# the developer happens to have. These tests describe a unified machine: they
# patch read_meminfo and read_gpu_processes and expect the pool to be built from
# those. Without the pin, accel.local_pool() shells out to nvidia-smi, and on a
# discrete box the real framebuffer walks in behind the stubs — a run on two
# L40s failed three tests that pass on a GB10, for no reason in the code under
# test. A test that wants the discrete path asks for it explicitly.
os.environ.setdefault("LLMD_ACCEL_MODE", "unified")


@pytest.fixture(scope="session", autouse=True)
def _database():
    from app import db

    db.init_db()
    yield
