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


@pytest.fixture(scope="session", autouse=True)
def _database():
    from app import db

    db.init_db()
    yield
