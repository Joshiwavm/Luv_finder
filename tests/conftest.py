import json
import os
import tempfile
from pathlib import Path

import pytest

from luv_finder._casa import configure_logging

# Keep CASA session logs out of the repo, including for tests that import
# casatasks directly. conftest is imported before test modules are collected,
# which is early enough: casaconfig fixes the log path at first CASA import.
os.environ.setdefault("LUV_CASA_LOG_DIR", tempfile.mkdtemp(prefix="luv-casa-log-"))
configure_logging()

from luv_finder import DataHandler  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def truth() -> dict:
    """Injected sources of the line13_line9 mock."""
    return json.loads((FIXTURES / "line13_line9_truth.json").read_text())


@pytest.fixture(scope="session")
def data() -> DataHandler:
    return DataHandler.from_npz(FIXTURES / "line13_line9_small.npz")
