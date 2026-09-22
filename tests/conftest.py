import json
from pathlib import Path

import pytest

from luv_finder import DataHandler

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def truth() -> dict:
    """Injected sources of the line13_line9 mock."""
    return json.loads((FIXTURES / "line13_line9_truth.json").read_text())


@pytest.fixture(scope="session")
def data() -> DataHandler:
    return DataHandler.from_npz(FIXTURES / "line13_line9_small.npz")
