import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from luv_finder._casa import configure_logging

# Keep CASA session logs out of the repo, including for tests that import
# casatasks directly. conftest is imported before test modules are collected,
# which is early enough: casaconfig fixes the log path at first CASA import.
os.environ.setdefault("LUV_CASA_LOG_DIR", tempfile.mkdtemp(prefix="luv-casa-log-"))
configure_logging()

from luv_finder import DataHandler  # noqa: E402
from luv_finder.data import write_npz  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
PLOTS_DIR = Path(__file__).parents[1] / "plots"


def pytest_addoption(parser):
    parser.addoption("--plots", action="store_true", help="write diagnostic figures to plots/")


@pytest.fixture(scope="session")
def plots(request) -> Path:
    """Output directory for diagnostic figures; skips unless --plots was passed."""
    if not request.config.getoption("--plots"):
        pytest.skip("needs --plots")
    if not getattr(request.config, "_luv_plots_cleaned", False):
        shutil.rmtree(PLOTS_DIR, ignore_errors=True)
        request.config._luv_plots_cleaned = True
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def pytest_sessionfinish(session, exitstatus):
    """Build the contact sheet once every figure has been written."""
    if not session.config.getoption("--plots") or not PLOTS_DIR.exists():
        return
    from luv_finder.plotting import contact_sheet

    path = contact_sheet(str(PLOTS_DIR))
    n = len(list(PLOTS_DIR.glob("*.png")))
    session.config.pluginmanager.get_plugin("terminalreporter").write_line(
        f"\n{n} diagnostic figures written. Open: {path}", bold=True
    )


@pytest.fixture(scope="session")
def truth() -> dict:
    """Injected sources of the line13_line9 mock."""
    return json.loads((FIXTURES / "line13_line9_truth.json").read_text())


@pytest.fixture(scope="session")
def fixture_npz() -> Path:
    return FIXTURES / "line13_line9_small.npz"


@pytest.fixture(scope="session")
def data(fixture_npz) -> DataHandler:
    return DataHandler.from_npz(fixture_npz)


@pytest.fixture(scope="session")
def two_field_npz(data, tmp_path_factory) -> Path:
    """The fixture's chunks as field 0 and, with its phase centre moved by (+10, 0) arcsec, as field 1."""
    chunks = [
        dataclasses.replace(c, field=field, offset=c.offset + np.array([10.0 * field, 0.0]))
        for field in (0, 1)
        for c in data.chunks
    ]
    path = tmp_path_factory.mktemp("mosaic") / "two_fields.npz"
    write_npz(str(path), data.metadata, chunks)
    return path
