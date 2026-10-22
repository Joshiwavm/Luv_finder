import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

from luv_finder._casa import configure_logging

# Reuse compiled kernels across test runs (most of the suite's time is small
# compiles); the cache is keyed on the jax version and the traced code.
jax.config.update("jax_compilation_cache_dir", str(Path(__file__).parents[1] / ".pytest_cache" / "jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

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


def contact_sheet(plots_dir: str, title: str = "Luv_finder diagnostics") -> str:
    """Write an index.html showing every PNG in ``plots_dir``."""
    pngs = sorted(f for f in os.listdir(plots_dir) if f.endswith(".png"))
    cards = "\n".join(f'<figure><img src="{f}" loading="lazy"><figcaption>{f[:-4]}</figcaption></figure>' for f in pngs)
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;background:#fafafa;color:#222}}
 h1{{font-size:1.2rem}} .grid{{display:grid;gap:1.5rem;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}}
 figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;padding:.75rem}}
 img{{width:100%;height:auto}}
 figcaption{{margin-top:.5rem;font-family:ui-monospace,monospace;font-size:12px;color:#555}}
</style>
<h1>{title}</h1><p>{len(pngs)} figures</p><div class="grid">{cards}</div>"""
    path = os.path.join(plots_dir, "index.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path


def pytest_sessionfinish(session, exitstatus):
    """Build the contact sheet once every figure has been written."""
    if not session.config.getoption("--plots") or not PLOTS_DIR.exists():
        return
    path = contact_sheet(str(PLOTS_DIR))
    n = len(list(PLOTS_DIR.glob("*.png")))
    session.config.pluginmanager.get_plugin("terminalreporter").write_line(
        f"\n{n} diagnostic figures written. Open: {path}", bold=True
    )


@pytest.fixture(scope="session")
def blind_catalogue(data):
    """The fixture catalogued blind on a +-30 arcsec lattice (one search for the tests that need it)."""
    from luv_finder.catalogue import catalogue
    from luv_finder.matchedfilter import MatchedFilter, build_grid, pb_corrected

    lattice = data.metadata.minresolution() / 2 * np.arange(-18, 19)
    mf = MatchedFilter(data, build_grid(data, {"dra": lattice, "ddec": lattice}))
    mf.run(jackknife=True)
    return catalogue(pb_corrected(mf.result, data), ref=data.metadata.ref)


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
