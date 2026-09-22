"""End-to-end CASA test: the committed smoke preset, simulated and then searched.

This is the one test that exercises simobserve/tclean, so it doubles as the check
that a mock preset and its matching grid preset agree -- including the sign flip
between image and model conventions for dra, and that a declared ``snr`` really
is the achieved matched-filter S/N.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("casatasks")

from luv_finder import DataHandler, Gaussian, MatchedFilter, MockObservation, Model  # noqa: E402
from luv_finder.cli.find_lines import build_grid  # noqa: E402

REPO = Path(__file__).parents[1]
MOCK_CFG = REPO / "configs" / "mocks" / "smoke.yaml"
GRID_CFG = REPO / "configs" / "grids" / "smoke.yaml"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """Simulate the smoke preset once and reuse it across the tests in this module."""
    tmp = tmp_path_factory.mktemp("smoke")
    cfg = yaml.safe_load(MOCK_CFG.read_text())
    cwd = Path.cwd()
    import os

    os.chdir(tmp)  # simobserve writes its project into the working directory
    try:
        mock = MockObservation.from_yaml(
            MOCK_CFG,
            ptg_file=str(tmp / "ptg.txt"),
            fits_filename=str(tmp / "support" / "smoke_input.fits"),
            output_folder=str(tmp / "out"),
            support_folder=str(tmp / "support"),
        )
        mock.run_all(plots_dir=str(tmp / "plots"))
        yield mock, cfg, tmp
    finally:
        os.chdir(cwd)


@pytest.mark.casa
def test_simulated_shape_matches_preset(smoke):
    mock, cfg, _ = smoke
    data = DataHandler(mock.ms_noisy)
    nchan = cfg["cube_shape"][0]
    assert data.n_freqs(data.uvdata) == nchan
    assert data.uvdata.UVreals.size == nchan * data.n_visbs(data.uvdata)


@pytest.mark.casa
def test_declared_snr_is_achieved(smoke):
    """calibrate_snr rescales the cube so `snr` means matched-filter S/N."""
    mock, cfg, _ = smoke
    wanted = [s["line"]["snr"] for s in cfg["sources"] if "line" in s]
    assert mock.achieved_snr() == pytest.approx(wanted, rel=0.1)


@pytest.mark.casa
def test_grid_preset_recovers_the_line(smoke):
    mock, cfg, _ = smoke
    data = DataHandler(mock.ms_noisy)
    comp = Gaussian()
    comp.grid = build_grid(data, yaml.safe_load(GRID_CFG.read_text()))
    mod = Model()
    mod.addcomponent(comp)
    mf = MatchedFilter(data, mod)
    mf.run(pool=1)

    src = cfg["sources"][0]
    peak_ghz = mf.frequencies()[np.argmax(mf.response[mf.best_index])]
    assert abs(peak_ghz - src["line"]["mean"]) < 0.05

    # the grid preset is written in model convention, which flips the sign of dra
    assert mf.best_params["src_00_dra"] == pytest.approx(-src["position"][0])
    assert mf.best_params["src_00_ddec"] == pytest.approx(src["position"][1])
    # response is in S/N units, so the peak should be near the declared value
    assert mf.response[mf.best_index].max() > 0.6 * src["line"]["snr"]
