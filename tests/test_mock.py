"""End-to-end CASA test: the committed smoke preset, simulated and then searched.

This is the one test that exercises simobserve/tclean, so it doubles as the
check that a mock preset and its matching grid preset agree -- including the
sign flip between image and model conventions for dra.
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


@pytest.mark.casa
def test_smoke_preset_simulates_and_is_recovered(tmp_path, monkeypatch):
    cfg = yaml.safe_load(MOCK_CFG.read_text())
    # simobserve writes its project into the working directory
    monkeypatch.chdir(tmp_path)

    mock = MockObservation.from_yaml(
        MOCK_CFG,
        alma_config=str(REPO / cfg["alma_config"]),
        ptg_file=str(tmp_path / "ptg.txt"),
        fits_filename=str(tmp_path / "support" / "smoke_input.fits"),
        output_folder=str(tmp_path / "out"),
        support_folder=str(tmp_path / "support"),
    )
    mock.run_all(plots_dir=str(tmp_path / "plots"))

    data = DataHandler(mock.ms_noisy)
    nchan = cfg["cube_shape"][0]
    assert data.n_freqs(data.uvdata) == nchan
    assert data.uvdata.UVreals.size == nchan * data.n_visbs(data.uvdata)

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
