"""Source size: does the filter recover the S/N a resolved mock was built with?

Three mocks (``configs/mocks/size_*.yaml``: a point source and sources of 1x and 3x
the beam FWHM), each calibrated so its declared ``snr`` is the optimal S/N, are
searched at the true position with both weightings. Weighting by the source
envelope ("template") should recover the declared S/N at every size; the plain
average ("natural") keeps only sum(w A) / sqrt(sum(w) sum(w A^2)) of it.

The mocks are exported to ``output/npz/size_<name>{,_noiseless}.npz`` and reused if
present, so they can be generated in advance with ``luv-mock`` and ``luv-export``.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from luv_finder import DataHandler, Gaussian, MatchedFilter, Model
from luv_finder.cli.find_lines import build_grid
from luv_finder.plotting import source_size_check

REPO = Path(__file__).parents[1]
NPZ_DIR = REPO / "output" / "npz"
SIZES = ("point", "beam", "3beam")


def _mock_npz(name: str) -> tuple[Path, Path]:
    """Noisy and noiseless NPZ of one preset, simulated with CASA if not already there."""
    noisy, clean = NPZ_DIR / f"size_{name}.npz", NPZ_DIR / f"size_{name}_noiseless.npz"
    if not (noisy.exists() and clean.exists()):
        pytest.importorskip("casatasks", exc_type=ImportError)
        from luv_finder import MockObservation

        mock = MockObservation.from_yaml(REPO / "configs" / "mocks" / f"size_{name}.yaml")
        mock.run_all()
        NPZ_DIR.mkdir(parents=True, exist_ok=True)
        DataHandler(mock.ms_noisy).to_npz(str(noisy))
        DataHandler(mock.ms_noiseless).to_npz(str(clean))
    return noisy, clean


def _response(data, grid_cfg, weighting):
    comp = Gaussian()
    comp.grid = build_grid(data, grid_cfg)
    mod = Model()
    mod.addcomponent(comp)
    mf = MatchedFilter(data, mod, weighting=weighting)
    mf.run(pool=1)
    return mf


@pytest.mark.casa
@pytest.mark.parametrize("weighting", ["natural", "template"])
@pytest.mark.parametrize("name", SIZES)
def test_recovers_declared_snr(name, weighting, request):
    mock_cfg = yaml.safe_load((REPO / "configs" / "mocks" / f"size_{name}.yaml").read_text())
    grid_cfg = yaml.safe_load((REPO / "configs" / "grids" / f"size_{name}.yaml").read_text())
    declared = mock_cfg["sources"][0]["line"]["snr"]
    noisy_path, clean_path = _mock_npz(name)

    noisy = DataHandler.from_npz(str(noisy_path))
    # expectation: the noiseless signal with the noise model of the noisy data
    clean = DataHandler.from_npz(str(clean_path))
    assert np.array_equal(clean.uvdata.uvtimes, noisy.uvdata.uvtimes)
    clean.uvdata.uvwghts = noisy.uvdata.uvwghts

    expected = _response(clean, grid_cfg, weighting)
    drawn = _response(noisy, grid_cfg, weighting)
    peak = expected.response[0].max()

    size = grid_cfg["bmaj"]
    a = Gaussian(bmin=size, bmaj=size).envelope(noisy.uvdata)
    w = noisy.uvdata.uvwghts
    kept = np.sum(w * a) / np.sqrt(np.sum(w) * np.sum(w * a**2))
    predicted = declared if weighting == "template" else declared * kept

    if request.config.getoption("--plots"):
        source_size_check(
            expected.frequencies(),
            expected.response[0],
            drawn.response[0],
            declared,
            f"size_{name}, {weighting}: expected {peak:.2f}, predicted {predicted:.2f}",
            plots_dir=str(REPO / "plots" / "source_size"),
            name=f"size_{name}_{weighting}",
        )

    assert peak == pytest.approx(predicted, rel=0.05)
    if weighting == "natural" and name == "3beam":
        assert peak < 0.6 * declared
