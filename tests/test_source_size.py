"""Source size: does the filter recover the S/N a resolved mock was built with?

Every ``configs/grids/size_*.yaml`` is a targeted search as a user would write it: one
position, width and size. The mock of the same name only sets the observation and the
declared ``snr``; its source geometry is taken from the grid, so the template and the injected
source cannot drift apart. Each mock is calibrated so its ``snr`` is the optimal S/N and is
searched with both weightings. Weighting by the source envelope ("template") should recover
the declared S/N at every size; the plain average ("natural") keeps only
sum(w A) / sqrt(sum(w) sum(w A^2)) of it.

The mocks are exported to ``output/npz/size_<name>{,_noiseless}.npz`` and rebuilt whenever
either YAML is newer than them.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from luv_finder import DataHandler, Gaussian, MatchedFilter, Model
from luv_finder.matchedfilter import build_grid
from luv_finder.plotting import source_size_check

REPO = Path(__file__).parents[1]
NPZ_DIR = REPO / "output" / "npz"
NAMES = [p.stem.removeprefix("size_") for p in sorted((REPO / "configs" / "grids").glob("size_*.yaml"))]


def _configs(name: str) -> tuple[Path, Path]:
    return REPO / "configs" / "mocks" / f"size_{name}.yaml", REPO / "configs" / "grids" / f"size_{name}.yaml"


def _mock_npz(name: str) -> tuple[Path, Path]:
    """Noisy and noiseless NPZ of one preset, simulated with CASA if missing or older than its YAMLs."""
    mock_cfg, grid_cfg = _configs(name)
    noisy, clean = NPZ_DIR / f"size_{name}.npz", NPZ_DIR / f"size_{name}_noiseless.npz"
    newest = max(mock_cfg.stat().st_mtime, grid_cfg.stat().st_mtime)
    if not all(p.exists() and p.stat().st_mtime > newest for p in (noisy, clean)):
        pytest.importorskip("casatasks", exc_type=ImportError)
        from luv_finder import MockObservation

        mock = MockObservation.from_yaml(mock_cfg, grid=grid_cfg)
        mock.run_all()
        NPZ_DIR.mkdir(parents=True, exist_ok=True)
        DataHandler(mock.ms_noisy).to_npz(str(noisy))
        DataHandler(mock.ms_noiseless).to_npz(str(clean))
    return noisy, clean


def _response(data, grid_cfg, weighting, correlation):
    comp = Gaussian()
    comp.grid = build_grid(data, grid_cfg)
    mod = Model()
    mod.addcomponent(comp)
    mf = MatchedFilter(data, mod, weighting=weighting, continuum_order=None, channel_correlation=correlation)
    mf.run()
    return mf


@pytest.mark.casa
@pytest.mark.parametrize("weighting", ["natural", "template"])
@pytest.mark.parametrize("name", NAMES)
def test_recovers_declared_snr(name, weighting, request):
    mock_path, grid_path = _configs(name)
    declared = yaml.safe_load(mock_path.read_text())["sources"][0]["line"]["snr"]
    grid_cfg = yaml.safe_load(grid_path.read_text())
    noisy_path, clean_path = _mock_npz(name)

    noisy = DataHandler.from_npz(str(noisy_path))
    # expectation: the noiseless signal with the noise model of the noisy data
    (signal,) = DataHandler.from_npz(str(clean_path)).chunks
    (chunk,) = noisy.chunks
    assert np.array_equal(signal.time, chunk.time)
    clean = DataHandler(chunks=[dataclasses.replace(chunk, X=chunk.w * signal.vis)], metadata=noisy.metadata)

    # the noise model, channel correlation included, is the noisy data's: a noiseless jackknife
    # holds only the signal's residual between integrations, smooth in frequency
    correlation = noisy.channel_correlation()
    expected = _response(clean, grid_cfg, weighting, correlation)
    drawn = _response(noisy, grid_cfg, weighting, correlation)
    peak = expected.response[0].max()

    a = Gaussian(bmin=grid_cfg["bmin"], bmaj=grid_cfg["bmaj"], pa=grid_cfg.get("pa", 0.0)).envelope(chunk)
    kept = np.sum(chunk.w * a) / np.sqrt(np.sum(chunk.w) * np.sum(chunk.w * a**2))
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
