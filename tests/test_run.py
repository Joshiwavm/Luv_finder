import numpy as np
import pytest
from astropy.table import Table

from luv_finder.run import RUN_FILE, Run, unit_name, window_units


@pytest.fixture(scope="module")
def run(fixture_npz, data, tmp_path_factory) -> Run:
    """The fixture searched on a +-30 arcsec lattice with one template, and fitted."""
    step = data.metadata.minresolution() / 2
    lattice = {"start": float(-18 * step), "stop": float(18.5 * step), "step": float(step)}
    run = Run({
        "data": str(fixture_npz),
        "output": str(tmp_path_factory.mktemp("runs") / "fixture"),
        "grid": {"dra": lattice, "ddec": lattice, "width": [300.0]},
    })  # fmt: skip
    run.search()
    run.fit()
    return run


def test_run_writes_every_product(run):
    (unit,) = run.units
    for path in (
        run.output / RUN_FILE,
        run.export_path(unit),
        run.path(unit, ".ecsv"),
        run.path(unit, "_maps.npz"),
        run.path(unit, "_weights.npz"),
        run.output / "catalogue.ecsv",
        run.output / "lines_fit.ecsv",
        run.output / "line_maps.npz",
    ):
        assert path.is_file(), path


def test_run_detects_the_injected_lines(run, truth, data):
    cat = run.catalogue
    assert cat.meta["rule"] == "or" and set(cat["unit"]) == set(run.units)
    channel = np.median(np.diff(data.freqs)) / 1e9
    for src in truth["sources"]:
        x, y = src["position"]
        near = (np.hypot(cat["dra"] - x, cat["ddec"] - y) < data.metadata.minresolution() / 2) & (
            np.abs(cat["freq_ghz"] - src["line"]["mean"]) <= channel
        )
        assert near.sum() == 1 and cat["detected"][near][0]


def test_run_maps_hold_the_filter_response_of_every_group(run):
    (unit,) = run.units
    m, cat = run.maps(unit), Table.read(run.path(unit, ".ecsv"))
    assert m["peak"].shape == (m["dra"].size, m["ddec"].size) == m["jk_peak"].shape
    assert m["group_snr"].shape == (len(cat), m["widths"].size, m["freqs"].size) == m["group_jk"].shape
    np.testing.assert_array_equal(m["group_id"], cat["unit_id"])
    np.testing.assert_allclose(np.nanmax(m["group_snr"], axis=(1, 2)), cat["snr"], rtol=1e-6)
    assert np.nanmax(m["peak"]) == pytest.approx(cat["snr"].max(), rel=1e-6)
    freqs, widths, snr, jk = run.responses(run.catalogue[:2])
    assert len(snr) == 2 and snr[0].shape == (widths[0].size, freqs[0].size)


def test_run_fits_the_detections(run):
    fit = run.lines
    detected = np.asarray(fit["detected"])
    assert np.all(np.isfinite(fit["fit_dra"][detected])) and "near_miss" in fit.colnames
    assert set(run.line_maps()) == set(fit["id"][np.isfinite(fit["fit_dra"])])
    assert "detected" in run.summary()


def test_run_reopens_from_its_output_and_skips_what_is_done(run):
    again = Run.open(run.output)
    assert again.units == run.units and again.config == run.config
    before = {p: p.stat().st_mtime for p in (run.output / "units").iterdir()}
    again.search()
    assert {p: p.stat().st_mtime for p in (run.output / "units").iterdir()} == before
    np.testing.assert_array_equal(again.catalogue["snr"], run.catalogue["snr"])


def test_run_releases_stale_claims(run, tmp_path):
    other = Run(run.config | {"output": str(tmp_path / "other")})
    (unit,) = other.units
    other.path(unit, "").parent.mkdir(parents=True)
    other.path(unit, ".claim").touch()
    assert other.release() == [unit] and not other.path(unit, ".claim").exists()


def test_run_needs_data_and_output():
    with pytest.raises(ValueError, match="needs"):
        Run({"data": "x.npz"})


def test_window_units_join_windows_that_overlap():
    """Two tunings over 100-102 GHz join; a window touching another at its edge stays apart."""
    ch = 0.01e9
    freqs = {
        0: np.arange(100e9, 101e9, ch),
        1: np.arange(100.5e9, 102e9, ch),
        2: np.arange(102e9 - ch / 2, 103e9, ch),
        3: np.arange(105e9, 106e9, ch),
    }
    assert window_units(freqs) == [[0, 1], [2], [3]]
    assert unit_name([14, 15, 17]) == "spw14+15+17"
