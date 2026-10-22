import numpy as np
import pytest
from astropy.table import Table

from luv_finder.matchedfilter import window_groups
from luv_finder.run import RUN_FILE, Run, duplicates, unit_name


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
    assert cat.meta["rule"] == "and" and set(cat["unit"]) == set(run.units)
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


def test_window_groups_join_windows_that_overlap_or_nearly_meet():
    """Two tunings over 100-102 GHz join, and so does a window at their edge; a gap joins if narrower than
    the widest template (400 km/s, 141 MHz at 106 GHz)."""
    ch = 0.01e9
    freqs = {
        0: np.arange(100e9, 101e9, ch),
        1: np.arange(100.5e9, 102e9, ch),
        2: np.arange(102e9 - ch / 2, 103e9, ch),
        3: np.arange(105e9, 106e9, ch),
        4: np.arange(106.1e9, 107e9, ch),  # 110 MHz after window 3's last channel
        5: np.arange(107.2e9, 108e9, ch),  # 210 MHz after window 4's
    }
    assert window_groups(freqs) == [[0, 1, 2], [3, 4], [5]]
    assert window_groups(freqs, widest=100.0) == [[0, 1, 2], [3], [4], [5]]
    assert unit_name([14, 15, 17]) == "spw14+15+17"


def test_run_plots_come_from_saved_products(run):
    from matplotlib.figure import Figure

    for figure in (run.plot.reliability, run.plot.spectra, run.plot.maps, run.plot.responses, run.plot.snr_map):
        assert isinstance(figure(plots_dir=False), Figure)
    with pytest.raises(ValueError, match="no weight scales"):
        run.diagnostics(plots_dir=False)


def test_duplicates_catch_the_two_halves_of_a_line_split_by_a_window_edge():
    """ASPECS 3mm.07: 609 km/s wide across the spw00/spw01 gap, found in both, its halves 300 km/s apart."""
    cat = Table({
        "id": [1, 2, 3, 4],
        "unit": ["spw01", "spw00", "spw00", "spw02"],
        "detected": [True, True, True, True],
        "dra": [0.0, 0.3, 7.0, 0.1],
        "ddec": [0.0, 0.2, 0.0, 0.0],
        "freq_ghz": [93.610, 93.516, 93.500, 93.600],
        "width": [300.0, 500.0, 100.0, 100.0],
        "fit_width": [243.0, 476.0, 172.0, np.nan],
    })  # fmt: skip
    beam = {"spw00": 0.7, "spw01": 0.7, "spw02": 0.7}
    assert duplicates(cat, beam).tolist() == [0, 1, 0, 1]


def test_merge_applies_the_run_files_reliability(run):
    """A rule changed in the run file takes effect on merge, without a new search."""
    other = "or" if run.config["reliability"]["rule"] == "and" else "and"
    again = Run({**run.config, "reliability": {**run.config["reliability"], "rule": other}})
    assert again.merge().meta["rule"] == other
    assert run.merge().meta["rule"] == run.config["reliability"]["rule"]


def test_unit_plan_chunks_blocks_with_their_neighbours():
    """Five adjacent windows and a second tuning inside the third: five blocks, cut into chunks of at most
    300 channels, each loading its neighbouring blocks and owning the centres between the gaps to them."""
    from luv_finder.run import unit_plan

    ch = 0.01e9
    freqs = {s: np.arange(100e9 + s * 1.01e9, 101e9 + s * 1.01e9, ch) for s in range(5)}
    freqs[5] = freqs[2][:50] + 0.25e9  # a second tuning inside window 2
    plan = unit_plan(freqs, widest=400.0, max_channels=300)
    assert plan["blocks"] == {"spw00": [0], "spw01": [1], "spw02+05": [2, 5], "spw03": [3], "spw04": [4]}
    units = plan["units"]  # greedily: 100 + 100, then 150 + 100, then 100
    assert list(units) == ["spw00+01", "spw02+03+05", "spw04"]
    assert units["spw00+01"].blocks == ["spw00", "spw01", "spw02+05"]
    assert units["spw02+03+05"].blocks == ["spw01", "spw02+05", "spw03", "spw04"]
    assert units["spw04"].blocks == ["spw03", "spw04"]
    gap = [(freqs[1].max() + freqs[2].min()) / 2, (freqs[3].max() + freqs[4].min()) / 2]
    assert units["spw00+01"].core == (-np.inf, gap[0]) and units["spw02+03+05"].core == tuple(gap)
    assert units["spw04"].core == (gap[1], np.inf)
    assert unit_plan(freqs, widest=400.0)["units"] == {"spw00+01+02+03+04+05": ([*plan["blocks"]], None)}
