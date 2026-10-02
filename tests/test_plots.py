"""Diagnostic figures. Only run with --plots; they are for the eye, not for CI.

Assertions here are deliberately weak: the point is to produce something to look
at. The numerical checks live in the other test modules.
"""

import numpy as np

from luv_finder import Gaussian, MatchedFilter, Model
from luv_finder.plotting import response_check, spectrum_check


def test_spectrum_per_source(data, truth, plots):
    """Re/Im visibility spectra: data, jackknife and model, centred and shifted."""
    res = data.metadata.minresolution()
    for i, src in enumerate(truth["sources"]):
        dra, ddec = src["position_model"]
        model = Gaussian(
            dra=dra,
            ddec=ddec,
            total_flux=src["line"]["flux"],
            bmin=res / 10,
            bmaj=res / 10,
            nu_center=src["line"]["mean"] * 1e9,
            width=src["line"]["width"],
        )
        path = spectrum_check(
            data,
            dra,
            ddec,
            model=model,
            plots_dir=str(plots),
            name=f"spectrum_src{i}",
            line_ghz=src["line"]["mean"],
        )
        assert path.endswith(".png")


def test_response_figure(data, truth, plots):
    """Matched-filter S/N spectrum with the jackknife overlaid."""
    positions = np.array([s["position_model"] for s in truth["sources"]])
    g = Gaussian()
    g.grid = {
        "dra": positions[:, 0],
        "ddec": positions[:, 1],
        "bmin": data.metadata.minresolution() / 10,
        "bmaj": data.metadata.minresolution() / 10,
        "width": 300.0,
    }
    mod = Model()
    mod.addcomponent(g)
    mf = MatchedFilter(data, mod)
    mf.run(jackknife=True)
    path = response_check(
        mf, plots_dir=str(plots), name="filter_response", line_ghz=truth["sources"][0]["line"]["mean"]
    )
    assert path.endswith(".png")


def test_search_figures(data, truth, plots):
    """S/N map with the injected sources circled, phase centre against the sources, and noise statistics."""
    from luv_finder.plotting import noise_check, responses_check, snr_map

    g = Gaussian()
    axis = np.arange(-30.0, 30.1, 2.0)
    g.grid = {"dra": axis, "ddec": axis, "bmin": 0.3, "bmaj": 0.3, "width": 300.0}
    mod = Model()
    mod.addcomponent(g)
    mf = MatchedFilter(data, mod)
    mf.run(jackknife=True)
    marks = {f"line {s['line']['mean']} GHz": s["position_model"] for s in truth["sources"]}
    assert snr_map(mf, plots_dir=str(plots), name="snr_map", marks=marks).endswith(".png")
    assert noise_check(mf, plots_dir=str(plots), name="noise").endswith(".png")

    points = {"phase centre": (0.0, 0.0), **{k: tuple(v) for k, v in marks.items()}}
    single = Gaussian()
    single.grid = {"dra": [p[0] for p in points.values()], "ddec": [p[1] for p in points.values()], "width": 300.0}
    mod = Model()
    mod.addcomponent(single)
    pick = MatchedFilter(data, mod)
    pick.run()
    rows = {k: pick.response[i * (len(points) + 1)] for i, k in enumerate(points)}  # the (dra_i, ddec_i) diagonal
    path = responses_check(pick.frequencies(), rows, "Phase centre vs sources", plots_dir=str(plots))
    assert path.endswith(".png")


def test_layout_and_scan_figures(data, plots):
    from luv_finder.plotting import mosaic_layout, scan_check, uv_profile

    pb = data.metadata.primarybeamsize()
    offsets = {0: (0.0, 0.0), 1: (-17.0, -30.0), 2: (17.0, -30.0)}
    assert mosaic_layout(offsets, pb, plots_dir=str(plots), marks={"line": (2.0, 20.5)}).endswith(".png")
    models = {"point": Gaussian(), "2 arcsec": Gaussian(bmin=2.0, bmaj=2.0)}
    path = uv_profile({k: data for k in models}, 4.25, 23.5, models, 39.9, plots_dir=str(plots))
    assert path.endswith(".png")
    x = np.linspace(0, 4, 9)
    assert scan_check(x, {"demo": np.exp(-((x - 2) ** 2))}, "size", "scan", truth=2.0, plots_dir=str(plots)).endswith(
        ".png"
    )


def test_amp_phase_and_dirty_map_figures(data, truth, plots):
    from luv_finder.matchedfilter import dirty_maps
    from luv_finder.plotting import amp_phase_check, dirty_maps_check

    marks = {f"line {s['line']['mean']} GHz": s["position_model"] for s in truth["sources"]}
    positions = {"phase centre": (0.0, 0.0), **marks}
    assert amp_phase_check(data, positions, plots_dir=str(plots), line_ghz=39.9).endswith(".png")
    axis = np.arange(-40.0, 40.1, 1.5)
    moment8, continuum, sigma = dirty_maps(data, axis, axis)
    assert dirty_maps_check(axis, axis, moment8, continuum, sigma, plots_dir=str(plots), marks=marks).endswith(".png")
