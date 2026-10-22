"""Diagnostic figures. Only run with --plots; they are for the eye, not for CI.

Assertions here are deliberately weak: the point is to produce something to look
at. The numerical checks live in the other test modules.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from luv_finder import Gaussian, MatchedFilter
from luv_finder.data import reweighted
from luv_finder.plotting import response_check, spectrum_check, weights_check
from luv_finder.utils import primary_beam_fwhm


def test_spectrum_per_source(data, truth, plots):
    """Re/Im visibility spectra: data, jackknife and model, centred and shifted."""
    res = data.metadata.minresolution()
    for i, src in enumerate(truth["sources"]):
        dra, ddec = src["position"]
        model = Gaussian(
            dra=dra,
            ddec=ddec,
            total_flux=src["line"]["flux"],
            bmin=res / 10,
            bmaj=res / 10,
            nu_center=src["line"]["mean"] * 1e9,
            width=src["line"]["width"],
        )
        fig = spectrum_check(
            data,
            dra,
            ddec,
            model=model,
            plots_dir=str(plots),
            name=f"spectrum_src{i}",
            line_ghz=src["line"]["mean"],
        )
        assert isinstance(fig, Figure)


def test_response_figure(data, truth, plots):
    """Matched-filter S/N spectrum with the jackknife overlaid."""
    positions = np.array([s["position"] for s in truth["sources"]])
    grid = {
        "dra": positions[:, 0],
        "ddec": positions[:, 1],
        "bmin": data.metadata.minresolution() / 10,
        "bmaj": data.metadata.minresolution() / 10,
        "width": 300.0,
    }
    mf = MatchedFilter(data, grid)
    mf.run(jackknife=True)
    fig = response_check(mf, plots_dir=str(plots), name="filter_response", line_ghz=truth["sources"][0]["line"]["mean"])
    assert isinstance(fig, Figure)


def test_search_figures(data, truth, plots):
    """S/N map with the injected sources circled, phase centre against the sources, and noise statistics."""
    from luv_finder.plotting import noise_check, responses_check, snr_map

    axis = np.arange(-30.0, 30.1, 2.0)
    grid = {"dra": axis, "ddec": axis, "bmin": 0.3, "bmaj": 0.3, "width": 300.0}
    mf = MatchedFilter(data, grid)
    mf.run(jackknife=True)
    marks = {f"line {s['line']['mean']} GHz": s["position"] for s in truth["sources"]}
    assert isinstance(snr_map(mf, plots_dir=str(plots), name="snr_map", marks=marks), Figure)
    assert isinstance(noise_check(mf, plots_dir=str(plots), name="noise"), Figure)

    points = {"phase centre": (0.0, 0.0), **{k: tuple(v) for k, v in marks.items()}}
    grid = {"dra": [p[0] for p in points.values()], "ddec": [p[1] for p in points.values()], "width": 300.0}
    pick = MatchedFilter(data, grid)
    pick.run()
    rows = {k: pick.response[i * (len(points) + 1)] for i, k in enumerate(points)}  # the (dra_i, ddec_i) diagonal
    fig = responses_check(pick.frequencies(), rows, "Phase centre vs sources", plots_dir=str(plots))
    assert isinstance(fig, Figure)


def test_layout_figure(data, plots):
    from luv_finder.plotting import mosaic_layout

    pb = primary_beam_fwhm(data.metadata.central_frequency, data.metadata.dish_diameter)
    offsets = {0: (0.0, 0.0), 1: (-17.0, -30.0), 2: (17.0, -30.0)}
    assert isinstance(mosaic_layout(offsets, pb, plots_dir=str(plots), marks={"line": (2.0, 20.5)}), Figure)


def test_amp_phase_and_dirty_map_figures(data, truth, plots):
    from luv_finder.imaging import dirty_maps
    from luv_finder.plotting import amp_phase_check, dirty_maps_check

    marks = {f"line {s['line']['mean']} GHz": s["position"] for s in truth["sources"]}
    positions = {"phase centre": (0.0, 0.0), **marks}
    assert isinstance(amp_phase_check(data, positions, plots_dir=str(plots), line_ghz=39.9), Figure)
    axis = np.arange(-40.0, 40.1, 1.5)
    moment8, continuum, sigma = dirty_maps(data, axis, axis)
    assert isinstance(
        dirty_maps_check(axis, axis, moment8, continuum, sigma, plots_dir=str(plots), marks=marks), Figure
    )


def _circle_mask(axis, radius):
    """True where the (dra, ddec) grid point lies inside ``radius``, as a dra x ddec array."""
    return np.hypot(*np.meshgrid(axis, axis, indexing="ij")) < radius


def test_pointings_figure(plots):
    """One position through four pointings: three of different S/N (the last partly covering), one not covering."""
    from luv_finder.plotting import pointings_check
    from luv_finder.utils import primary_beam

    line_ghz, noise, dish = 39.9, 5e-5, 12.0
    freqs = np.linspace(39.5, 40.3, 160)
    offsets = np.linspace(0, 160, 300)
    beam = (offsets, primary_beam(offsets, line_ghz * 1e9, dish))
    pointings = {}
    fields = (("field 0", 20.0, 8.0), ("field 1", 70.0, 5.0), ("field 2", 110.0, 3.0), ("field 3", 150.0, 2.0))
    for label, distance, snr in fields:
        pb = float(primary_beam(distance, line_ghz * 1e9, dish))
        raw = snr * noise * np.exp(-0.5 * ((freqs - line_ghz) / 0.04) ** 2)
        pointings[label] = {
            "distance": distance,
            "pb": pb,
            "flux": raw / pb,
            "error": np.full_like(freqs, noise / pb),
            "snr": raw / noise,
        }
    for key in ("flux", "error", "snr"):
        pointings["field 2"][key][:30] = np.nan
        pointings["field 3"][key][:] = np.nan
    flux, error = (np.array([p[k] for p in pointings.values()]) for k in ("flux", "error"))
    weight = 1 / error**2
    combined = {
        "flux": np.nansum(weight * flux, axis=0) / np.nansum(weight, axis=0),
        "error": 1 / np.sqrt(np.nansum(weight, axis=0)),
    }
    combined["snr"] = combined["flux"] / combined["error"]
    fig = pointings_check(freqs, pointings, combined, beam, plots_dir=str(plots), line_ghz=line_ghz)
    assert isinstance(fig, Figure)


def test_dirty_maps_with_noise_map(plots):
    """Mosaic-style dirty maps: per-pixel noise rising outwards, NaN outside the covered region."""
    from luv_finder.plotting import dirty_maps_check

    axis = np.arange(-40.0, 40.1, 1.5)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    rng = np.random.default_rng(0)
    sigma = 1e-5 * (1 + (np.hypot(x, y) / 30) ** 2)
    moment8 = 4 + 10 * np.exp(-0.5 * (np.hypot(x - 10, y + 5) / 3) ** 2) + rng.normal(size=x.shape)
    continuum = sigma * (12 * np.exp(-0.5 * (np.hypot(x + 15, y - 10) / 4) ** 2) + rng.normal(size=x.shape))
    outside = ~_circle_mask(axis, 35)
    for image in (sigma, moment8, continuum):
        image[outside] = np.nan
    fig = dirty_maps_check(axis, axis, moment8, continuum, sigma, plots_dir=str(plots), name="dirty_maps_noise_map")
    assert isinstance(fig, Figure)


def test_search_figures_with_nan(plots):
    """S/N map and noise statistics when positions outside every primary beam are NaN."""
    from luv_finder.plotting import noise_check, snr_map

    axis = np.arange(-40.0, 40.1, 2.0)
    rng = np.random.default_rng(1)
    outside = ~_circle_mask(axis, 30).ravel()
    responses = []
    for _ in range(2):
        response = rng.normal(size=(len(axis) ** 2, 50))
        response[outside] = np.nan
        responses.append(response)
    mf = SimpleNamespace(axes={"dra": axis, "ddec": axis}, response=responses[0], response_jackknife=responses[1])
    assert isinstance(snr_map(mf, plots_dir=str(plots), name="snr_map_nan", marks={"x": (5.0, 5.0)}), Figure)
    assert isinstance(noise_check(mf, plots_dir=str(plots), name="noise_nan"), Figure)


def test_response_shape_figure(plots):
    """Null correlation of smoothed white noise, and cuts through a blob that has the same shape."""
    from scipy.ndimage import gaussian_filter

    from luv_finder.catalogue import response_correlation
    from luv_finder.matchedfilter import SearchResult
    from luv_finder.model import C_KMS, FWHM_TO_SIGMA
    from luv_finder.plotting import response_shape_check

    step, dv, widths, shape = 0.5, 50.0, (100.0, 200.0), (60, 60, 64)
    sigmas = [(4.0, 2.0, w / dv * FWHM_TO_SIGMA) for w in widths]
    axes = {"dra": (step * np.arange(shape[0])).tolist(), "ddec": (step * np.arange(shape[1])).tolist()}
    axes |= {"bmin": [0.0], "bmaj": [0.0], "pa": [0.0], "width": list(widths)}
    freqs = 40e9 * (1 + dv / C_KMS * (np.arange(shape[2]) - shape[2] // 2))

    def noise(seed):
        white = np.random.default_rng(seed).standard_normal(shape)
        return np.stack([gaussian_filter(white, s, mode="wrap") for s in sigmas], 2)

    def blob(sigma, peak=10.0):
        offsets = np.ogrid[tuple(slice(-n // 2, n - n // 2) for n in shape)]
        return peak * np.exp(-sum((o / s) ** 2 / 4 for o, s in zip(offsets, sigma, strict=True)))

    def search(snr, jackknife=None):
        error = np.full(snr.shape, 1e-3)
        return SearchResult(axes, freqs, snr, error, np.ones(shape), jackknife)

    noisy, jackknife = (n / n.std(axis=(0, 1, 3), keepdims=True) for n in map(noise, (1, 2)))
    data = noisy + np.stack([blob(s) for s in sigmas], 2)
    result = search(data, search(jackknife))
    nc = response_correlation(result, max_lag=10.0)
    peaks = []
    for t in range(len(widths)):
        i, j, k = np.unravel_index(result.snr[:, :, t].argmax(), shape)
        peaks.append((i, j, t, k))
    fig = response_shape_check(nc, result, peaks, plots_dir=str(plots))
    assert isinstance(fig, Figure)


def _reliability_catalogue(data, jackknife, negative, floor=4.0, k=3.0, fidelity_min=0.6):
    """A catalogue table with the columns and meta that ``reliability_check`` reads, from group S/N lists."""
    from astropy.table import Table

    from luv_finder.catalogue import _reliability

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        columns, meta = _reliability(dict(data=data, jackknife=jackknife, negative=negative), floor, k, fidelity_min)
    return Table({"snr": data} | columns, meta=meta)


def test_reliability_figure(plots):
    """Counts, likelihood ratio and fidelity: a normal catalogue, a NaN fit without jackknife, and no detections."""
    from luv_finder.plotting import reliability_check

    rng = np.random.default_rng(3)

    def noise(n):
        return 4.0 + np.abs(rng.normal(0, 0.6, n))

    bright = np.array([5.1, 5.5, 5.9, 6.4, 7.2, 8.0])
    data, jackknife, negative = np.r_[noise(160), bright], noise(150), noise(140)
    normal = _reliability_catalogue(data, jackknife, negative)
    assert normal["detected"].any()
    nan_fit = _reliability_catalogue(bright, np.array([]), np.array([]))
    assert np.isnan(nan_fit.meta["noise_sigma"])
    nothing = _reliability_catalogue(noise(160), jackknife, negative)
    nothing["detected"] = False
    empty = _reliability_catalogue(np.array([]), np.array([]), np.array([]))
    for name, cat in (("normal", normal), ("nan_fit", nan_fit), ("nothing", nothing), ("empty", empty)):
        fig = reliability_check(cat, plots_dir=str(plots), name=f"reliability_{name}")
        assert isinstance(fig, Figure)


def test_weights_figure(data, plots):
    """The scatter of the visibilities against the spread their weights predict, as read, reweighted and pooled."""
    (chunk,) = data.chunks
    fixed = reweighted([chunk])
    fig = weights_check([chunk, fixed[0]], ["as stored", "reweighted"], plots_dir=str(plots), pooled=fixed * 2)
    assert isinstance(fig, Figure)


def test_figures_are_saved_only_when_asked(tmp_path):
    """A figure is returned either way; ``<plots_dir>/<name>.png`` is written only with a ``plots_dir``."""
    from luv_finder.plotting import mosaic_layout

    assert isinstance(mosaic_layout({0: (0.0, 0.0)}, 60.0), Figure)
    assert not list(tmp_path.iterdir())
    mosaic_layout({0: (0.0, 0.0)}, 60.0, plots_dir=str(tmp_path), name="layout")
    assert (tmp_path / "layout.png").is_file()


def test_style_is_scoped():
    """The package style applies inside :func:`style` and the plotting functions, not outside."""
    import matplotlib.pyplot as plt
    from matplotlib.markers import TICKUP

    from luv_finder.plotting import mosaic_layout, style

    before = plt.rcParams["xtick.direction"]
    with style():
        assert plt.rcParams["xtick.direction"] == "in"
    fig = mosaic_layout({0: (0.0, 0.0)}, 60.0)
    assert plt.rcParams["xtick.direction"] == before
    assert fig.axes[0].xaxis.get_major_ticks()[0].tick1line.get_marker() == TICKUP  # bottom ticks point in


def test_peak_snr_map(plots):
    """Two units on lattices of different steps, combined by their maximum on the finer one."""
    from luv_finder.plotting import peak_snr_map

    rng = np.random.default_rng(1)
    coarse, fine = np.arange(-30.0, 30.1, 1.0), np.arange(-20.0, 20.01, 0.6)
    maps = [(x, x, np.abs(rng.normal(0, 1.5, (x.size, x.size)))) for x in (coarse, fine)]
    maps[1][2][40, 30] = 9.0
    marks = {"#1": (fine[40], fine[30])}
    fig = peak_snr_map(maps, marks=marks, plots_dir=str(plots), name="peak_snr")
    assert isinstance(fig, Figure)
    (mesh,) = fig.axes[0].collections
    assert np.nanmax(mesh.get_array()) == 9.0


def test_filter_responses(plots):
    """Groups from two units, with their own frequencies and widths, and the jackknife."""
    from luv_finder.plotting import filter_responses

    rng = np.random.default_rng(2)
    freqs = [np.linspace(97e9, 99e9, 300), np.linspace(230e9, 232e9, 120)]
    widths = [np.array([100.0, 300.0, 500.0])] * 2
    snr, jk = [], []
    for f in freqs:
        line = np.exp(-0.5 * ((f - f.mean()) / (2e-4 * f.mean())) ** 2)
        snr.append(rng.normal(0, 1, (3, f.size)) + np.array([[5.0], [7.0], [6.0]]) * line)
        jk.append(rng.normal(0, 1, (3, f.size)))
    fig = filter_responses(freqs, snr, widths, ["#1", "#2"], jk=jk, plots_dir=str(plots), name="filter_responses")
    assert isinstance(fig, Figure) and len([ax for ax in fig.axes if ax.lines]) == 2


def test_erf_curve_recovers_centre_and_width():
    from scipy import special

    from luv_finder.plotting import erf_curve

    snr = np.arange(4.0, 9.0, 0.25)
    centre, width = erf_curve(snr, 0.5 * (1 + special.erf((snr - 6.0) / (np.sqrt(2) * 0.4))))
    assert (centre, width) == (pytest.approx(6.0), pytest.approx(0.4))
    assert np.isnan(erf_curve([5.0, 6.0], [0.0, 1.0])).all()
