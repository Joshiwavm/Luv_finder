"""Diagnostic figures. Only run with --plots; they are for the eye, not for CI.

Assertions here are deliberately weak: the point is to produce something to look
at. The numerical checks live in the other test modules.
"""

import warnings
from pathlib import Path
from types import SimpleNamespace

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
    path = pointings_check(freqs, pointings, combined, beam, plots_dir=str(plots), line_ghz=line_ghz)
    assert path.endswith(".png")


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
    path = dirty_maps_check(axis, axis, moment8, continuum, sigma, plots_dir=str(plots), name="dirty_maps_noise_map")
    assert path.endswith(".png")


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
    assert snr_map(mf, plots_dir=str(plots), name="snr_map_nan", marks={"x": (5.0, 5.0)}).endswith(".png")
    assert noise_check(mf, plots_dir=str(plots), name="noise_nan").endswith(".png")


def test_response_shape_figure(plots):
    """Null correlation of smoothed white noise, and cuts through a blob that has the same shape."""
    from scipy.ndimage import gaussian_filter

    from luv_finder.catalogue import noise_correlation
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
        return SearchResult(axes, freqs, snr, snr * error, error, np.ones(shape), jackknife)

    noisy, jackknife = (n / n.std(axis=(0, 1, 3), keepdims=True) for n in map(noise, (1, 2)))
    data = noisy + np.stack([blob(s) for s in sigmas], 2)
    result = search(data, search(jackknife))
    nc = noise_correlation(result, max_lag=10.0)
    peaks = []
    for t in range(len(widths)):
        i, j, k = np.unravel_index(result.snr[:, :, t].argmax(), shape)
        peaks.append((i, j, t, k))
    path = response_shape_check(nc, result, peaks, plots_dir=str(plots))
    assert Path(path).is_file()


def _reliability_catalogue(data, jackknife, negative, floor=4.0, k=3.0, fidelity_min=0.6):
    """A catalogue table with the columns and meta that ``reliability_check`` reads, from group S/N lists."""
    from astropy.table import Table
    from scipy import special

    from luv_finder.catalogue import fidelity, likelihood

    ratio, lower = likelihood(data, jackknife)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values, (centre, sigma), bins = fidelity(data, jackknife, floor)
    detected = ((ratio >= k) | lower) & ((values >= fidelity_min) | np.isnan(values))
    meta = {"floor": floor, "k": k, "fidelity_min": fidelity_min, "fidelity_centre": centre, "fidelity_sigma": sigma}
    meta["snr_likelihood"] = float(np.min(data[ratio >= k], initial=np.inf)) if np.any(ratio >= k) else np.nan
    meta["snr_fidelity"] = float(centre + sigma * special.erfinv(2 * fidelity_min - 1))
    meta |= {"snr_data": data.tolist(), "snr_jackknife": jackknife.tolist(), "snr_negative": negative.tolist()}
    meta["fidelity_bins"] = {key: value.tolist() for key, value in bins.items()}
    columns = {"snr": data, "likelihood": ratio, "likelihood_lower_limit": lower, "fidelity": values}
    return Table(columns | {"detected": detected}, meta=meta)


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
    nan_fit.meta |= {"fidelity_centre": np.nan, "fidelity_sigma": np.nan, "snr_fidelity": np.nan}
    nothing = _reliability_catalogue(noise(160), jackknife, negative)
    nothing["detected"] = False
    empty = _reliability_catalogue(np.array([]), np.array([]), np.array([]))
    for name, cat in (("normal", normal), ("nan_fit", nan_fit), ("nothing", nothing), ("empty", empty)):
        path = reliability_check(cat, plots_dir=str(plots), name=f"reliability_{name}")
        assert Path(path).is_file()
