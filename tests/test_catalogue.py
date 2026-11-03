import dataclasses
import json

import numpy as np
import pytest
from astropy.table import Table

from luv_finder.catalogue import (
    MIN_NOISE,
    NOISE_FLOOR,
    ResponseCorrelation,
    _reliability,
    bright,
    catalogue,
    fidelity,
    groups,
    jackknife_spread,
    likelihood,
    merge_catalogues,
    noise_above,
    peak_exponent,
    response_correlation,
)
from luv_finder.data import sky_offset
from luv_finder.matchedfilter import SearchResult
from luv_finder.model import C_KMS, FWHM_TO_SIGMA, covariance, envelope

STEP = 0.5  # arcsec
NU, DV = 40e9, 50.0  # Hz, km/s per channel
WIDTHS = (100.0, 200.0)  # km/s: FWHM of 2 and 4 channels at NU
SHAPE = (128, 128, 128)  # n_dra, n_ddec, n_chan
SPREAD = 1.6  # jackknife S/N spread, as from Hanning-correlated channels
#: HWHM of the correlation of white noise smoothed by a unit-sigma Gaussian: exp(-x^2 / 4) = 1/2
HWHM = 2 * np.sqrt(np.log(2))


def _smooth(noise, major, minor, pa, chan, steps=(STEP, STEP)):
    """White ``(n_dra, n_ddec, n_chan)`` noise smoothed periodically, to unit variance.

    The kernel is a Gaussian of sigma ``major`` x ``minor`` arcsec, the major axis at ``pa`` deg
    east of north, and ``chan`` channels; ``steps`` are the lattice steps in arcsec, signed.
    """
    q_ra, q_dec, q_chan = np.meshgrid(
        np.fft.fftfreq(SHAPE[0], steps[0]), np.fft.fftfreq(SHAPE[1], steps[1]), np.fft.fftfreq(SHAPE[2]), indexing="ij"
    )
    transfer = envelope(q_ra, q_dec, covariance(major / FWHM_TO_SIGMA, minor / FWHM_TO_SIGMA, pa)) * np.exp(
        -2 * np.pi**2 * (chan * q_chan) ** 2
    )
    x = np.fft.ifftn(np.fft.fftn(noise) * transfer).real
    return x / x.std()


def _field(major, minor, pa, chan, steps=(STEP, STEP), seed=0):
    """Jackknife S/N of two templates, independent realisations of the same smoothing."""
    rng = np.random.default_rng(seed)
    return SPREAD * np.stack([_smooth(rng.standard_normal(SHAPE), major, minor, pa, chan, steps) for _ in WIDTHS], 2)


def _result(jackknife, step_ra=STEP, channels=None):
    """Search of white noise on a uniform lattice, error 1 mJy, whose jackknife S/N is ``jackknife``.

    ``channels`` place the channels on a grid of ``DV`` steps (default: contiguous).
    """
    n_dra, n_ddec, _, n_chan = jackknife.shape
    channels = np.arange(n_chan) if channels is None else channels
    dra = step_ra * np.arange(n_dra)
    axes = {"dra": dra.tolist(), "ddec": (STEP * np.arange(n_ddec)).tolist(), "bmin": [0.0], "bmaj": [0.0]}
    axes |= {"pa": [0.0], "width": list(WIDTHS)}
    freqs = NU * (1 + DV / C_KMS * (channels - n_chan // 2))

    def search(snr):
        error = np.full(snr.shape, 1e-3)
        return SearchResult(axes, freqs, snr, error, np.ones((n_dra, n_ddec, n_chan)))

    data = SPREAD * np.random.default_rng(1).standard_normal(jackknife.shape)
    return dataclasses.replace(search(data), jackknife=search(jackknife))


@pytest.fixture(scope="module")
def result():
    return _result(_field(2.0, 1.0, 90.0, 2.0))


def _angle(a, b):
    """Difference of two position angles, modulo 180 deg."""
    return (a - b + 90) % 180 - 90


def test_jackknife_spread(result):
    np.testing.assert_allclose(jackknife_spread(result), SPREAD, rtol=0.05)
    with pytest.raises(ValueError, match="jackknife"):
        jackknife_spread(dataclasses.replace(result, jackknife=None))


@pytest.mark.parametrize("pa, step_ra", [(90.0, STEP), (0.0, STEP), (30.0, -STEP)])
def test_response_correlation(pa, step_ra):
    """Smoothing by sigma s gives rho = exp(-d^2 / (4 s^2)), whatever the orientation of the dra axis."""
    major, minor, chan = 2.0, 1.0, 2.0
    nc = response_correlation(_result(_field(major, minor, pa, chan, (step_ra, STEP)), step_ra), max_lag=10.0)
    assert nc.spatial.shape == (len(WIDTHS), len(nc.lags_ra), len(nc.lags_dec)) == (2, 41, 41)
    np.testing.assert_allclose(nc.spatial[:, 20, 20], 1.0)
    np.testing.assert_allclose(nc.hwhm_major, HWHM * major, rtol=0.1)
    np.testing.assert_allclose(nc.hwhm_minor, HWHM * minor, rtol=0.1)
    assert np.all(np.abs(_angle(nc.pa, pa)) < 10)
    # default channel lags: 4 x the widest template's FWHM, 4 channels
    assert nc.spectral.shape == (2, 17)
    np.testing.assert_allclose(nc.spectral_hwhm, HWHM * chan, rtol=0.1)
    np.testing.assert_allclose(nc.spectral_hwhm_kms, HWHM * chan * DV, rtol=0.1)
    assert np.all(nc.sidelobe < 0.05)
    # the templates are independent realisations
    np.testing.assert_allclose(nc.templates, np.eye(2), atol=0.03)


def test_template_correlation():
    """Two Gaussian smoothings s1, s2 of the same noise correlate by sqrt(2 s1 s2 / (s1^2 + s2^2))."""
    s1, s2 = 1.5, 4.0
    noise = np.random.default_rng(2).standard_normal(SHAPE)
    jackknife = np.stack([_smooth(noise, 1.0, 1.0, 0.0, s) for s in (s1, s2)], 2)
    nc = response_correlation(_result(jackknife), max_lag=5.0)
    np.testing.assert_allclose(np.diag(nc.templates), 1.0)
    np.testing.assert_allclose(nc.templates[0, 1], np.sqrt(2 * s1 * s2 / (s1**2 + s2**2)), atol=0.03)
    np.testing.assert_allclose(nc.spectral_hwhm, HWHM * np.array([s1, s2]), rtol=0.1)


def test_spectral_lags_stay_in_windows():
    """Hanning-smoothed channels correlate by 2/3 and 1/6 at lags 1 and 2, measured within each window.

    Pairs across the gap between two windows are independent; counting them would bias rho(1) by 1/16.
    """
    n_window, n_chan = 8, 16
    noise = np.random.default_rng(3).standard_normal((*SHAPE[:2], len(WIDTHS), n_window, n_chan + 2))
    hanning = 0.25 * noise[..., :-2] + 0.5 * noise[..., 1:-1] + 0.25 * noise[..., 2:]
    channels = (np.arange(n_window)[:, None] * (n_chan + 3) + np.arange(n_chan)).ravel()
    nc = response_correlation(_result(hanning.reshape(*SHAPE[:2], len(WIDTHS), -1), channels=channels), max_lag=2.0)
    np.testing.assert_allclose(nc.spectral[:, 1:4], np.broadcast_to([2 / 3, 1 / 6, 0.0], (2, 3)), atol=0.01)


def test_masked_positions(result):
    """A NaN corner and hole, in the data and the jackknife, do not bias the correlation."""
    i, j = np.indices(SHAPE[:2])
    cut = ((i < 48) & (j < 48)) | (np.hypot(i - 90, j - 80) < 10)

    def mask(r):
        return dataclasses.replace(r, snr=np.where(cut[:, :, None, None], np.nan, r.snr))

    full = response_correlation(result, max_lag=10.0)
    masked = response_correlation(dataclasses.replace(mask(result), jackknife=mask(result.jackknife)), max_lag=10.0)
    for key in ("hwhm_major", "hwhm_minor", "spectral_hwhm"):
        np.testing.assert_allclose(getattr(masked, key), getattr(full, key), rtol=0.1)
    assert np.all(masked.sidelobe < 0.05)


#: Grid of the grouping tests: 40 x 40 positions, 64 channels of ``DV``.
GRID = _result(np.zeros((40, 40, len(WIDTHS), 64)))


def _blob(x, y, chan):
    """Unit Gaussian of sigma 0.5 arcsec and 1 channel at (x, y) arcsec and channel ``chan`` on ``GRID``."""
    dra, ddec, k = np.meshgrid(GRID.axes["dra"], GRID.axes["ddec"], np.arange(len(GRID.freqs)), indexing="ij")
    return np.exp(-((dra - x) ** 2 + (ddec - y) ** 2) / (2 * 0.5**2) - (k - chan) ** 2 / 2)


def _snr(*blobs):
    """S/N on ``GRID`` of blobs ``(x, y, lines)``, ``lines`` holding (channel, amplitude) per template."""
    snr = np.zeros(GRID.snr.shape)
    for x, y, lines in blobs:
        for t, (chan, amplitude) in enumerate(lines):
            snr[:, :, t] += amplitude * _blob(x, y, chan)
    return snr


def _nc(major=6.0, minor=3.0, pa=30.0, chan=4.0):
    """Noise correlation of both templates with these half-power semi-axes (arcsec), PA (deg) and HWHM (channels)."""
    every = np.ones(len(WIDTHS))
    return ResponseCorrelation(
        **dict.fromkeys(("spatial", "lags_ra", "lags_dec", "spectral", "templates")),
        hwhm_major=major * every,
        hwhm_minor=minor * every,
        pa=pa * every,
        spectral_hwhm=chan * every,
        spectral_hwhm_kms=chan * DV * every,
        sidelobe=0 * every,
    )


def test_groups_merge_the_maxima_of_one_line():
    """Templates peaking 3 channels apart give two maxima of the best-template S/N, one group; NaN is skipped."""
    snr = _snr((8.0, 8.0, ((20, 10.0), (23, 9.0))), (14.0, 3.0, ((40, 6.0), (40, 7.0))))
    snr[:4, :4] = np.nan
    found = groups(snr, GRID, _nc())
    assert [(g.i, g.j, g.t, g.k, g.n_peaks) for g in found] == [(16, 16, 0, 20, 2), (28, 6, 1, 40, 1)]
    assert [g.snr for g in found] == pytest.approx([10.0, 7.0])
    assert groups(snr, GRID, _nc(chan=2.9))[1].k == 23


def test_groups_separate_lines_in_frequency():
    """Lines 3 template widths apart are two groups, and so are neighbouring channels of two windows."""
    width = WIDTHS[1] / DV  # channels
    snr = _snr((8.0, 8.0, ((20, 10.0), (20, 9.0))), (8.0, 8.0, ((20 + 3 * width, 8.0), (20 + 3 * width, 8.0))))
    assert [g.k for g in groups(snr, GRID, _nc())] == [20, 20 + 3 * width]
    windows = _result(np.zeros((40, 40, 2, 64)), channels=np.r_[np.arange(32), 100 + np.arange(32)])
    snr = _snr((8.0, 8.0, ((30, 10.0), (30, 9.0))), (8.0, 8.0, ((33, 8.0), (33, 8.0))))
    assert [g.k for g in groups(snr, GRID, _nc())] == [30]
    assert [g.k for g in groups(snr, windows, _nc())] == [30, 33]


@pytest.mark.parametrize(
    "offset, n_groups",
    [
        ((1.5, 3.0), 1),  # along the major axis at PA 30 deg: 0.56 HWHM
        ((3.0, -1.5), 2),  # along the minor axis: 1.12 HWHM
        ((-3.0, 1.5), 2),
    ],
)
def test_groups_follow_the_half_power_ellipse(offset, n_groups):
    snr = _snr((8.0, 8.0, ((20, 10.0), (20, 9.0))), (8.0 + offset[0], 8.0 + offset[1], ((20, 7.0), (20, 6.0))))
    assert len(groups(snr, GRID, _nc())) == n_groups


def _peaks(rng, n, m=2.0, floor=NOISE_FLOOR):
    """S/N of ``n`` noise peaks drawn from the peak-count model ``N(>= u) ~ u^m exp(-u^2 / 2)`` above ``floor``."""
    u = np.linspace(floor, 12.0, 20001)
    survival = noise_above(u, 1, m, floor)
    return np.interp(rng.uniform(survival[-1], 1.0, n), survival[::-1], u[::-1])


def _counts(snr, at):
    return np.sum(np.asarray(snr)[None, :] >= np.asarray(at)[:, None], axis=1)


@pytest.mark.parametrize("m", [1.0, 2.0, 3.0])
def test_peak_exponent_comes_back(m):
    rng = np.random.default_rng(int(m))
    assert peak_exponent(_peaks(rng, 20000, m), dim=3) == pytest.approx(m, abs=0.1)
    assert noise_above(NOISE_FLOOR, 500, m) == pytest.approx(500)


def test_few_peaks_keep_the_dimension():
    """Twenty peaks of a 2D field give an exponent near 1 under the prior, where ML alone scatters by ~1."""
    rng = np.random.default_rng(8)
    fits = [peak_exponent(_peaks(rng, 20, 1.0), dim=2) for _ in range(200)]
    assert np.median(fits) == pytest.approx(1.0, abs=0.2) and np.std(fits) < 0.8 and min(fits) > -2


def test_peak_exponent_needs_noise():
    with pytest.warns(UserWarning, match="too few"):
        assert np.isnan(peak_exponent([3.0] * (MIN_NOISE - 1), dim=3))


def test_likelihood_of_noise_is_one():
    rng = np.random.default_rng(4)
    noise, data = _peaks(rng, 30000), _peaks(rng, 30000)
    data = data[data >= 4.0]
    ratio = likelihood(data, lambda u: noise_above(u, noise.size, peak_exponent(noise, dim=3)))
    n_data = _counts(data, data)
    moderate = n_data >= 50
    assert moderate.sum() > 100
    assert np.all(np.abs(ratio - 1)[moderate] < 4 / np.sqrt(n_data[moderate]))


def test_likelihood_of_lines_is_large():
    """The model, not the brightest noise peak, sets the ratio: a line has no lower limit, and no noise means none."""
    rng = np.random.default_rng(5)
    noise = _peaks(rng, 30000)
    data = np.concatenate([(x := _peaks(rng, 30000))[x >= 4.0], rng.uniform(6.0, 12.0, 60)])
    ratio = likelihood(data, lambda u: noise_above(u, noise.size, peak_exponent(noise, dim=3)))
    assert np.all(np.isfinite(ratio)) and np.all(ratio[data >= 6.0] > 10)
    assert np.all(likelihood([4.5, 5.0], lambda u: noise_above(u, 0, 2.0)) == np.inf)


def test_fidelity_from_the_noise_model():
    """Data 5% above the noise near the floor (as at 1.2 mm) leave lines at 1; every group gets its bin's value."""
    rng = np.random.default_rng(6)
    noise = _peaks(rng, 200000)
    data = np.concatenate([(x := _peaks(rng, 210000))[x >= 4.0], rng.uniform(4.0, 10.0, 600)])
    model = lambda u: noise_above(u, noise.size, peak_exponent(noise, dim=3))  # noqa: E731
    values, bins = fidelity(data, noise[noise >= 4.0], model, floor=4.0)
    assert np.all(values[data > 7.0] > 0.95) and np.all(values[data < 4.5] < 0.2)
    centre = 4.0 + 0.25 * ((data - 4.0) // 0.25 + 0.5)
    np.testing.assert_array_equal(values, bins["model"][np.searchsorted(bins["snr"], centre - 0.1)])
    assert bins["n_data"].sum() == data.size
    np.testing.assert_allclose((bins["snr"] - 4.0) / 0.25 % 1, 0.5)


def _one_kernel(snr):
    """Every group of ``snr`` found with one template."""
    return {name: np.zeros((len(values), 4)) for name, values in snr.items()}


def _noise_and_lines(rng, n=20000, lines=40):
    noise = _peaks(rng, n)
    snr = {"data": np.concatenate([(x := _peaks(rng, n))[x >= 4.0], rng.uniform(5.0, 8.0, lines)])}
    return snr | {"jackknife": noise, "negative": noise}


def test_detection_rule_or_contains_and():
    snr = _noise_and_lines(np.random.default_rng(7))
    kernels = _one_kernel(snr)
    either, both = (_reliability(snr, kernels, 3, 4.0, 3.0, 0.6, rule)[0]["detected"] for rule in ("or", "and"))
    assert both.sum() < either.sum() and np.all(either[both])
    with pytest.raises(ValueError, match="rule"):
        _reliability(snr, kernels, 3, 4.0, 3.0, 0.6, "xor")


def test_nothing_is_detected_without_a_noise_model():
    """Too few noise peaks: no model, so no likelihood ratio or fidelity passes, whatever the rule."""
    snr = {"data": np.array([9.0, 6.0, 4.5]), "jackknife": np.array([3.0, 2.6]), "negative": np.array([])}
    for rule in ("or", "and"):
        with pytest.warns(UserWarning, match="too few"):
            columns, meta = _reliability(snr, _one_kernel(snr), 3, 4.0, 3.0, 0.6, rule)
        assert not columns["detected"].any() and np.isnan(meta["noise_exponent"])


def test_each_kernel_is_tested_against_its_own_noise():
    """A broad kernel's group is tested against the broad kernel's few noise peaks, not the narrow one's many."""
    rng = np.random.default_rng(3)
    narrow, broad = _peaks(rng, 40000), _peaks(rng, 400)
    snr = {"data": np.r_[(x := _peaks(rng, 40000))[x >= 4.0], 5.0], "jackknife": np.r_[narrow, broad]}
    snr["negative"] = snr["jackknife"]
    templates = np.array([[0, 0, 0, 100.0], [0, 0, 0, 500.0]])
    label = {
        "data": np.r_[np.zeros(snr["data"].size - 1, int), 1],
        "jackknife": np.r_[np.zeros(40000, int), np.ones(400, int)],
    }
    kernels = {name: templates[label.get(name, label["jackknife"])] for name in snr}
    per_kernel = _reliability(snr, kernels, 3, 4.0, 3.0, 0.6)
    pooled = _reliability(snr, _one_kernel(snr), 3, 4.0, 3.0, 0.6)
    assert per_kernel[0]["fidelity"][-1] > 0.6 > pooled[0]["fidelity"][-1]
    assert per_kernel[0]["detected"][-1] and not pooled[0]["detected"][-1]
    assert [k["n_jackknife"] for k in per_kernel[1]["kernels"]] == [40000, 400]


def test_catalogue_of_the_fixture(data, truth, blind_catalogue, tmp_path):
    """A blind lattice of +-30 arcsec finds both injected lines, each once, and detects them."""
    cat = blind_catalogue.copy()
    assert cat["id"].tolist() == list(range(1, len(cat) + 1))
    assert np.all(np.diff(cat["snr"]) <= 0) and cat["snr"].tolist() == cat.meta["snr_data"]
    channel = np.median(np.diff(data.freqs)) / 1e9
    for src in truth["sources"]:
        x, y = src["position"]
        near = (np.hypot(cat["dra"] - x, cat["ddec"] - y) < data.metadata.minresolution() / 2) & (
            np.abs(cat["freq_ghz"] - src["line"]["mean"]) <= channel
        )
        assert near.sum() == 1
        assert cat["detected"][near][0]
    np.testing.assert_allclose(
        sky_offset(np.radians([cat["ra"], cat["dec"]]), data.metadata.ref), [cat["dra"], cat["ddec"]], atol=1e-6
    )

    path = tmp_path / "catalogue.ecsv"
    cat.write(path)
    back = Table.read(path)
    assert back.colnames == cat.colnames
    for name in cat.colnames:
        np.testing.assert_array_equal(back[name], cat[name])
    assert back["line_flux"].unit == "Jy km / s"
    assert json.dumps(dict(back.meta), sort_keys=True) == json.dumps(dict(cat.meta), sort_keys=True)

    one, two = merge_catalogues([cat]), merge_catalogues([back, cat])
    for name in ("snr", "likelihood", "fidelity", "detected"):
        np.testing.assert_array_equal(one[name], cat[name])
    assert two["id"].tolist() == list(range(1, 2 * len(cat) + 1)) and len(two.meta["parts"]) == 2
    assert sorted(two.meta["snr_jackknife"]) == sorted(2 * cat.meta["snr_jackknife"])
    np.testing.assert_allclose(two.meta["noise_exponent"], cat.meta["noise_exponent"], atol=0.05)
    np.testing.assert_allclose(two["likelihood"][::2], cat["likelihood"], rtol=0.1)  # twice the peaks, same prior


def test_bright_takes_the_detections_whose_sidelobes_reach_the_floor():
    cat = Table({"snr": [80.0, 30.0, 9.0, 50.0], "detected": [True, True, True, False]}, meta={"sidelobe": [0.1, 0.25]})
    assert bright(cat, 4.0).tolist() == [True, True, False, False]  # from 80 * 0.25 = 20, the undetected aside
    assert not bright(cat[1:3], 8.0).any()  # 30 * 0.25 = 7.5 < 8: no sidelobe reaches the floor
    assert not bright(cat[3:], 4.0).any()


def test_chunks_of_a_spectrum_share_its_groups_out():
    """Catalogues of two bands, split at any frequency, hold the groups of the whole one, each once."""
    result = _result(_field(2.0, 1.0, 90.0, 2.0))
    whole = catalogue(result)
    mid = result.freqs[result.freqs.size // 2 + 1]
    parts = [catalogue(result, band=band) for band in ((-np.inf, mid), (mid, np.inf))]
    for name in ("data", "jackknife", "negative"):
        pooled = sorted(v for p in parts for v in p.meta[f"snr_{name}"])
        assert pooled == sorted(whole.meta[f"snr_{name}"]), name
