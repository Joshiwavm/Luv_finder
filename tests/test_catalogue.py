import dataclasses
import json

import numpy as np
import pytest
from astropy.table import Table

from luv_finder import MatchedFilter
from luv_finder.catalogue import (
    NoiseCorrelation,
    catalogue,
    fidelity,
    groups,
    jackknife_spread,
    likelihood,
    noise_correlation,
)
from luv_finder.data import sky_offset
from luv_finder.matchedfilter import SearchResult, build_grid, grid_model, pb_corrected
from luv_finder.model import C_KMS, envelope

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
    transfer = envelope(q_ra, q_dec, major, minor, pa) * np.exp(-2 * np.pi**2 * (chan * q_chan) ** 2)
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
        return SearchResult(axes, freqs, snr, snr * error, error, np.ones((n_dra, n_ddec, n_chan)))

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
def test_noise_correlation(pa, step_ra):
    """Smoothing by sigma s gives rho = exp(-d^2 / (4 s^2)), whatever the orientation of the dra axis."""
    major, minor, chan = 2.0, 1.0, 2.0
    nc = noise_correlation(_result(_field(major, minor, pa, chan, (step_ra, STEP)), step_ra), max_lag=10.0)
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
    nc = noise_correlation(_result(jackknife), max_lag=5.0)
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
    nc = noise_correlation(_result(hanning.reshape(*SHAPE[:2], len(WIDTHS), -1), channels=channels), max_lag=2.0)
    np.testing.assert_allclose(nc.spectral[:, 1:4], np.broadcast_to([2 / 3, 1 / 6, 0.0], (2, 3)), atol=0.01)


def test_masked_positions(result):
    """A NaN corner and hole, in the data and the jackknife, do not bias the correlation."""
    i, j = np.indices(SHAPE[:2])
    cut = ((i < 48) & (j < 48)) | (np.hypot(i - 90, j - 80) < 10)

    def mask(r):
        return dataclasses.replace(r, snr=np.where(cut[:, :, None, None], np.nan, r.snr))

    full = noise_correlation(result, max_lag=10.0)
    masked = noise_correlation(dataclasses.replace(mask(result), jackknife=mask(result.jackknife)), max_lag=10.0)
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
    return NoiseCorrelation(
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


def _tail(rng, n, scale=0.5, floor=4.0):
    """Group S/N of noise: an exponential tail above ``floor``."""
    return floor + rng.exponential(scale, n)


def _counts(snr, at):
    return np.sum(np.asarray(snr)[None, :] >= np.asarray(at)[:, None], axis=1)


def test_likelihood_of_noise_is_one():
    rng = np.random.default_rng(4)
    data, noise = _tail(rng, 3000), _tail(rng, 3000)
    ratio, lower = likelihood(data, noise)
    n_data, n_noise = _counts(data, data), _counts(noise, data)
    np.testing.assert_array_equal(ratio, n_data / np.maximum(n_noise, 1))
    np.testing.assert_array_equal(lower, n_noise == 0)
    moderate = n_noise >= 100
    assert moderate.sum() > 1000
    assert np.all(np.abs(ratio - 1)[moderate] < 4 * np.sqrt(1 / n_data + 1 / n_noise)[moderate])


def test_likelihood_of_lines_is_large():
    rng = np.random.default_rng(5)
    noise = _tail(rng, 3000)
    data = np.concatenate([_tail(rng, 3000), rng.uniform(6.0, 12.0, 60)])
    ratio, lower = likelihood(data, noise)
    np.testing.assert_array_equal(lower, data > noise.max())
    np.testing.assert_array_equal(ratio[lower], _counts(data, data[lower]))
    assert np.all(ratio[(data >= 7.5) & (data <= 10.0)] > 10)


def test_fidelity_fits_the_crossover():
    """Real lines of uniform density m over noise of density N / s exp(-(snr - 4) / s) match at 4 + s ln(N / (s m))."""
    rng = np.random.default_rng(6)
    n_noise, scale, density = 20000, 0.5, 400
    data = np.concatenate([_tail(rng, n_noise, scale), rng.uniform(4.0, 10.0, 6 * density)])
    values, (centre, sigma), bins = fidelity(data, _tail(rng, n_noise, scale), floor=4.0)
    assert centre == pytest.approx(4 + scale * np.log(n_noise / (scale * density)), abs=0.3)
    assert 0.3 < sigma < 2.0
    assert np.all(values[data > 9.0] > 0.95)
    assert bins["n_data"].sum() == data.size
    np.testing.assert_allclose((bins["snr"] - 4.0) / 0.25 % 1, 0.5)


def test_fidelity_needs_three_bins():
    with pytest.warns(UserWarning, match="too few"):
        values, fit, bins = fidelity([4.1, 4.2, 4.6], [4.05], floor=4.0)
    assert np.isnan(values).all() and np.isnan(fit).all()
    assert bins["n_data"].tolist() == [2, 1] and bins["n_noise"].tolist() == [1, 0]


def test_catalogue_of_the_fixture(data, truth, tmp_path):
    """A blind lattice of +-30 arcsec finds both injected lines, each once, and detects them."""
    lattice = data.metadata.minresolution() / 2 * np.arange(-18, 19)
    mf = MatchedFilter(data, grid_model(build_grid(data, {"dra": lattice, "ddec": lattice})))
    mf.run(jackknife=True)
    cat = catalogue(pb_corrected(mf.result, data), ref=data.metadata.ref)
    assert cat["id"].tolist() == list(range(1, len(cat) + 1))
    assert np.all(np.diff(cat["snr"]) <= 0) and cat["snr"].tolist() == cat.meta["snr_data"]
    channel = np.median(np.diff(data.freqs)) / 1e9
    for src in truth["sources"]:
        x, y = src["position_model"]
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
