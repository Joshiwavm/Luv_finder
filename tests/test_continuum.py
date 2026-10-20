"""The continuum polynomial: one default for search and fit, and mock continua with a spectral index."""

import numpy as np
import pytest

from luv_finder.mock import MockObservation
from luv_finder.model import CONTINUUM_ORDER
from luv_finder.run import Run


@pytest.mark.parametrize("index", [0.0, 3.5, -1.0])
def test_mock_continuum_follows_its_spectral_index(index):
    mock = MockObservation(
        cube_shape=(20, 32, 32),
        freq_center="85GHz",
        dv="55km/s",
        integration_time="10min",
        sources=[{"position": (0.0, 0.0), "continuum": {"snr": 10.0, "index": index}}],
    )
    cube = mock.create_cube()
    df, f_start, _ = mock._freq_axis()
    nu = f_start + df * np.arange(20)
    spectrum = cube[:, 16, 16]
    np.testing.assert_allclose(spectrum / spectrum[10], (nu / nu[10]) ** index, rtol=1e-12)


@pytest.mark.parametrize(("filter_cfg", "order"), [({}, CONTINUUM_ORDER), ({"continuum_order": 2}, 2),
                                                    ({"continuum_order": None}, None)])  # fmt: skip
def test_run_fits_with_the_degree_it_searched_with(tmp_path, filter_cfg, order):
    run = Run({"data": "data.npz", "output": str(tmp_path), "filter": filter_cfg})
    assert run.fit_options == {"continuum_order": order}


# --- a bright line A and a faint B in one window, noiseless: the response is linear, so B's bias is
# the response to A + B (+ continuum) minus the response to B alone (blend_tests, 2026-10-15)

N_CHAN, LINE_A = 128, 20  # 128 channels of 100 km/s; A near the low edge, B up to 12 FWHM above
FWHM_CHANNELS = 3  # 300 km/s lines


def _window(spectra):
    """One unit-weight window holding point sources at the phase centre with the given spectra (Jy per channel)."""
    import dataclasses

    from luv_finder import DataHandler
    from luv_finder.data import Chunk, Metadata
    from luv_finder.utils import C

    r = np.random.default_rng(0)
    nvis, freq0 = 400, 40e9
    freq = freq0 * (1 + (np.arange(N_CHAN) - N_CHAN // 2) * 100 / 299792.458)
    u, v = r.uniform(-3e4, 3e4, (2, nvis)) * C / freq0
    X = np.repeat(np.sum(spectra, axis=0)[:, None], nvis, axis=1).astype(complex)
    chunk = Chunk(0, 0, np.zeros(2), freq, u, v, np.arange(nvis, dtype=float), np.zeros(nvis, dtype=int),
                  np.zeros(nvis, dtype=int), X, np.ones((N_CHAN, nvis), dtype=np.float32))  # fmt: skip
    meta = Metadata(12.0, (0.0, 0.0), float(freq.mean()), float(np.hypot(u, v).max() * freq.max() / C))
    return DataHandler(chunks=[dataclasses.replace(chunk)], metadata=meta)


def _line(channel, snr):
    """A 300 km/s line at ``channel`` of optimal S/N ``snr`` in :func:`_window` (400 unit-weight rows)."""
    profile = np.exp(-0.5 * ((np.arange(N_CHAN) - channel) / (FWHM_CHANNELS / 2.355)) ** 2)
    return profile * snr / np.sqrt(400 * np.sum(profile**2))


def _snr_at(spectra, channel, order):
    from luv_finder import Gaussian, MatchedFilter, Model

    g = Gaussian()
    g.grid = {"dra": 0.0, "ddec": 0.0, "bmin": 0.0, "bmaj": 0.0, "width": 300.0}
    m = Model()
    m.addcomponent(g)
    mf = MatchedFilter(_window(spectra), m, continuum_order=order, channel_noise=None)
    mf.run()
    return float(mf.response[0, channel])


PEAK_A = _line(LINE_A, 15.0).max()  # continua at A's line peak, as in the simulations
CONTINUA = {
    "flat": np.full(N_CHAN, PEAK_A),
    "slope": PEAK_A * (1 + np.linspace(-0.5, 0.5, N_CHAN)),
}


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("continuum", ["flat", "slope"])
def test_a_polynomial_of_degree_one_or_more_absorbs_a_linear_continuum(order, continuum):
    b = _line(60, 6.0)
    assert _snr_at([b, CONTINUA[continuum]], 60, order) == pytest.approx(_snr_at([b], 60, order), abs=1e-6)


def test_a_constant_continuum_fit_leaves_a_slope_in_the_response():
    """Away from the window's middle, where the residual of a slope minus its mean is largest."""
    b = _line(100, 6.0)
    assert abs(_snr_at([b, CONTINUA["slope"]], 100, 0) - _snr_at([b], 100, 0)) > 1.0


def test_without_a_continuum_fit_a_continuum_reads_as_line():
    b = _line(60, 6.0)
    assert _snr_at([b, CONTINUA["flat"]], 60, None) - _snr_at([b], 60, None) > 15


@pytest.mark.parametrize("separation", [3, 6, 12])
def test_a_brighter_line_in_the_window_leaves_a_fainter_one_unbiased(separation):
    """A peaks above MASK_SNR, so the continuum is fitted without its channels for B's templates."""
    a, channel = _line(LINE_A, 15.0), LINE_A + separation * FWHM_CHANNELS
    b = _line(channel, 6.0)
    for order in (None, 1, 2):
        bias = _snr_at([a, b], channel, order) - _snr_at([b], channel, order)
        assert abs(bias) < (1e-3 if order is None else 0.15), order  # None: only the lines' far tails overlap


def _joint(spectra, channel, order):
    """The joint fit of template and polynomial over the whole window, in numpy."""
    y = 400 * np.sum(spectra, axis=0)  # 400 unit-weight rows at the phase centre
    W = np.full(N_CHAN, 400.0)
    g = np.exp(-0.5 * ((np.arange(N_CHAN) - channel) / (FWHM_CHANNELS / 2.355)) ** 2)
    P = np.linspace(-1.0, 1.0, N_CHAN)[:, None] ** np.arange(order + 1)
    h = g - P @ np.linalg.solve(P.T @ (W[:, None] * P), P.T @ (W * g))
    return h @ y / np.sqrt(h @ (W * h))


@pytest.mark.parametrize("snr", [4.0, 15.0])
def test_a_lone_line_gets_the_joint_fit(snr):
    """Below MASK_SNR nothing is masked; above, the line keeps its own channels: the joint fit either way."""
    line = _line(60, snr)
    for order in (1, 2):
        # rel 1e-5: the library's template is Gaussian in frequency, this one in channel index
        assert _snr_at([line], 60, order) == pytest.approx(_joint([line], 60, order), rel=1e-5)


def test_a_fit_leaves_out_the_other_lines_but_keeps_its_own_channels():
    from luv_finder.fit import _without
    from luv_finder.matchedfilter import MASK_FWHM

    data = _window([_line(LINE_A, 15.0)])
    (chunk,) = data.chunks
    nu = chunk.freq[100]
    other = (chunk.freq[LINE_A] / 1e9, 300.0)
    out = _without(chunk, nu, 300.0, [other])
    v_other = np.abs(chunk.freq / chunk.freq[LINE_A] - 1) * 299792.458
    dropped = np.all(out.w == 0, axis=1)
    assert np.array_equal(dropped, v_other <= MASK_FWHM * 300.0)
    assert np.all(out.X[dropped] == 0) and np.array_equal(out.w[~dropped], chunk.w[~dropped])
    # the fitted line's own channels stay even inside another's
    assert not np.all(_without(chunk, chunk.freq[LINE_A + 1], 300.0, [other]).w[LINE_A + 1] == 0)


def _search(spectra, dra, ddec):
    from luv_finder import MatchedFilter
    from luv_finder.matchedfilter import grid_model

    grid = {"dra": dra, "ddec": ddec, "bmin": 0.0, "bmaj": 0.0, "pa": 0.0, "width": [100.0, 300.0]}
    mf = MatchedFilter(_window(spectra), grid_model(grid), continuum_order=1, channel_noise=None)
    mf.run()
    return mf.result


def test_rows_without_a_bright_line_skip_the_masked_fit_unchanged():
    """A row of positions is refitted only if a line there reaches MASK_SNR: each position as if searched alone."""
    from luv_finder.matchedfilter import MASK_SNR

    spectra, offsets = [_line(LINE_A, 15.0), _line(LINE_A + 3 * FWHM_CHANNELS, 6.0)], [0.0, 40.0]
    result = _search(spectra, offsets, offsets)
    best = result.snr.max(axis=(2, 3))
    assert best[0, 0] >= MASK_SNR and best[0, 1] < MASK_SNR and best[1].max() < MASK_SNR
    for i, j in np.ndindex(best.shape):
        alone = _search(spectra, [offsets[i]], [offsets[j]])
        np.testing.assert_allclose(result.snr[i, j], alone.snr[0, 0], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(result.error[i, j], alone.error[0, 0], rtol=1e-9)
