import dataclasses
import os

import numpy as np
import pytest
from astropy.table import Table

from luv_finder import DataHandler, Gaussian, MatchedFilter, Model
from luv_finder.cli.find_lines import main as find_lines
from luv_finder.data import C, Chunk, Metadata
from luv_finder.fit import COLUMNS as FIT_COLUMNS
from luv_finder.matchedfilter import (
    SearchResult,
    build_grid,
    combine_pointings,
    default_cores,
    dirty_cube,
    dirty_maps,
    mosaic_dirty_maps,
    pb_corrected,
    search_pointings,
)
from luv_finder.model import FWHM_TO_SIGMA
from luv_finder.utils import primary_beam, primary_beam_radius

C_KMS = 299792.458


def _finder(data, weighting="natural", continuum_order=2, channel_correlation=None, **grid):
    g = Gaussian()
    g.grid = {
        "bmin": data.metadata.minresolution() / 10,
        "bmaj": data.metadata.minresolution() / 10,
        "width": 300.0,
        **grid,
    }
    m = Model()
    m.addcomponent(g)
    # synthetic rows share one baseline at random uv, so their jackknife keeps the sky: give the
    # (independent) channel correlation instead of measuring it
    return MatchedFilter(
        data, m, weighting=weighting, continuum_order=continuum_order, channel_correlation=channel_correlation
    )


def _chunk(nchan, nvis, seed=0, field=0, spw=0, offset=(0.0, 0.0), freq0=40e9):
    """Empty unit-weight window: random uv within 30 klambda, channels of 100 km/s."""
    r = np.random.default_rng(seed)
    freq = freq0 + (np.arange(nchan) - nchan // 2) * freq0 * 100 / C_KMS
    u, v = r.uniform(-3e4, 3e4, (2, nvis)) * C / freq0
    empty = np.zeros((nchan, nvis))
    return Chunk(
        field, spw, np.asarray(offset), freq, u, v, np.arange(nvis, dtype=float), np.zeros(nvis, dtype=int),
        empty.astype(complex), np.ones(nvis), empty.astype(bool),
    )  # fmt: skip


def _metadata(chunk):
    return Metadata(12.0, (0.0, 0.0), float(chunk.freq.mean()), float(np.hypot(chunk.u, chunk.v).max() * 41e9 / C))


def _synthetic(
    nchan, snr, nvis=400, seed=0, size=0.0, field=0, offset=(0.0, 0.0), pos=(0.0, 0.0), freq0=40e9, line=None
):
    """Noiseless unit-weight visibilities of one window holding a line of exactly the requested optimal S/N.

    ``size`` is the source's Gaussian sigma in arcsec (0: a point source), ``pos`` its position
    and ``offset`` the field's phase centre, both in the sky frame; ``line`` the line's channel
    (default: the middle one).
    """
    chunk = _chunk(nchan, nvis, seed, field=field, offset=offset, freq0=freq0)
    nu = chunk.freq[nchan // 2 if line is None else line]
    src = Gaussian(dra=pos[0], ddec=pos[1], nu_center=nu, width=300.0, bmin=size, bmaj=size)
    vis = src.profile(chunk)
    # optimal S/N of the line at its own position: sqrt(sum w Re[V shifted]^2)
    vis *= snr / np.sqrt(np.sum((vis * chunk.phase(*pos)).real ** 2))
    return DataHandler(chunks=[dataclasses.replace(chunk, X=vis)], metadata=_metadata(chunk))


def _peak(data, weighting="natural", **grid):
    mf = _finder(data, weighting, continuum_order=None, dra=0.0, ddec=0.0, **grid)
    mf.run()
    return mf.response[0].max()


def _direct(chunk, dra, ddec, bmin, bmaj, pa, width, template):
    """The statistic summed out channel by channel, without the scan or the position GEMM."""
    shape = Gaussian(bmin=bmin, bmaj=bmaj, pa=pa).envelope(chunk)
    t = shape if template else 1.0
    weight = np.sum(chunk.w * t**2, axis=1)
    amplitude = np.sum(chunk.w * t * shape, axis=1)
    sig = np.sum(t * (chunk.X * chunk.phase(dra, ddec)).real, axis=1)
    q = np.divide(amplitude, weight, out=np.zeros_like(weight), where=weight > 0)
    sigma = chunk.freq * width * FWHM_TO_SIGMA / C_KMS
    profile = np.exp(-0.5 * ((chunk.freq[None, :] - chunk.freq[:, None]) / sigma[:, None]) ** 2)
    return (profile @ (q * sig)) / np.sqrt(profile**2 @ (q**2 * weight))


@pytest.mark.parametrize("nchan", [16, 50, 120])
def test_response_equals_injected_snr_on_the_line_channel(nchan):
    """The response peaks on the line's channel at its S/N, independent of the number of channels."""
    mf = _finder(_synthetic(nchan, snr=10.0), continuum_order=None, dra=0.0, ddec=0.0)
    mf.run()
    response = mf.response[0]
    assert np.argmax(response) == nchan // 2
    assert response.max() == pytest.approx(10.0, abs=1e-3)
    assert response[nchan // 2 - 1] == pytest.approx(response[nchan // 2 + 1], rel=1e-2)


@pytest.mark.parametrize("template", [False, True])
def test_matches_the_direct_sum(template):
    """Scan recurrence and position GEMM reproduce the channel-by-channel sum, also over 960 channels."""
    r = np.random.default_rng(3)
    chunks = []
    for spw, nchan in ((0, 30), (1, 960)):
        chunk = _chunk(nchan, 200, seed=spw, spw=spw, offset=(4.0, -2.0), freq0=40e9 + spw * 2e9)
        flag = r.random(chunk.flag.shape) < 0.1
        flag[:3] = True
        w_row = r.uniform(0.5, 2.0, chunk.u.size)
        vis = r.normal(size=flag.shape) + 1j * r.normal(size=flag.shape)
        chunks.append(dataclasses.replace(chunk, X=np.where(flag, 0, w_row * vis), w_row=w_row, flag=flag))
    data = DataHandler(chunks=chunks, metadata=_metadata(chunks[0]))
    grid = {
        "dra": [-3.0, 0.0, 5.0],
        "ddec": [1.0, 7.0],
        "bmin": [0.5, 2.0],
        "bmaj": 3.0,
        "pa": 40.0,
        "width": [150.0, 500.0],
    }
    mf = _finder(data, "template" if template else "natural", continuum_order=None, **grid)
    mf.run()
    for row, p in enumerate(mf.grid_params):
        args = {k.split("_", 2)[-1]: v for k, v in p.items()}
        expected = np.concatenate([_direct(c, template=template, **args) for c in data.chunks])
        assert mf.response[row] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_line_at_the_window_edge_has_no_mirror():
    """A line two channels from the edge is found there at its S/N; the far edge stays empty."""
    response = _finder(_synthetic(50, snr=10.0, line=2), continuum_order=None, dra=0.0, ddec=0.0)
    response.run()
    r = response.response[0]
    assert np.argmax(r) == 2
    assert r.max() == pytest.approx(10.0, abs=1e-3)
    assert np.abs(r[-10:]).max() < 1e-6


def test_noise_has_unit_variance_at_every_channel():
    """White noise with unequal weights and flagged channels gives a unit-variance response, edges included."""
    r = np.random.default_rng(7)
    chunk = _chunk(40, 3000, seed=1)
    flag = np.zeros(chunk.flag.shape, dtype=bool)
    flag[[0, 17, 18]] = True
    w_row = r.uniform(0.2, 5.0, chunk.u.size)
    noise = (r.normal(size=flag.shape) + 1j * r.normal(size=flag.shape)) / np.sqrt(w_row)
    data = DataHandler(
        chunks=[dataclasses.replace(chunk, X=np.where(flag, 0, w_row * noise), w_row=w_row, flag=flag)],
        metadata=_metadata(chunk),
    )
    axis = np.linspace(-60, 60, 16)
    mf = _finder(data, continuum_order=None, dra=axis, ddec=axis)
    mf.run()
    per_channel = mf.response.std(axis=0)
    assert per_channel[1:].mean() == pytest.approx(1.0, abs=0.05)
    assert np.all(np.abs(per_channel[1:] - 1.0) < 0.3)  # 256 positions per channel


def test_rows_follow_grid_params():
    """The best row is the injected position and width, in grid_params order."""
    data = _synthetic(50, snr=20.0, pos=(4.0, -2.0))
    mf = _finder(data, dra=[-4.0, 0.0, 4.0], ddec=[-2.0, 0.0, 2.0], width=[150.0, 300.0, 600.0])
    mf.run(jackknife=True)
    best = mf.best_params
    assert (best["src_00_dra"], best["src_00_ddec"], best["src_00_width"]) == (4.0, -2.0, 300.0)
    assert len(mf.grid_params) == 27
    assert mf.response.shape == mf.response_jackknife.shape == (27, 50)


def test_grid_rejects_total_flux(data):
    with pytest.raises(ValueError, match="not searchable"):
        _finder(data, total_flux=np.array([1.0, 2.0]))


def test_grid_expansion(data):
    mf = _finder(data, dra=np.array([0.0, 1.0]), ddec=np.array([0.0, 2.0, 4.0]))
    assert len(mf.grid_params) == 6
    assert all(p["src_00_width"] == 300.0 for p in mf.grid_params)


def _fixture_search(data, truth, jackknife=False, **options):
    """Search the fixture's two source positions; returns the filter and the rows of the sources.

    The grid is the dra x ddec product of the positions, so it holds them on its diagonal.
    """
    positions = np.array([s["position_model"] for s in truth["sources"]])
    mf = _finder(data, dra=positions[:, 0], ddec=positions[:, 1], **options)
    mf.run(jackknife=jackknife)
    at = [(p["src_00_dra"], p["src_00_ddec"]) for p in mf.grid_params]
    return mf, [next(k for k, p in enumerate(at) if np.allclose(pos, p)) for pos in positions]


def test_recovers_injected_lines(data, truth):
    mf, rows = _fixture_search(data, truth, jackknife=True)
    freqs = mf.frequencies()
    for src, i in zip(truth["sources"], rows, strict=True):
        assert abs(freqs[np.argmax(mf.response[i])] - src["line"]["mean"]) < 0.01  # GHz, under a channel
    assert np.abs(mf.response_jackknife).max() < 0.5 * mf.response.max()


def test_continuum_fit_keeps_the_fixture_lines(data, truth):
    """Both lines (S/N 13 and 9) peak at their position and channel with the continuum fitted.

    Their S/N drops by a few per cent: the polynomial takes up the part of each line it can mimic.
    """
    fitted, rows = _fixture_search(data, truth)
    bare = _fixture_search(data, truth, continuum_order=None)[0]
    freqs = fitted.frequencies()
    for src, i in zip(truth["sources"], rows, strict=True):
        chan = np.argmax(fitted.response[i])
        assert abs(freqs[chan] - src["line"]["mean"]) < 0.01  # GHz, under a channel
        assert np.argmax(fitted.response[:, chan]) == i
        assert fitted.response[i, chan] == pytest.approx(bare.response[i].max(), rel=0.1)


def test_continuum_fit_keeps_noise_at_unit_variance_at_every_lag():
    """Fitted jointly, the continuum leaves noise at zero mean and unit variance at every lag, edges included.

    Four windows of unequal weights and flagged channels, an edge one among them; 48 x 48
    positions a beam or more apart give 9216 nearly independent draws per lag, so the standard
    deviation of each is known to about 0.01.
    """
    r = np.random.default_rng(11)
    chunks = []
    for spw in range(4):
        chunk = _chunk(40, 2000, seed=spw, spw=spw, freq0=40e9 + spw * 2e9)
        flag = np.zeros(chunk.flag.shape, dtype=bool)
        flag[[0, 17, 18]] = True
        w_row = r.uniform(0.2, 5.0, chunk.u.size)
        noise = (r.normal(size=flag.shape) + 1j * r.normal(size=flag.shape)) / np.sqrt(w_row)
        chunks.append(dataclasses.replace(chunk, X=np.where(flag, 0, w_row * noise), w_row=w_row, flag=flag))
    data = DataHandler(chunks=chunks, metadata=_metadata(chunks[0]))
    axis = np.arange(-235.0, 236.0, 10.0)
    mf = _finder(data, dra=axis, ddec=axis)
    mf.run()
    snr = mf.response.reshape(-1, 40)  # one row per position and window, one column per lag
    assert np.all(np.isfinite(snr))
    assert np.abs(snr.std(axis=0) - 1.0).max() < 0.03
    assert np.abs(snr.mean(axis=0)).max() < 0.05


@pytest.mark.parametrize("weighting", ["natural", "template"])
def test_continuum_far_off_centre_is_fitted_at_its_position(weighting):
    """A steep-spectrum continuum source whose phase winds across the window changes neither S/N nor flux.

    Unit weights, so the response is in S/N for 1 Jy noise per visibility; the 1 Jy continuum
    is then a S/N 30 signal at every lag. Two trial shapes, each fitted on its own. The fit is
    linear, so the continuum adds its own response, that of the part of a cubic no quadratic takes.
    """
    nchan, pos = 100, (60.0, -40.0)
    line = _synthetic(nchan, snr=10.0, pos=pos)
    (chunk,) = line.chunks
    phase = chunk.phase(*pos)
    assert np.ptp(np.unwrap(np.angle(phase), axis=0), axis=0).max() > 2.0  # rad, across the window
    continuum = (chunk.freq / chunk.freq.mean())[:, None] ** 3 * np.conj(phase)  # spectral index 3
    both = DataHandler(chunks=[dataclasses.replace(chunk, X=chunk.X + continuum)], metadata=line.metadata)

    def search(data, order):
        mf = _finder(data, weighting, continuum_order=order, dra=pos[0], ddec=pos[1], bmin=[0.0, 1.0], bmaj=1.0)
        mf.run()
        return mf.result

    alone, fitted, kept, bare = search(line, 2), search(both, 2), search(both, None), search(line, None)
    assert np.abs(fitted.response - alone.response).max() < 1e-3
    assert np.all(np.argmax(fitted.response, axis=1) == nchan // 2)
    assert fitted.flux[..., nchan // 2] == pytest.approx(bare.flux[..., nchan // 2], rel=1e-5)
    free = np.abs(np.arange(nchan) - nchan // 2) >= 10  # lags clear of the line
    assert kept.response[:, free].min() > 10.0


def test_lags_the_continuum_reproduces_are_uncovered():
    """In a window of three channels a quadratic mimics any line: no lag is covered, and none blows up."""
    data = _synthetic(3, snr=10.0)
    quadratic, linear = (_finder(data, continuum_order=order, dra=0.0, ddec=0.0) for order in (2, 1))
    quadratic.run()
    linear.run()
    assert np.all(quadratic.result.snr == 0.0) and np.all(np.isnan(quadratic.result.error))
    assert np.all(np.isfinite(linear.result.error)) and np.argmax(linear.response[0]) == 1


def test_flagged_channel_gives_a_finite_response():
    data = _synthetic(50, snr=10.0)
    (chunk,) = data.chunks
    flag = chunk.flag.copy()
    flag[10] = True  # away from the line, so the peak is unchanged
    data.chunks = [dataclasses.replace(chunk, X=np.where(flag, 0, chunk.X), flag=flag)]
    mf = _finder(data, continuum_order=None, dra=0.0, ddec=0.0)
    mf.run()
    assert np.all(np.isfinite(mf.response))
    assert mf.response.max() == pytest.approx(10.0, abs=1e-3)


def test_windows_are_searched_separately_in_frequency_order():
    """Each window gets its own templates; the spectra join in frequency order."""
    high = _synthetic(40, snr=10.0, freq0=41e9).chunks[0]
    low = dataclasses.replace(_synthetic(30, snr=0.0, freq0=39e9).chunks[0], spw=1)
    data = DataHandler(chunks=[high, low], metadata=_metadata(high))
    mf = _finder(data, continuum_order=None, dra=0.0, ddec=0.0)
    mf.run()
    assert np.all(np.diff(mf.frequencies()) > 0)
    assert mf.response.shape == (1, 70)
    assert np.argmax(mf.response[0]) == 30 + 20
    assert mf.response.max() == pytest.approx(10.0, abs=1e-3)


def test_pointings_share_one_sky_grid():
    """A source seen by two pointings peaks at the same sky position, with its S/N, in both."""
    pos = (5.0, -3.0)
    for field, offset in ((0, (0.0, 0.0)), (1, (20.0, 10.0))):
        data = _synthetic(50, snr=10.0, field=field, offset=offset, pos=pos)
        mf = _finder(data, continuum_order=None, dra=[0.0, 5.0, 10.0], ddec=[-3.0, 0.0])
        mf.run()
        best = mf.best_params
        assert (best["src_00_dra"], best["src_00_ddec"]) == pos
        assert mf.response[mf.best_index].max() == pytest.approx(10.0, abs=1e-3)


def test_one_field_at_a_time():
    a = _synthetic(20, snr=1.0, field=0).chunks[0]
    b = _synthetic(20, snr=1.0, field=1).chunks[0]
    with pytest.raises(ValueError, match="one field at a time"):
        _finder(DataHandler(chunks=[a, b], metadata=_metadata(a)))


def test_template_weighting_equals_natural_for_a_point_source():
    data = _synthetic(50, snr=10.0)
    assert _peak(data, "template", bmin=0.0, bmaj=0.0) == pytest.approx(_peak(data, "natural", bmin=0.0, bmaj=0.0))


def test_template_weighting_recovers_a_resolved_source():
    """Weighting by A(u, v) attains the optimal S/N; the plain average loses the predicted fraction."""
    size = 3.0  # arcsec sigma, resolved by the 30 klambda baselines
    data = _synthetic(50, snr=10.0, size=size)
    template = _peak(data, "template", bmin=size, bmaj=size)
    natural = _peak(data, "natural", bmin=size, bmaj=size)

    (chunk,) = data.chunks
    a = Gaussian(bmin=size, bmaj=size).envelope(chunk)
    kept = np.sum(chunk.w * a) / np.sqrt(np.sum(chunk.w) * np.sum(chunk.w * a**2))
    assert template == pytest.approx(10.0, abs=0.01)
    assert natural == pytest.approx(10.0 * kept, abs=0.05)
    assert kept < 0.8


def test_dirty_cube_of_a_point_source_is_its_flux():
    """Natural weighting: the dirty map of a point source peaks at its flux, in every channel."""
    chunk = _chunk(20, 500, offset=(3.0, 1.0))
    data = DataHandler(
        chunks=[dataclasses.replace(chunk, X=2.0 * np.conj(chunk.phase(5.0, -3.0)))], metadata=_metadata(chunk)
    )
    c = dirty_cube(data, [-10.0, 0.0, 5.0], [-3.0, 0.0, 8.0])
    assert c.cube.shape == (20, 3, 3)
    assert np.allclose(c.cube[:, 2, 0], 2.0)
    assert np.allclose(c.weight, 500.0)
    assert (c.dra, c.ddec, c.window_sizes) == ([-10.0, 0.0, 5.0], [-3.0, 0.0, 8.0], [20])
    assert np.array_equal(c.offset, (3.0, 1.0)) and np.array_equal(c.freqs, chunk.freq)


def test_natural_search_hands_back_its_dirty_cube(monkeypatch):
    """run(cube=True) gives dirty_cube's cube from the search's own collapse; template weighting has none."""
    monkeypatch.setattr("luv_finder.matchedfilter.BLOCK_BYTES", 2 * 16 * 3 * 300)  # two dra rows per block
    r = np.random.default_rng(5)
    chunks = []
    for spw, nchan in ((0, 30), (1, 20)):
        chunk = _chunk(nchan, 300, seed=spw, spw=spw, offset=(4.0, -2.0), freq0=40e9 + spw * 2e9)
        flag = r.random(chunk.flag.shape) < 0.1
        flag[3] = True
        vis = r.normal(size=flag.shape) + 1j * r.normal(size=flag.shape)
        chunks.append(dataclasses.replace(chunk, X=np.where(flag, 0, vis), flag=flag))
    data = DataHandler(chunks=chunks, metadata=_metadata(chunks[0]))
    dra, ddec = [-3.0, 0.0, 5.0], [1.0, 7.0]
    mf = _finder(data, dra=dra, ddec=ddec, width=[150.0, 300.0])
    mf.run(jackknife=True, cube=True)
    expected = dirty_cube(data, dra, ddec)
    assert np.allclose(mf.cube.cube, expected.cube, rtol=0, atol=1e-12)
    assert np.array_equal(mf.cube.weight, expected.weight)
    assert (mf.cube.dra, mf.cube.ddec, mf.cube.window_sizes) == (dra, ddec, [30, 20])
    with pytest.raises(ValueError, match="natural weighting"):
        _finder(data, "template", dra=dra, ddec=ddec).run(cube=True)


def test_dirty_maps_separate_continuum_and_line():
    """The continuum map shows the continuum source; the moment-8 the line, not the continuum."""
    chunk = _chunk(40, 800)
    continuum = 0.5 * np.conj(chunk.phase(20.0, -15.0))
    line = Gaussian(dra=-10.0, ddec=8.0, nu_center=chunk.freq[20], width=300.0, total_flux=4.0).profile(chunk)
    data = DataHandler(chunks=[dataclasses.replace(chunk, X=continuum + line)], metadata=_metadata(chunk))
    axis = np.arange(-30.0, 31.0, 5.0)
    moment8, cont, sigma = dirty_maps(data, axis, axis)
    assert sigma == pytest.approx(1 / np.sqrt(40 * 800))
    assert np.unravel_index(np.argmax(cont), cont.shape) == (10, 3)  # (+20, -15)
    assert np.unravel_index(np.argmax(moment8), moment8.shape) == (4, 8)  # (-10, +8)
    assert moment8[10, 3] < 0.2 * moment8.max()


FREQ = _chunk(50, 1).freq


def _line(dra=0.0, ddec=0.0):
    """A point source with a 2 Jy km/s line of 300 km/s on channel 25 of a 50-channel window."""
    return Gaussian(dra=dra, ddec=ddec, nu_center=FREQ[25], width=300.0, total_flux=2.0)


def _seen_by(offset, src, field=0):
    """Noiseless unit-weight pointing at ``offset`` that sees ``src`` through its primary beam."""
    chunk = _chunk(50, 400, seed=field, field=field, offset=offset)
    pb = primary_beam(np.hypot(src.dra - offset[0], src.ddec - offset[1]), chunk.freq, 12.0)
    X = pb[:, None] * src.profile(chunk)
    return DataHandler(chunks=[dataclasses.replace(chunk, X=X)], metadata=_metadata(chunk))


def _point_search(data, dra, ddec):
    mf = _finder(data, continuum_order=None, dra=dra, ddec=ddec, bmin=0.0, bmaj=0.0)
    mf.run(jackknife=True)
    return mf.result


def test_flux_is_the_injected_peak_flux_density():
    """S/N times error is the line's peak flux density, its integrated flux the injected total flux."""
    src = _line()
    r = _point_search(_seen_by((0.0, 0.0), src), 0.0, 0.0)
    assert r.snr.shape == r.flux.shape == r.error.shape == (1, 1, 1, 50)
    assert np.all(r.coverage == 1.0)
    assert r.flux[0, 0, 0, 25] == pytest.approx(src.spectrum(FREQ[25]), rel=1e-6)
    profile = np.exp(-0.5 * ((FREQ - FREQ[25]) / (FREQ[25] * 300.0 * FWHM_TO_SIGMA / C_KMS)) ** 2)
    assert r.error[0, 0, 0, 25] == pytest.approx(1 / np.sqrt(400 * np.sum(profile**2)))
    flux, error = r.integrated_flux()
    assert flux[0, 0, 0, 25] == pytest.approx(2.0, rel=1e-9)
    assert error[0, 0, 0, 25] == pytest.approx(r.error[0, 0, 0, 25] * flux[0, 0, 0, 25] / r.flux[0, 0, 0, 25])


def test_pb_correction_divides_the_flux_and_masks_the_beam_edge():
    src = _line(dra=40.0)
    data = _seen_by((0.0, 0.0), src)
    raw = _point_search(data, np.arange(-160.0, 161.0, 40.0), [0.0, 40.0])
    out = pb_corrected(raw, data, pb_limit=0.2)
    distance = np.hypot(*np.meshgrid(raw.axes["dra"], raw.axes["ddec"], indexing="ij"))
    pb = primary_beam(distance[..., None], raw.freqs, 12.0)
    keep = pb >= 0.2
    assert 0 < keep.sum() < keep.size
    for corrected, uncorrected in ((out, raw), (out.jackknife, raw.jackknife)):
        snr, flux, error = (getattr(corrected, k)[:, :, 0] for k in ("snr", "flux", "error"))
        assert np.array_equal(snr[keep], uncorrected.snr[:, :, 0][keep])
        assert np.allclose(flux[keep], uncorrected.flux[:, :, 0][keep] / pb[keep], rtol=1e-12, atol=0)
        assert np.allclose(error[keep], uncorrected.error[:, :, 0][keep] / pb[keep], rtol=1e-12, atol=0)
        assert np.all(np.isnan(snr[~keep]) & np.isnan(flux[~keep]) & np.isnan(error[~keep]))
        assert np.array_equal(corrected.coverage, np.where(keep, pb, 0.0))
    assert (out.best_params["src_00_dra"], out.best_params["src_00_ddec"]) == (40.0, 0.0)
    assert out.flux[5, 0, 0, 25] == pytest.approx(src.spectrum(FREQ[25]), rel=1e-6)


def test_combined_pointings_recover_the_flux_and_add_snr_in_quadrature():
    """Inverse-variance weights restore the flux; a pointing beyond its pb_limit changes nothing."""
    src = _line(dra=20.0)

    def pointing(field, offset, dra, ddec):
        data = _seen_by(offset, src, field)
        return pb_corrected(_point_search(data, dra, ddec), data)

    a = pointing(0, (0.0, 0.0), [-20.0, 0.0, 20.0, 40.0], [-20.0, 0.0, 20.0])
    b = pointing(1, (60.0, 0.0), [20.0, 40.0, 60.0, 80.0], [0.0, 20.0])
    far = pointing(2, (220.0, 0.0), np.arange(20.0, 221.0, 20.0), [0.0])
    pair, mosaic = combine_pointings([a, b]), combine_pointings([a, b, far])
    assert mosaic.axes["dra"] == np.arange(-20.0, 221.0, 20.0).tolist()
    assert mosaic.axes["ddec"] == [-20.0, 0.0, 20.0]

    def at(r, dra=20.0, ddec=0.0):
        return r.axes["dra"].index(dra), r.axes["ddec"].index(ddec)

    single = [r.snr[(*at(r), 0, 25)] for r in (a, b)]
    for r in (pair, mosaic):
        i, j = at(r)
        assert r.flux[i, j, 0, 25] == pytest.approx(src.spectrum(FREQ[25]), rel=1e-6)
        assert r.snr[i, j, 0, 25] == pytest.approx(np.hypot(*single), rel=1e-12)
        assert r.coverage[i, j, 25] == pytest.approx(np.hypot(a.coverage[(*at(a), 25)], b.coverage[(*at(b), 25)]))
    for k in ("snr", "flux", "error"):
        assert np.array_equal(getattr(pair, k)[at(pair)], getattr(mosaic, k)[at(mosaic)])
    nowhere = at(mosaic, 220.0, -20.0)  # on no pointing's grid
    assert np.all(np.isnan(mosaic.snr[nowhere])) and np.all(mosaic.coverage[nowhere] == 0)

    # the jackknives are combined with their own inverse-variance weights
    snr, error = (np.array([getattr(r.jackknife, k)[at(r)][0] for r in (a, b)]) for k in ("snr", "error"))
    expected = np.sum(snr / error, axis=0) / np.sqrt(np.sum(error**-2.0, axis=0))
    assert mosaic.jackknife.snr[at(mosaic)][0] == pytest.approx(expected, rel=1e-12)
    assert combine_pointings([dataclasses.replace(a, jackknife=None), b]).jackknife is None


def test_combine_pointings_rejects_other_templates_channels_and_lattices():
    def result(dra, width=(300.0,), freqs=FREQ):
        shape = (len(dra), 1, len(width), len(freqs))
        axes = {"dra": list(dra), "ddec": [0.0], "bmin": [0.0], "bmaj": [0.0], "pa": [0.0], "width": list(width)}
        ones = np.ones(shape)
        return SearchResult(axes, freqs, ones, ones, ones, np.ones((*shape[:2], len(freqs))))

    ref = result([0.0, 5.0])
    assert combine_pointings([ref, result([15.0, 20.0])]).axes["dra"] == [0.0, 5.0, 10.0, 15.0, 20.0]
    with pytest.raises(ValueError, match="width templates"):
        combine_pointings([ref, result([0.0, 5.0], width=(300.0, 400.0))])
    with pytest.raises(ValueError, match="frequencies"):
        combine_pointings([ref, result([0.0, 5.0], freqs=FREQ * 1.0001)])
    with pytest.raises(ValueError, match="lattice"):
        combine_pointings([ref, result([2.5, 7.5])])  # half a step off
    with pytest.raises(ValueError, match="lattice"):
        combine_pointings([ref, result([0.0, 10.0, 20.0])])  # a coarser step


def test_mosaic_dirty_maps_restore_a_continuum_source():
    """Two pointings see a continuum source through their beams; the linear mosaic restores its flux."""
    pos, flux, ivar, cubes = (20.0, 0.0), 2.0, 0.0, []
    pointings = (
        ((0.0, 0.0), 500, [-20.0, 0.0, 20.0], [0.0, 20.0]),
        ((60.0, 0.0), 300, [20.0, 40.0, 60.0, 80.0, 240.0], [-20.0, 0.0]),
    )
    for field, (offset, nvis, dra, ddec) in enumerate(pointings):
        chunk = _chunk(20, nvis, seed=field, field=field, offset=offset)
        pb = primary_beam(np.hypot(pos[0] - offset[0], pos[1] - offset[1]), chunk.freq, 12.0)
        X = flux * pb[:, None] * np.conj(chunk.phase(*pos))
        data = DataHandler(chunks=[dataclasses.replace(chunk, X=X)], metadata=_metadata(chunk))
        cubes.append(dirty_cube(data, dra, ddec))
        ivar += np.sum(pb**2 * nvis)
    dra, ddec, moment8, continuum, sigma = mosaic_dirty_maps(cubes, pb_limit=0.2)
    assert dra.tolist() == np.arange(-20.0, 241.0, 20.0).tolist()
    assert ddec.tolist() == [-20.0, 0.0, 20.0]
    assert continuum[2, 1] == pytest.approx(flux, rel=1e-9)
    assert sigma[2, 1] == pytest.approx(1 / np.sqrt(ivar), rel=1e-9)
    assert abs(moment8[2, 1]) < 1e-6  # no line, so the continuum is all subtracted
    # (-20, -20) is on neither grid; (240, 0) only on the second, beyond its pb_limit
    for i, j in ((0, 0), (13, 1)):
        assert np.isnan(moment8[i, j]) and np.isnan(continuum[i, j]) and np.isnan(sigma[i, j])
    assert np.all(np.isfinite(continuum[:5, 1]))


def test_build_grid_with_pb_limit_reaches_the_beam_radius_on_the_same_lattice():
    data = _synthetic(20, snr=1.0, offset=(10.0, -5.0))
    default, wide = build_grid(data, None), build_grid(data, {"pb_limit": 0.2})
    half = primary_beam_radius(0.2, data.freqs.min(), data.metadata.dish_diameter)
    step = data.metadata.minresolution() / 2
    for axis, centre in zip(("dra", "ddec"), data.chunks[0].offset, strict=True):
        offsets = np.abs(wide[axis] - centre)
        assert offsets.max() <= half < offsets.max() + step
        assert np.allclose(wide[axis] / step, np.round(wide[axis] / step))
        assert np.all(np.isin(default[axis], wide[axis]))


#: A lattice of 0.25 arcsec steps through the fixture's two sources, (4.25, 23.5) and (12.0, -4.25).
SOURCES_GRID = {"dra": [4.25, 4.5, 12.0, 12.25], "ddec": [-4.25, -4.0, 23.5, 23.75], "width": 300.0}


def _assert_same(a, b):
    """Equal searches, NaN included, down to the jackknife."""
    assert a.axes == b.axes
    for key in ("freqs", "snr", "flux", "error", "coverage"):
        assert np.allclose(getattr(a, key), getattr(b, key), rtol=1e-12, atol=0, equal_nan=True), key
    assert (a.jackknife is None) == (b.jackknife is None)
    if a.jackknife is not None:
        _assert_same(a.jackknife, b.jackknife)


def _search_by_hand(path, field, continuum_order=2, cube=False):
    """One field searched step by step as ``search_pointings`` does: ``(data, filter)`` after ``run``."""
    data = DataHandler.from_npz(path, [field])
    mf = _finder(data, continuum_order=continuum_order, **build_grid(data, {**SOURCES_GRID, "pb_limit": 0.2}))
    mf.run(jackknife=True, cube=cube)
    return data, mf


def test_search_pointings_combines_the_pb_corrected_fields(two_field_npz):
    by_hand = [_search_by_hand(two_field_npz, field, cube=True) for field in (0, 1)]
    result, cubes = search_pointings(two_field_npz, SOURCES_GRID, jackknife=True, cube=True)
    expected = combine_pointings([pb_corrected(mf.result, data, 0.2) for data, mf in by_hand])
    _assert_same(result, expected)
    assert result.jackknife is not None
    assert np.isnan(result.snr).any() and np.isfinite(result.snr).any()
    assert len(cubes) == 2
    for cube, (_, mf) in zip(cubes, by_hand, strict=True):
        assert np.array_equal(cube.cube, mf.cube.cube)
        assert np.array_equal(cube.offset, mf.cube.offset)
    assert cubes[1].offset[0] - cubes[0].offset[0] == 10.0


@pytest.mark.parametrize("continuum_order", [2, None])
def test_search_pointings_of_one_field_is_its_pb_corrected_search(fixture_npz, continuum_order):
    result, cubes = search_pointings(fixture_npz, SOURCES_GRID, continuum_order=continuum_order)
    data, mf = _search_by_hand(fixture_npz, 0, continuum_order)
    _assert_same(result, pb_corrected(mf.result, data, 0.2))
    assert cubes == []


def test_search_pointings_without_jackknife_has_none(fixture_npz):
    assert search_pointings(fixture_npz, SOURCES_GRID, jackknife=False)[0].jackknife is None


#: A blind lattice around both sources, as the catalogue needs (its noise correlation is measured on a lattice).
LATTICE = {"dra": {"start": 0.0, "stop": 16.0, "step": 0.5}, "ddec": {"start": -8.0, "stop": 28.0, "step": 0.5}}


def _find_lines(tmp_path, npz, *options, grid_cfg=SOURCES_GRID):
    """Run ``luv-find`` on ``npz`` with ``grid_cfg``, writing its figure into ``tmp_path``."""
    grid = tmp_path / "grid.yaml"
    grid.write_text("\n".join(f"{k}: {v}" for k, v in grid_cfg.items()))
    find_lines(["--ms", str(npz), "--grid", str(grid), "--plots-dir", str(tmp_path), *options])


@pytest.mark.parametrize("order", [[], ["--continuum-order", "none"], ["--continuum-order", "-1"]])
def test_find_lines_combines_a_mosaic_without_field(tmp_path, capsys, two_field_npz, order):
    out = tmp_path / "response.npz"
    _find_lines(tmp_path, two_field_npz, "--jackknife", "--out", str(out), *order)
    text = capsys.readouterr().out
    assert "best grid point" in text and "PB-corrected peak flux density" in text
    assert "jackknife max" in text
    assert (tmp_path / "filter_response.png").exists()
    with np.load(out, allow_pickle=True) as saved:
        assert saved["response"].shape[1] == len(saved["freqs"]) == 50
        assert saved["response"].shape[0] == len(saved["grid_params"])


@pytest.mark.parametrize("npz", ["fixture_npz", "two_field_npz"])
def test_find_lines_writes_a_catalogue(tmp_path, capsys, request, npz):
    out = tmp_path / "lines.ecsv"
    _find_lines(tmp_path, request.getfixturevalue(npz), "--jackknife", "--catalogue", str(out), grid_cfg=LATTICE)
    cat = Table.read(out, format="ascii.ecsv")
    assert len(cat) and {"ra", "dec", "likelihood", "fidelity", "detected"} <= set(cat.colnames)
    assert "candidates detected" in capsys.readouterr().out
    assert (tmp_path / "reliability.png").exists()


def test_find_lines_catalogue_needs_the_jackknife(tmp_path, fixture_npz):
    with pytest.raises(SystemExit):
        _find_lines(tmp_path, fixture_npz, "--catalogue", str(tmp_path / "lines.ecsv"))


def test_find_lines_fits_the_detected_lines(tmp_path, capsys, fixture_npz):
    out = tmp_path / "lines.ecsv"
    _find_lines(tmp_path, fixture_npz, "--jackknife", "--catalogue", str(out), "--fit", grid_cfg=LATTICE)
    cat = Table.read(out, format="ascii.ecsv")
    assert set(FIT_COLUMNS) <= set(cat.colnames)
    assert np.all(np.isfinite(cat["fit_line_flux"][cat["detected"]]))
    assert np.all(np.isnan(cat["fit_line_flux"][~cat["detected"]]))
    assert "fit_line_flux" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        _find_lines(tmp_path, fixture_npz, "--jackknife", "--fit")


def test_find_lines_searches_one_field_without_pb_correction(tmp_path, capsys, two_field_npz):
    _find_lines(tmp_path, two_field_npz, "--field", "1")
    text = capsys.readouterr().out
    assert "peak flux density" in text and "PB-corrected" not in text


@pytest.mark.parametrize(("total", "used"), [(48, 12), (8, 2), (4, 1), (2, 1), (1, 1)])
def test_default_cores_leave_the_machine_mostly_free(monkeypatch, total, used):
    monkeypatch.setattr(os, "cpu_count", lambda: total)
    assert default_cores() == used


def test_unknown_weighting_is_rejected(data):
    with pytest.raises(ValueError, match="weighting"):
        _finder(data, weighting="uniform")


def _hanning_noise(nchan=64, nvis=4000, seed=1):
    """Noise only, Hanning-smoothed along the channels, at unit weight."""
    chunk = _chunk(nchan, nvis)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((nchan + 2, nvis)) + 1j * rng.standard_normal((nchan + 2, nvis))
    noise = (0.25 * noise[:-2] + 0.5 * noise[1:-1] + 0.25 * noise[2:]) / np.sqrt(0.375)
    return DataHandler(chunks=[dataclasses.replace(chunk, X=noise)], metadata=_metadata(chunk))


@pytest.mark.parametrize("continuum_order", [None, 2])
def test_correlated_channels_keep_unit_variance(continuum_order):
    """Hanning noise inflates the S/N of broad templates unless the measured correlation is used."""
    data, axis = _hanning_noise(), np.linspace(-20.0, 20.0, 9)
    spread = {}
    for corr in (None, "measure"):
        mf = _finder(data, continuum_order=continuum_order, channel_correlation=corr, dra=axis, ddec=axis,
                     width=[200.0, 400.0])  # fmt: skip
        mf.run()
        spread[corr] = np.nanstd(mf.result.snr, axis=(0, 1, 3))
    assert np.all(spread[None] > 1.3)
    assert spread["measure"] == pytest.approx(1.0, abs=0.04)
    assert mf.correlation[0] == pytest.approx((1.0, 2 / 3, 1 / 6), abs=0.02)


def test_flux_does_not_depend_on_the_channel_correlation():
    """The correlation changes the S/N and the error, not the best-fit flux density."""
    data = _synthetic(50, snr=10.0)
    hanning = {0: np.array([1.0, 2 / 3, 1 / 6])}
    plain, smooth = (_finder(data, channel_correlation=c, dra=0.0, ddec=0.0) for c in (None, hanning))
    plain.run()
    smooth.run()
    np.testing.assert_allclose(smooth.result.flux, plain.result.flux, rtol=1e-9)
    assert np.all(smooth.result.error > plain.result.error)


def test_unknown_channel_correlation_is_rejected(data):
    with pytest.raises(ValueError, match="channel_correlation"):
        MatchedFilter(data, Model(), channel_correlation="hanning")
