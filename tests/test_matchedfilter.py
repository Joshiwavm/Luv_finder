import dataclasses
import os

import numpy as np
import pytest

from luv_finder import DataHandler, Gaussian, MatchedFilter, Model
from luv_finder.data import C, Chunk, Metadata
from luv_finder.matchedfilter import default_cores, dirty_cube, dirty_maps
from luv_finder.model import FWHM_TO_SIGMA

C_KMS = 299792.458


def _finder(data, weighting="natural", **grid):
    g = Gaussian()
    g.grid = {
        "bmin": data.metadata.minresolution() / 10,
        "bmaj": data.metadata.minresolution() / 10,
        "width": 300.0,
        **grid,
    }
    m = Model()
    m.addcomponent(g)
    return MatchedFilter(data, m, weighting=weighting)


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
    mf = _finder(data, weighting, dra=0.0, ddec=0.0, **grid)
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
    mf = _finder(_synthetic(nchan, snr=10.0), dra=0.0, ddec=0.0)
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
    mf = _finder(data, "template" if template else "natural", **grid)
    mf.run()
    for row, p in enumerate(mf.grid_params):
        args = {k.split("_", 2)[-1]: v for k, v in p.items()}
        expected = np.concatenate([_direct(c, template=template, **args) for c in data.chunks])
        assert mf.response[row] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_line_at_the_window_edge_has_no_mirror():
    """A line two channels from the edge is found there at its S/N; the far edge stays empty."""
    response = _finder(_synthetic(50, snr=10.0, line=2), dra=0.0, ddec=0.0)
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
    mf = _finder(data, dra=axis, ddec=axis)
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


def test_recovers_injected_lines(data, truth):
    positions = np.array([s["position_model"] for s in truth["sources"]])
    mf = _finder(data, dra=positions[:, 0], ddec=positions[:, 1])
    mf.run(jackknife=True)
    freqs = mf.frequencies()

    # the grid is the dra x ddec product; the two on-source points must peak on their line
    for src in truth["sources"]:
        i = next(
            k
            for k, p in enumerate(mf.grid_params)
            if np.allclose(src["position_model"], (p["src_00_dra"], p["src_00_ddec"]))
        )
        assert abs(freqs[np.argmax(mf.response[i])] - src["line"]["mean"]) < 0.01  # GHz, under a channel

    assert np.abs(mf.response_jackknife).max() < 0.5 * mf.response.max()


def test_flagged_channel_gives_a_finite_response():
    data = _synthetic(50, snr=10.0)
    (chunk,) = data.chunks
    flag = chunk.flag.copy()
    flag[10] = True  # away from the line, so the peak is unchanged
    data.chunks = [dataclasses.replace(chunk, X=np.where(flag, 0, chunk.X), flag=flag)]
    mf = _finder(data, dra=0.0, ddec=0.0)
    mf.run()
    assert np.all(np.isfinite(mf.response))
    assert mf.response.max() == pytest.approx(10.0, abs=1e-3)


def test_windows_are_searched_separately_in_frequency_order():
    """Each window gets its own templates; the spectra join in frequency order."""
    high = _synthetic(40, snr=10.0, freq0=41e9).chunks[0]
    low = dataclasses.replace(_synthetic(30, snr=0.0, freq0=39e9).chunks[0], spw=1)
    data = DataHandler(chunks=[high, low], metadata=_metadata(high))
    mf = _finder(data, dra=0.0, ddec=0.0)
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
        mf = _finder(data, dra=[0.0, 5.0, 10.0], ddec=[-3.0, 0.0])
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
    cube, weight = dirty_cube(data, [-10.0, 0.0, 5.0], [-3.0, 0.0, 8.0])
    assert cube.shape == (20, 3, 3)
    assert np.allclose(cube[:, 2, 0], 2.0)
    assert np.allclose(weight, 500.0)


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


@pytest.mark.parametrize(("total", "used"), [(48, 12), (8, 2), (4, 1), (2, 1), (1, 1)])
def test_default_cores_leave_the_machine_mostly_free(monkeypatch, total, used):
    monkeypatch.setattr(os, "cpu_count", lambda: total)
    assert default_cores() == used


def test_unknown_weighting_is_rejected(data):
    with pytest.raises(ValueError, match="weighting"):
        _finder(data, weighting="uniform")
