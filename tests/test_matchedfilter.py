import dataclasses

import numpy as np
import pytest

from luv_finder import DataHandler, Gaussian, MatchedFilter, Model
from luv_finder.data import C, Chunk, Metadata
from luv_finder.matchedfilter import nu_center_func

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


def _synthetic(nchan, snr, nvis=400, seed=0, size=0.0, field=0, offset=(0.0, 0.0), pos=(0.0, 0.0), freq0=40e9):
    """Noiseless unit-weight visibilities of one window holding a line of exactly the requested optimal S/N.

    ``size`` is the source's Gaussian sigma in arcsec (0: a point source), ``pos`` its position
    and ``offset`` the field's phase centre, both in the sky frame. The line sits where the
    filter places its kernel, so the kernel normalisation holds exactly even though the weights
    of a resolved source change with frequency.
    """
    r = np.random.default_rng(seed)
    freq = freq0 + (np.arange(nchan) - nchan / 2) * freq0 * 100 / C_KMS
    u, v = r.uniform(-3e4, 3e4, (2, nvis)) * C / freq0
    empty = np.zeros((nchan, nvis))
    chunk = Chunk(
        field, 0, np.asarray(offset), freq, u, v, np.arange(nvis, dtype=float), np.zeros(nvis, dtype=int),
        empty.astype(complex), np.ones(nvis), empty.astype(bool),
    )  # fmt: skip
    nu_center = nu_center_func(300.0, freq[0])
    src = Gaussian(dra=pos[0], ddec=pos[1], nu_center=nu_center, width=300.0, bmin=size, bmaj=size)
    vis = src.profile(chunk)
    # optimal S/N of the line at its own position: sqrt(sum w Re[V shifted]^2)
    vis *= snr / np.sqrt(np.sum((vis * chunk.phase(*pos)).real ** 2))
    meta = Metadata(12.0, (0.0, 0.0), float(freq.mean()), float(np.hypot(u, v).max() * freq.max() / C))
    return DataHandler(chunks=[dataclasses.replace(chunk, X=vis)], metadata=meta)


def _point(**params) -> dict:
    """A full grid point: a point source of 300 km/s at the reference, overridden by ``params``."""
    return {
        "src_00_dra": 0.0,
        "src_00_ddec": 0.0,
        "src_00_bmin": 0.0,
        "src_00_bmaj": 0.0,
        "src_00_width": 300.0,
        "src_00_total_flux": 1.0,
        **params,
    }


def _peak(data, weighting="natural", **params):
    return _finder(data, weighting).response_at(_point(**params)).max()


@pytest.mark.parametrize("nchan", [16, 50, 120])
def test_response_equals_injected_snr(nchan):
    """The response is the line's S/N, independent of how many channels there are."""
    assert _peak(_synthetic(nchan, snr=10.0)) == pytest.approx(10.0, abs=0.05)


def test_response_is_independent_of_template_amplitude():
    """total_flux cancels in the kernel normalisation, so it is not a search axis."""
    data = _synthetic(50, snr=10.0)
    peaks = [_peak(data, src_00_total_flux=f) for f in (0.25, 1.0, 4.0, 100.0)]
    assert np.ptp(peaks) < 1e-9


def test_grid_rejects_total_flux(data):
    mf = _finder(data, total_flux=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="not searchable"):
        mf._expand_grid()


def test_grid_expansion(data):
    mf = _finder(data, dra=np.array([0.0, 1.0]), ddec=np.array([0.0, 2.0, 4.0]))
    pts = mf._expand_grid()
    assert len(pts) == 6
    assert all(p["src_00_width"] == 300.0 for p in pts)


def test_recovers_injected_lines(data, truth):
    positions = np.array([s["position_model"] for s in truth["sources"]])
    mf = _finder(data, dra=positions[:, 0], ddec=positions[:, 1])
    mf.run(pool=1, jackknife=True)
    freqs = mf.frequencies()

    # the grid is the dra x ddec product; the two on-source points must peak on their line
    for src in truth["sources"]:
        i = next(
            k
            for k, p in enumerate(mf.grid_params)
            if np.allclose(src["position_model"], (p["src_00_dra"], p["src_00_ddec"]))
        )
        assert abs(freqs[np.argmax(mf.response[i])] - src["line"]["mean"]) < 0.05  # GHz

    assert np.abs(mf.response_jackknife).max() < 0.5 * mf.response.max()


def test_flagged_channel_gives_a_finite_response():
    data = _synthetic(50, snr=10.0)
    (chunk,) = data.chunks
    flag = chunk.flag.copy()
    flag[30] = True  # away from the line, so the peak is unchanged
    data.chunks = [dataclasses.replace(chunk, X=np.where(flag, 0, chunk.X), flag=flag)]
    response = _finder(data).response_at(_point())
    assert np.all(np.isfinite(response))
    assert response.max() == pytest.approx(10.0, abs=0.05)


def test_windows_are_searched_separately_in_frequency_order():
    """Each window gets its own kernel; the spectra join in frequency order."""
    high = _synthetic(40, snr=10.0, freq0=41e9).chunks[0]
    low = dataclasses.replace(_synthetic(30, snr=0.0, freq0=39e9).chunks[0], spw=1)
    data = DataHandler(chunks=[high, low], metadata=_synthetic(40, snr=10.0, freq0=41e9).metadata)
    mf = _finder(data)
    response = mf.response_at(_point())
    assert np.all(np.diff(mf.frequencies()) > 0)
    assert response.shape == (70,)
    assert np.argmax(response) >= 30
    assert response.max() == pytest.approx(10.0, abs=0.05)


def test_pointings_share_one_sky_grid():
    """A source seen by two pointings peaks at the same sky position, with its S/N, in both."""
    pos = (5.0, -3.0)
    grid = {"dra": np.array([0.0, 5.0, 10.0]), "ddec": np.array([-3.0, 0.0])}
    for field, offset in ((0, (0.0, 0.0)), (1, (20.0, 10.0))):
        data = _synthetic(50, snr=10.0, field=field, offset=offset, pos=pos)
        mf = _finder(data, **grid)
        mf.run(pool=1)
        best = mf.best_params
        assert (best["src_00_dra"], best["src_00_ddec"]) == pos
        assert mf.response[mf.best_index].max() == pytest.approx(10.0, abs=0.05)


def test_one_field_at_a_time():
    a = _synthetic(20, snr=1.0, field=0).chunks[0]
    b = _synthetic(20, snr=1.0, field=1).chunks[0]
    with pytest.raises(ValueError, match="one field at a time"):
        _finder(DataHandler(chunks=[a, b], metadata=_synthetic(20, snr=1.0).metadata))


def test_template_weighting_equals_natural_for_a_point_source():
    data = _synthetic(50, snr=10.0)
    assert _peak(data, "template") == pytest.approx(_peak(data, "natural"), rel=1e-12)


def test_template_weighting_recovers_a_resolved_source():
    """Weighting by A(u, v) attains the optimal S/N; the plain average loses the predicted fraction."""
    size = 3.0  # arcsec sigma, resolved by the 30 klambda baselines
    data = _synthetic(50, snr=10.0, size=size)
    template = _peak(data, "template", src_00_bmin=size, src_00_bmaj=size)
    natural = _peak(data, "natural", src_00_bmin=size, src_00_bmaj=size)

    (chunk,) = data.chunks
    a = Gaussian(bmin=size, bmaj=size).envelope(chunk)
    kept = np.sum(chunk.w * a) / np.sqrt(np.sum(chunk.w) * np.sum(chunk.w * a**2))
    assert template == pytest.approx(10.0, abs=0.05)
    assert natural == pytest.approx(10.0 * kept, abs=0.05)
    assert kept < 0.8


def test_unknown_weighting_is_rejected(data):
    with pytest.raises(ValueError, match="weighting"):
        _finder(data, weighting="uniform")
