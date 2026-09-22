import functools
from types import SimpleNamespace

import numpy as np
import pytest

from luv_finder import DataHandler, Gaussian, MatchedFilter, Model
from luv_finder.matchedfilter import _grid_point_response, nu_center_func

C_KMS = 299792.458


def _finder(data, **grid):
    g = Gaussian()
    g.grid = {
        "bmin": data.metadata.minresolution() / 10,
        "bmaj": data.metadata.minresolution() / 10,
        "width": 300.0,
        "nu_center": functools.partial(nu_center_func, uvfreq_min=data.uvdata.uvfreqs.min()),
        **grid,
    }
    m = Model()
    m.addcomponent(g)
    return MatchedFilter(data, m)


def _synthetic(nchan, snr, nvis=400, seed=0):
    """Noiseless visibilities holding a line of exactly the requested S/N."""
    r = np.random.default_rng(seed)
    df = 40e9 * 100 / C_KMS
    freqs = 40e9 + (np.arange(nchan) - nchan / 2) * df
    uw = r.uniform(-3e4, 3e4, nvis)
    vw = r.uniform(-3e4, 3e4, nvis)
    ns = SimpleNamespace(
        uvfreqs=np.repeat(freqs, nvis),
        uwaves=np.tile(uw, nchan),
        vwaves=np.tile(vw, nchan),
        uvwghts=np.ones(nchan * nvis),
        uvtimes=np.tile(np.arange(nvis), nchan),
        uvdists=np.tile(np.hypot(uw, vw), nchan) * 1e-3,
        jacked=False,
        UVreals=np.zeros(nchan * nvis),
        UVimags=np.zeros(nchan * nvis),
    )
    sig = Gaussian(nu_center=freqs[nchan // 2], width=300.0, total_flux=1.0).profile(ns).UVreals
    s_ch, w_ch = np.average(sig.reshape(nchan, nvis), weights=ns.uvwghts.reshape(nchan, nvis), axis=1, returned=True)
    ns.UVreals = sig * (snr / np.sqrt(np.sum(s_ch**2 * w_ch)))
    return DataHandler(uvdata=ns, dish_diameter=12.0)


def _peak(data, **params):
    mf = _finder(data)
    nf, nv = data.n_freqs(data.uvdata), data.n_visbs(data.uvdata)
    p = {
        "src_00_dra": 0.0,
        "src_00_ddec": 0.0,
        "src_00_bmin": 0.0,
        "src_00_bmaj": 0.0,
        "src_00_width": 300.0,
        "src_00_total_flux": 1.0,
        "src_00_nu_center": nu_center_func(300.0, data.uvdata.uvfreqs.min()),
        **params,
    }
    return _grid_point_response((p, mf, nv, nf))[0].max()


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
    assert all("src_00_nu_center" in p for p in pts)


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
