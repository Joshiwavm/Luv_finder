import functools

import numpy as np

from luv_finder import Gaussian, MatchedFilter, Model
from luv_finder.matchedfilter import nu_center_func


def _finder(data, **grid):
    g = Gaussian()
    g.grid = {
        "bmin": data.metadata.minresolution() / 10,
        "bmaj": data.metadata.minresolution() / 10,
        "width": 300.0,
        "total_flux": 1.0,
        "nu_center": functools.partial(nu_center_func, uvfreq_min=data.uvdata.uvfreqs.min()),
        **grid,
    }
    m = Model()
    m.addcomponent(g)
    return MatchedFilter(data, m)


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
