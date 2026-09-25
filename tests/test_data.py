from types import SimpleNamespace

import numpy as np

from luv_finder import DataHandler, Gaussian
from luv_finder.data import UV_FIELDS


def test_npz_roundtrip(tmp_path, data):
    out = tmp_path / "rt.npz"
    data.to_npz(out)
    back = DataHandler.from_npz(out)
    for k in UV_FIELDS:
        assert np.array_equal(getattr(back.uvdata, k), getattr(data.uvdata, k))
    assert back.metadata.dish_diameter == data.metadata.dish_diameter


def test_shape_helpers(data):
    n = data.n_freqs(data.uvdata) * data.n_visbs(data.uvdata)
    assert n == data.uvdata.UVreals.size


def test_metadata_scales(data):
    assert 10 < data.metadata.primarybeamsize() < 300  # arcsec, band 1-ish at 40 GHz
    assert data.metadata.minresolution() < data.metadata.primarybeamsize()


def test_phase_shift_recentres_model(data):
    g = Gaussian(dra=6.0, ddec=-3.0, nu_center=40e9, width=300.0)
    uv = g.profile(data.uvdata)
    shifted = data.apply_phase_shift(6.0, -3.0, uv)
    assert np.abs(shifted.UVimags_shifted).max() < 1e-6 * np.abs(shifted.UVreals_shifted).max()


def test_jackknife_removes_signal(data):
    """Differencing consecutive integrations cancels a constant signal."""
    strong = Gaussian(total_flux=50.0, nu_center=40e9, width=300.0).profile(data.uvdata)
    jacked = data.jackknife(strong)
    assert jacked.jacked
    assert jacked.UVreals.size <= strong.UVreals.size // 2 + strong.UVreals.size % 2
    assert np.allclose(jacked.UVreals, 0.0, atol=1e-6 * strong.UVreals.max())


def test_jackknife_splits_by_integration_not_randomly(data):
    """Every pair of consecutive integrations is differenced, in order."""
    nt, nb = 6, 4
    times = np.repeat(np.arange(nt, dtype=float), nb)
    ns = SimpleNamespace(
        UVreals=times.copy(),
        UVimags=np.zeros(nt * nb),
        uvtimes=times.copy(),
        uvwghts=np.ones(nt * nb),
        uvfreqs=np.full(nt * nb, 40e9),
        uwaves=np.ones(nt * nb),
        vwaves=np.ones(nt * nb),
        jacked=False,
    )
    jacked = DataHandler(uvdata=ns, dish_diameter=12.0).jackknife(ns)
    # pairs (0,1), (2,3), (4,5) -> 0.5 * (even - odd) = -0.5 everywhere
    assert np.allclose(jacked.UVreals, -0.5)
