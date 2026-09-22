import numpy as np
import pytest

from luv_finder import Gaussian, Model
from luv_finder.model import C_KMS


def test_prefixed_params_and_grid():
    g = Gaussian(nu_center=40e9)
    g.grid = {"dra": np.array([0.0, 1.0]), "width": 300.0}
    m = Model()
    m.addcomponent(g)
    assert "src_00_nu_center" in m.params
    assert set(m.grid) == {"src_00_dra", "src_00_width"}
    assert m.component(0) is g


def test_point_source_is_flat_in_uv(data):
    """Zero-size source at the phase centre: |V| = spectral profile, no phase."""
    g = Gaussian(total_flux=1.0, nu_center=40e9, width=300.0)
    uv = g.profile(data.uvdata)
    assert np.allclose(uv.UVimags, 0.0)
    ch = uv.uvfreqs == uv.uvfreqs.min()
    assert np.allclose(uv.UVreals[ch], uv.UVreals[ch][0])


def test_spectral_integral_matches_total_flux():
    g = Gaussian(total_flux=2.0, nu_center=40e9, width=300.0)
    freqs = np.linspace(39.5e9, 40.5e9, 4001)
    uv = type("uv", (), {})()
    uv.uvfreqs, uv.uwaves, uv.vwaves = freqs, np.zeros_like(freqs), np.zeros_like(freqs)
    spec = g.profile(uv).UVreals
    flux_hz = np.trapezoid(spec, freqs)
    assert flux_hz * C_KMS / g.nu_center == pytest.approx(2.0, rel=1e-3)


def test_offset_source_phase_gradient(data):
    g = Gaussian(dra=5.0, nu_center=40e9, width=300.0)
    uv = g.profile(data.uvdata)
    assert np.abs(uv.UVimags).max() > 0
    assert np.allclose(
        np.hypot(uv.UVreals, uv.UVimags), Gaussian(nu_center=40e9, width=300.0).profile(data.uvdata).UVreals
    )
