import numpy as np
import pytest

from luv_finder import Gaussian, Model
from luv_finder.data import C, Chunk
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
    (chunk,) = data.chunks
    vis = Gaussian(total_flux=1.0, nu_center=40e9, width=300.0).profile(chunk)
    assert np.allclose(vis.imag, 0.0)
    assert np.allclose(vis.real, vis.real[:, :1])


def test_spectral_integral_matches_total_flux():
    g = Gaussian(total_flux=2.0, nu_center=40e9, width=300.0)
    freqs = np.linspace(39.5e9, 40.5e9, 4001)
    flux_hz = np.trapezoid(g.spectrum(freqs), freqs)
    assert flux_hz * C_KMS / g.nu_center == pytest.approx(2.0, rel=1e-3)


def test_offset_source_phase_gradient(data):
    (chunk,) = data.chunks
    vis = Gaussian(dra=5.0, nu_center=40e9, width=300.0).profile(chunk)
    assert np.abs(vis.imag).max() > 0
    assert np.allclose(np.abs(vis), Gaussian(nu_center=40e9, width=300.0).profile(chunk).real)


def test_position_angle_rotates_the_axes(data):
    """Turning by 90 degrees swaps the axes; turning by 180 changes nothing."""
    (chunk,) = data.chunks
    env = lambda **kw: Gaussian(**kw).envelope(chunk)  # noqa: E731
    assert np.allclose(env(bmaj=4.0, bmin=1.0, pa=0.0), env(bmaj=1.0, bmin=4.0, pa=90.0))
    assert np.allclose(env(bmaj=4.0, bmin=1.0, pa=30.0), env(bmaj=4.0, bmin=1.0, pa=210.0))


@pytest.mark.parametrize(("pa", "resolved"), [(0.0, "v"), (90.0, "u"), (45.0, "u+v")])
def test_major_axis_lies_at_the_position_angle(pa, resolved):
    """A source long along pa (east of north) is resolved fastest along that direction in uv."""
    k = 3e4  # wavelengths
    baselines = {
        "u": (k, 0.0),
        "v": (0.0, k),
        "u+v": (k / np.sqrt(2), k / np.sqrt(2)),
        "u-v": (k / np.sqrt(2), -k / np.sqrt(2)),
    }
    u, v = np.array(list(baselines.values())).T * C / 40e9
    n = len(u)
    empty = np.zeros((1, n))
    chunk = Chunk(0, 0, np.zeros(2), np.array([40e9]), u, v, empty[0], np.arange(n), empty + 0j, np.ones(n), empty > 0)
    env = dict(zip(baselines, Gaussian(bmaj=3.0, bmin=0.5, pa=pa).envelope(chunk)[0], strict=True))
    assert min(env, key=env.get) == resolved
