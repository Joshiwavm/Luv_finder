import dataclasses

import numpy as np
import pytest

from luv_finder import DataHandler, Gaussian
from luv_finder import data as data_module
from luv_finder.data import (
    ARCSEC,
    CHUNK_ARRAYS,
    Chunk,
    Metadata,
    channel_correlation,
    fields_in,
    load,
    pair_weight,
    sky_direction,
    sky_offset,
    stokes_i,
)
from luv_finder.utils import X_HALF, C, primary_beam, primary_beam_fwhm, primary_beam_radius


def _chunk(times, baselines, values=None, n_chan=3, field=0, spw=0, freq0=40e9):
    """Unit-weight rows (time, baseline) whose visibilities are ``values`` (default: the time)."""
    times = np.asarray(times, dtype=float)
    vis = np.broadcast_to(times if values is None else values, (n_chan, times.size)).astype(complex)
    return Chunk(
        field,
        spw,
        np.zeros(2),
        freq0 + 1e7 * np.arange(n_chan),
        np.ones(times.size),
        np.ones(times.size),
        times,
        np.asarray(baselines),
        vis.copy(),
        np.ones(times.size),
        np.zeros(vis.shape, dtype=bool),
    )


def test_npz_roundtrip(tmp_path, data):
    out = tmp_path / "rt.npz"
    data.to_npz(out)
    back = DataHandler.from_npz(out)
    assert back.metadata == data.metadata
    for k in CHUNK_ARRAYS:
        assert np.array_equal(getattr(back.chunks[0], k), getattr(data.chunks[0], k))


def test_npz_loads_only_the_selected_fields(tmp_path):
    chunks = [_chunk([0, 1], [0, 0], field=f, spw=s) for f in (0, 1) for s in (0, 1)]
    out = tmp_path / "fields.npz"
    DataHandler(chunks=chunks, metadata=Metadata(12.0, (0.0, 0.0), 40e9, 1e4)).to_npz(out)
    back = DataHandler.from_npz(out, fields=[1])
    assert back.fields == [1]
    assert [c.spw for c in back.chunks] == [0, 1]


def test_fields_in_an_npz_are_unique_and_sorted(tmp_path, fixture_npz, two_field_npz):
    assert fields_in(fixture_npz) == [0]
    assert fields_in(str(two_field_npz)) == [0, 1]
    chunks = [_chunk([0, 1], [0, 0], field=f, spw=s) for f in (5, 2) for s in (1, 0)]
    out = tmp_path / "fields.npz"
    DataHandler(chunks=chunks, metadata=Metadata(12.0, (0.0, 0.0), 40e9, 1e4)).to_npz(out)
    assert fields_in(out) == [2, 5]


def test_fields_in_a_measurement_set_are_its_target_fields(monkeypatch):
    monkeypatch.setattr(data_module, "target_selection", lambda path: (np.array([3, 1], dtype=np.int32), [0]))
    assert fields_in("obs.ms") == [1, 3]


def test_load_reads_all_or_the_selected_fields_of_an_npz(two_field_npz):
    assert load(two_field_npz).fields == [0, 1]
    assert load(str(two_field_npz), [1]).fields == [1]


def test_metadata_scales(data):
    assert 10 < data.metadata.primarybeamsize() < 300  # arcsec, band 1-ish at 40 GHz
    assert data.metadata.minresolution() < data.metadata.primarybeamsize()


def test_primary_beam_is_unity_on_axis_and_half_at_half_the_fwhm():
    fwhm = primary_beam_fwhm(92e9, 12.0)
    assert primary_beam(0.0, 92e9, 12.0) == 1.0
    assert primary_beam(fwhm / 2, 92e9, 12.0) == pytest.approx(0.5, abs=1e-9)
    assert X_HALF == pytest.approx(1.6163, abs=1e-4)


def test_primary_beam_decreases_over_the_main_lobe():
    offsets = np.linspace(0, primary_beam_radius(1e-3, 92e9, 12.0), 200)
    assert np.all(np.diff(primary_beam(offsets, 92e9, 12.0)) < 0)


def test_primary_beam_broadcasts_over_offsets_and_channels():
    offsets = np.linspace(0, 60, 5)[:, None]
    freqs = np.array([90e9, 92e9, 94e9])
    pb = primary_beam(offsets, freqs, 12.0)
    assert pb.shape == (5, 3)
    assert np.all(pb[0] == 1.0)
    assert np.all(np.diff(pb[1:], axis=1) < 0)  # a higher frequency has a narrower beam


def test_primary_beam_fwhm_scales_inversely_with_frequency():
    assert primary_beam_fwhm(184e9, 12.0) == pytest.approx(primary_beam_fwhm(92e9, 12.0) / 2)
    assert primary_beam_fwhm(92e9, 12.0) == pytest.approx(1.13 * C / 92e9 / 12.0 / ARCSEC)


def test_primary_beam_radius_inverts_the_beam():
    fwhm = primary_beam_fwhm(92e9, 12.0)
    assert primary_beam_radius(0.5, 92e9, 12.0) == pytest.approx(fwhm / 2)
    assert primary_beam(primary_beam_radius(0.2, 92e9, 12.0), 92e9, 12.0) == pytest.approx(0.2)
    assert primary_beam_radius(1.0, 92e9, 12.0) == 0.0


def test_metadata_primarybeamsize_is_the_fwhm():
    meta = Metadata(12.0, (0.0, 0.0), 92e9, 1e4)
    assert meta.primarybeamsize() == pytest.approx(63.3, abs=0.1)
    assert meta.primarybeamsize() == pytest.approx(primary_beam_fwhm(92e9, 12.0))
    assert meta.primarybeamsize(7.0) == pytest.approx(63.3 * 12 / 7, abs=0.2)


def test_phase_shift_recentres_model(data):
    (chunk,) = data.chunks
    g = Gaussian(dra=6.0, ddec=-3.0, nu_center=40e9, width=300.0)
    shifted = g.profile(chunk) * chunk.phase(6.0, -3.0)
    assert np.abs(shifted.imag).max() < 1e-6 * np.abs(shifted.real).max()


def test_sky_offset_is_east_and_north():
    ref = (1.0, -0.8)
    d = 30 * ARCSEC
    # an offset purely in RA curves north by ~theta^2 tan(dec) / 2 on the tangent plane
    assert sky_offset((ref[0] + d / np.cos(ref[1]), ref[1]), ref) == pytest.approx([30, 0], abs=1e-2)
    assert sky_offset((ref[0], ref[1] + d), ref) == pytest.approx([0, 30], abs=1e-6)


@pytest.mark.parametrize("ref", [(1.0, -0.8), (1e-5, 0.3), (2.0, np.radians(89.99)), (4.0, -np.radians(89.99))])
def test_sky_direction_inverts_sky_offset(ref):
    """Round trips either way, also across RA = 0 and over the pole, 36 arcsec from the last two references."""
    east, north = np.meshgrid([-60.0, -0.5, 0.0, 3.0, 45.0], [-60.0, 0.0, 30.0, 60.0])
    offset = np.array([east.ravel(), north.ravel()])
    direction = sky_direction(offset, ref)
    assert np.all((direction[0] >= 0) & (direction[0] < 2 * np.pi))
    np.testing.assert_allclose(sky_offset(direction, ref), offset, rtol=0, atol=1e-6)
    np.testing.assert_allclose(sky_direction(sky_offset(direction, ref), ref), direction, rtol=0, atol=1e-12)


def test_stokes_i_combines_hands_and_flags():
    data = np.array([np.full((2, 3), 1.0 + 1j), np.full((2, 3), 3.0 - 1j)])
    weight = np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 1.0]])
    flag = np.zeros((2, 2, 3), dtype=bool)
    flag[1, 0, 1] = True  # one hand of channel 0, row 1
    vis, w_row, flagged = stokes_i(data, weight, flag)
    assert np.allclose(vis, 2.0)
    assert np.allclose(w_row, [3.0, 2.0, 0.0])  # 4 w_a w_b / (w_a + w_b); 0 when a hand has no weight
    assert flagged.tolist() == [[False, True, False], [False, False, False]]


def test_spectrum_is_the_weighted_mean():
    chunk = _chunk([0, 0, 1], [0, 1, 0])
    flag = np.zeros_like(chunk.flag)
    flag[1, 0] = True
    w_row = np.array([1.0, 2.0, 3.0])
    chunk = dataclasses.replace(chunk, X=np.where(flag, 0.0, w_row) * (2.0 + 1.0j), w_row=w_row, flag=flag)
    assert np.allclose(DataHandler(chunks=[chunk]).spectrum(), 2.0 + 1.0j)


def test_jackknife_removes_signal(data):
    """Differencing consecutive integrations cancels a constant signal."""
    (chunk,) = data.chunks
    strong = Gaussian(total_flux=50.0, nu_center=40e9, width=300.0).profile(chunk)
    jacked = dataclasses.replace(chunk, X=chunk.w * strong).jackknife()
    assert jacked.X.shape[1] <= chunk.X.shape[1] // 2
    assert np.allclose(jacked.X, 0.0, atol=1e-6 * np.abs(strong).max())


def test_jackknife_splits_by_integration_not_randomly():
    """Every pair of consecutive integrations is differenced, in order."""
    nt, nb = 6, 4
    jacked = _chunk(np.repeat(np.arange(nt), nb), np.tile(np.arange(nb), nt)).jackknife()
    # pairs (0,1), (2,3), (4,5) -> 0.5 * (even - odd) = -0.5 everywhere, with weight 2
    assert np.allclose(jacked.vis, -0.5)
    assert np.allclose(jacked.w_row, 2.0)


def test_jackknife_pairs_rows_by_baseline():
    """Rows missing or reordered in one integration do not misalign the pairs."""
    times = [0, 0, 0, 0, 1, 1, 1]
    baselines = [0, 1, 2, 3, 3, 1, 0]  # baseline 2 dropped from the second integration
    jacked = _chunk(times, baselines, values=10.0 * np.array(times) + np.array(baselines)).jackknife()
    assert sorted(jacked.baseline) == [0, 1, 3]
    assert np.allclose(jacked.vis, -5.0)


def test_pair_weight_is_the_inverse_variance_of_the_mean():
    assert pair_weight(1.0, 3.0) == pytest.approx(3.0)
    assert pair_weight(0.0, 0.0) == 0.0


HANNING = (1.0, 2 / 3, 1 / 6)


def _noise_chunk(n_chan=32, n_row=20000, smooth=True, spw=0, seed=0):
    """Pure noise, consecutive integrations of one baseline; Hanning-smoothed along the channels."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_chan + 2, n_row)) + 1j * rng.standard_normal((n_chan + 2, n_row))
    noise = 0.25 * noise[:-2] + 0.5 * noise[1:-1] + 0.25 * noise[2:] if smooth else noise[1:-1]
    chunk = _chunk(np.arange(n_row), np.zeros(n_row, dtype=int), n_chan=n_chan, spw=spw)
    return dataclasses.replace(chunk, X=noise)


def test_channel_correlation_of_hanning_noise():
    rho = channel_correlation([_noise_chunk()])
    assert rho.size == 3 and rho == pytest.approx(HANNING, abs=0.01)


@pytest.mark.parametrize(
    "chunk", [_noise_chunk(smooth=False), _chunk(np.arange(100), np.zeros(100, dtype=int), values=np.ones(100))]
)
def test_channel_correlation_of_independent_or_no_noise(chunk):
    """White noise has no significant lag, and a noiseless (constant) signal leaves nothing to measure."""
    assert channel_correlation([chunk]).tolist() == [1.0]


def test_channel_correlation_per_spectral_window():
    data = DataHandler(chunks=[_noise_chunk(spw=0), _noise_chunk(smooth=False, spw=1, seed=1)], metadata=None)
    rho = data.channel_correlation()
    assert sorted(rho) == [0, 1] and rho[0].size == 3 and rho[1].tolist() == [1.0]
