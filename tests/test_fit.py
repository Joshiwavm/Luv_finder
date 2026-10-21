import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.table import Table

from luv_finder import DataHandler, Gaussian, MatchedFilter
from luv_finder.catalogue import catalogue
from luv_finder.data import ChannelNoise, Chunk, Metadata, sky_offset
from luv_finder.fit import COLUMNS, _moments, _system, fit_line, fit_lines
from luv_finder.imaging import line_maps
from luv_finder.kernel import on_device
from luv_finder.matchedfilter import build_grid, grid_model, pb_corrected
from luv_finder.model import C_KMS, FWHM_TO_SIGMA, LINE_FLUX, covariance
from luv_finder.plotting import line_maps_check, line_spectra_check
from luv_finder.utils import C, primary_beam

NU0, N_CHAN, DNU = 100e9, 64, 10e6  # Hz: channels of 30 km/s
OFFSETS = ((0.0, 0.0), (30.0, 0.0))  # pointing centres, arcsec; the primary beam's FWHM is 58 arcsec
SOURCE = {"dra": 12.0, "ddec": -5.0, "bmaj": 2.0, "bmin": 1.0, "pa": 40.0}  # PB 0.86 and 0.74
LINE = {"nu0": NU0 + 2.3 * DNU, "width": 300.0, "peak": 0.2}
CONTINUUM = (0.05, 0.01, -0.02)  # Jy, coefficients in the channel index scaled to [-1, 1]
ORDER = len(CONTINUUM) - 1  # the fit models the injected quadratic
#: A catalogue's start: the nearest half-resolution grid point, channel and template.
START = (11.0, -4.0, (NU0 + 2 * DNU) / 1e9, 200.0)
FREQ = NU0 + (np.arange(N_CHAN) - N_CHAN // 2) * DNU


def _spectrum(nu0, width, peak, continuum):
    sigma = nu0 * width * FWHM_TO_SIGMA / C_KMS
    t = np.linspace(-1.0, 1.0, N_CHAN)
    return peak * np.exp(-0.5 * ((FREQ - nu0) / sigma) ** 2) + np.polynomial.polynomial.polyval(t, continuum)


def _model(chunk, spectrum, dra, ddec, bmaj, bmin, pa):
    """The source's visibilities in this pointing, visibility by visibility: ``PB F E e^{+i phi}``."""
    pb = primary_beam(np.hypot(dra - chunk.offset[0], ddec - chunk.offset[1]), chunk.freq, 12.0)
    shape = Gaussian(bmaj=bmaj, bmin=bmin, pa=pa).envelope(chunk)
    return (pb * spectrum)[:, None] * shape * np.conj(chunk.phase(dra, ddec))


def _data(source=SOURCE, noise=False, seed=0, n_row=3000, line=LINE) -> DataHandler:
    """Two pointings of the source, random uv within 30 klambda (resolution 4.9 arcsec), weights 0.5-2, 5% flagged."""
    rng = np.random.default_rng(seed)
    spectrum = _spectrum(continuum=CONTINUUM, **line)
    chunks = []
    for field, offset in enumerate(OFFSETS):
        u, v = rng.uniform(-3e4, 3e4, (2, n_row)) * C / NU0
        w_row = rng.uniform(0.5, 2.0, n_row)
        flag = rng.random((N_CHAN, n_row)) < 0.05
        rows = np.arange(n_row)
        w = np.where(flag, 0.0, w_row).astype(np.float32)
        chunk = Chunk(
            field,
            0,
            np.asarray(offset),
            FREQ,
            u,
            v,
            rows * 1.0,
            rows,
            np.zeros(n_row, dtype=int),
            np.zeros(flag.shape),
            w,
        )
        vis = _model(chunk, spectrum, **source)
        if noise:
            vis = vis + (rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape)) / np.sqrt(w_row)
        chunks.append(dataclasses.replace(chunk, X=np.where(flag, 0, w_row * vis)))
    return DataHandler(chunks=chunks, metadata=Metadata(12.0, (0.0, 0.0), NU0, 3e4 * np.sqrt(2)))


def _truth(source=SOURCE) -> dict:
    t = -1 + 2 * (LINE["nu0"] - FREQ[0]) / (FREQ[-1] - FREQ[0])
    return {
        **{f"fit_{k}": source[k] for k in ("dra", "ddec", "bmaj", "bmin")},
        "fit_freq_ghz": LINE["nu0"] / 1e9,
        "fit_width": LINE["width"],
        "fit_peak": LINE["peak"],
        "fit_line_flux": LINE["peak"] * LINE["width"] * LINE_FLUX,
        "fit_continuum": np.polynomial.polynomial.polyval(t, CONTINUUM),
    }


def _pulls(fit: dict, truth: dict) -> dict:
    return {k: (fit[k] - v) / fit[f"{k}_error"] for k, v in truth.items()}


def test_reduced_chi2_is_the_direct_sum():
    """S and Q give chi^2 = sum w |V - model|^2 over every visibility, less the data's own sum w |V|^2."""
    data = _data(noise=True)
    basis = np.linspace(-1.0, 1.0, N_CHAN)[:, None] ** np.arange(3)
    own = sum(np.sum(c.w * np.abs(c.vis) ** 2) for c in data.chunks)
    for (peak, *continuum, nu0, width), spatial in [
        ([0.2, 0.05, 0.01, -0.02, LINE["nu0"], 300.0], (12.0, -5.0, 2.0, 1.0, 40.0)),
        ([0.1, 0.0, 0.03, 0.01, LINE["nu0"] + 3 * DNU, 200.0], (10.0, -3.0, 0.5, 1.5, 100.0)),
    ]:
        spectrum = _spectrum(nu0, width, peak, continuum)
        direct = sum(np.sum(c.w * np.abs(c.vis - _model(c, spectrum, *spatial)) ** 2) for c in data.chunks)
        theta = [*spatial[:2], *covariance(*spatial[2:]), nu0, width, peak, *continuum]
        with jax.enable_x64(True):
            chunks = [on_device(c) for c in data.chunks]
            spectral = {chunks[0].spw: (jnp.asarray(FREQ), jnp.asarray(basis))}
            chi2 = float(_system(jnp.asarray(theta), _moments(theta, chunks), chunks, spectral, 12.0)[0])
        assert chi2 + own == pytest.approx(direct, rel=1e-10)


def test_noise_free_fit_recovers_every_parameter():
    fit = fit_line(_data(), *START, continuum_order=ORDER, channel_noise=None)
    assert fit["fit_converged"] and fit["fit_n_pointings"] == 2
    for key, pull in _pulls(fit, _truth()).items():
        assert abs(pull) < 0.01, key
    assert abs(fit["fit_pa"] - SOURCE["pa"]) < 0.01 * fit["fit_pa_error"]
    assert fit["fit_chi2_reduced"] < 1e-8
    np.testing.assert_allclose(
        sky_offset(np.radians([fit["fit_ra"], fit["fit_dec"]]), (0.0, 0.0)), [12.0, -5.0], atol=1e-3
    )


def test_noisy_fit_is_within_its_errors():
    fit = fit_line(_data(noise=True), *START, continuum_order=ORDER, channel_noise=None)
    assert fit["fit_converged"]
    for key, pull in _pulls(fit, _truth()).items():
        assert abs(pull) < 3, key
    assert abs(fit["fit_pa"] - SOURCE["pa"]) < 3 * fit["fit_pa_error"]
    assert 0.8 < fit["fit_chi2_reduced"] < 1.2


def test_point_source_fits_through_zero_size():
    point = {**SOURCE, "bmaj": 0.0, "bmin": 0.0}
    fit = fit_line(_data(point), *START, continuum_order=ORDER, channel_noise=None)
    assert fit["fit_converged"] and fit["fit_bmaj"] < 0.05
    truth = {k: v for k, v in _truth(point).items() if k not in ("fit_bmaj", "fit_bmin")}
    for key, pull in _pulls(fit, truth).items():
        assert abs(pull) < 0.01, key


def test_fit_lines_survives_a_failed_line():
    cat = Table(
        {
            "dra": [START[0], 0.0],
            "ddec": [START[1], 0.0],
            "freq_ghz": [START[2], 50.0],
            "width": [200.0, 200.0],
            "snr": [10.0, 10.0],
        }
    )
    with pytest.warns(UserWarning, match="row 1"):
        fitted = fit_lines(_data(noise=True), cat, channel_noise=None)
    assert set(COLUMNS) <= set(fitted.colnames) and fitted["fit_line_flux"].unit == "Jy km / s"
    assert fitted["fit_converged"].tolist() == [True, False]
    assert np.isfinite(fitted["fit_line_flux"][0]) and np.isnan(fitted["fit_line_flux"][1])


def test_fit_lines_of_the_fixture(data, truth, tmp_path):
    """Catalogue the fixture blind, then fit its detected lines: both injected lines come back."""
    lattice = data.metadata.minresolution() / 2 * np.arange(-18, 19)
    mf = MatchedFilter(data, grid_model(build_grid(data, {"dra": lattice, "ddec": lattice})))
    mf.run(jackknife=True)
    cat = catalogue(pb_corrected(mf.result, data), ref=data.metadata.ref)
    # fit only the detection nearest each injected line: fitting all of them costs ~2 s each
    nearest = [
        np.argmin(np.where(cat["detected"], np.hypot(cat["dra"] - x, cat["ddec"] - y), np.inf))
        for x, y in (src["position"] for src in truth["sources"])
    ]
    cat = fit_lines(data, cat, rows=np.isin(np.arange(len(cat)), nearest))
    for src, i in zip(truth["sources"], nearest, strict=True):
        row = cat[i]
        near = np.hypot(row["fit_dra"] - src["position"][0], row["fit_ddec"] - src["position"][1])
        assert near < 3 * np.hypot(row["fit_dra_error"], row["fit_ddec_error"])
        assert abs(row["fit_freq_ghz"] - src["line"]["mean"]) < 3 * row["fit_freq_ghz_error"]
    path = tmp_path / "fitted.ecsv"
    cat.write(path)
    back = Table.read(path)
    for name in COLUMNS:
        np.testing.assert_array_equal(back[name], cat[name])
    assert set(back.meta["spectra"]) == {str(cat["id"][i]) for i in nearest}
    maps = line_maps(data, cat)
    for (dra, ddec, m0), i in zip(maps, sorted(nearest), strict=True):
        peak = np.unravel_index(np.argmax(m0), m0.shape)
        assert np.hypot(dra[peak[0]] - cat["fit_dra"][i], ddec[peak[1]] - cat["fit_ddec"][i]) < 2.0
    far = cat[[nearest[0]]]
    far["fit_dra"] = 1e4  # arcsec: a fit that ran outside every pointing
    ((_, _, m0),) = line_maps(data, far)
    assert np.isnan(m0).all()
    line_maps_check(maps, back, plots_dir=str(tmp_path))
    line_spectra_check(back, plots_dir=str(tmp_path))
    assert (tmp_path / "line_maps.png").is_file() and (tmp_path / "line_spectra.png").is_file()


HANNING = np.array([1.0, 2 / 3, 1 / 6])


def test_channel_correlation_scales_the_errors_not_the_fit():
    """Hanning inflates every error by more than 5%, at most sqrt(8/3), and leaves the values alone."""
    data = _data(noise=True)
    hanning = {0: ChannelNoise(np.ones(N_CHAN), HANNING)}
    plain, smooth = (fit_line(data, *START, channel_noise=c) for c in (None, hanning))
    for key in ("fit_dra", "fit_freq_ghz", "fit_width", "fit_line_flux"):
        assert smooth[key] == plain[key]
        assert 1.05 * plain[f"{key}_error"] < smooth[f"{key}_error"] < np.sqrt(8 / 3) * plain[f"{key}_error"]


def test_the_fit_uses_the_visibility_weights():
    """A quarter of the weight on every visibility doubles every error and leaves a noiseless fit where it was."""
    data = _data()
    quarter = DataHandler(
        chunks=[dataclasses.replace(c, X=c.X / 4, w=c.w / 4) for c in data.chunks], metadata=data.metadata
    )
    plain, scaled = (fit_line(d, *START, channel_noise=None) for d in (data, quarter))
    for key in ("fit_dra", "fit_freq_ghz", "fit_width", "fit_line_flux"):
        assert abs(scaled[key] - plain[key]) < 1e-3 * plain[f"{key}_error"]
        assert scaled[f"{key}_error"] == pytest.approx(2 * plain[f"{key}_error"], rel=1e-3)


def test_a_line_narrower_than_the_bound_fits_at_min_width():
    """40 km/s, about one channel: the width stops at MIN_WIDTH with no error, and the rest still fits."""
    from luv_finder.fit import MIN_WIDTH

    fit = fit_line(_data(line={**LINE, "width": 40.0}), *START, continuum_order=ORDER, channel_noise=None)
    assert fit["fit_width"] == MIN_WIDTH and np.isnan(fit["fit_width_error"])
    assert abs(fit["fit_freq_ghz"] - LINE["nu0"] / 1e9) < 0.1 * DNU / 1e9


def _two_windows(shift, line, noise=False, seed=1, n_row=3000) -> DataHandler:
    """One pointing of a point source, in two windows of N_CHAN channels, the second ``shift`` channels above.

    The continuum is linear in frequency, so a degree-1 polynomial in each window's channels holds it exactly.
    """
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(-3e4, 3e4, (2, n_row)) * C / NU0
    w = np.ones((N_CHAN, n_row), dtype=np.float32)
    chunks = []
    for spw, freq in enumerate((FREQ, FREQ + shift * DNU)):
        sigma = line["nu0"] * line["width"] * FWHM_TO_SIGMA / C_KMS
        spectrum = line["peak"] * np.exp(-0.5 * ((freq - line["nu0"]) / sigma) ** 2) + 0.05 + 0.01 * (freq - NU0) / 1e9
        rows = np.arange(n_row)
        chunk = Chunk(
            0, spw, np.zeros(2), freq, u, v, rows * 1.0, rows, np.zeros(n_row, dtype=int), np.zeros(w.shape), w
        )
        vis = _model(chunk, spectrum, 12.0, -5.0, 0.0, 0.0, 0.0)
        if noise:
            vis = vis + (rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape))
        chunks.append(dataclasses.replace(chunk, X=w * vis))
    return DataHandler(chunks=chunks, metadata=Metadata(12.0, (0.0, 0.0), NU0, 3e4 * np.sqrt(2)))


@pytest.mark.parametrize("shift, nu0", [(32.5, NU0 + 16 * DNU), (N_CHAN + 1, NU0 + 32 * DNU)])
def test_a_line_in_two_windows_is_fitted_in_both(shift, nu0):
    """In two overlapping windows the line has both windows' information; across a gap, more: a half line
    alone is nearly degenerate with its window's continuum."""
    line = {"nu0": nu0, "width": 300.0, "peak": 0.2}
    both = _two_windows(shift, line)
    fit = fit_line(both, 12.0, -5.0, nu0 / 1e9, 300.0, continuum_order=1, channel_noise=None)
    assert fit["fit_peak"] == pytest.approx(0.2, rel=1e-6) and fit["fit_width"] == pytest.approx(300.0, rel=1e-6)
    assert fit["fit_freq_ghz"] == pytest.approx(nu0 / 1e9, rel=1e-10)
    alone = [
        fit_line(DataHandler(chunks=[c], metadata=both.metadata), 12.0, -5.0, nu0 / 1e9, 300.0,
                 continuum_order=1, channel_noise=None)
        for c in both.chunks
    ]  # fmt: skip
    information = sum(f["fit_peak_error"] ** -2 for f in alone)
    if shift < N_CHAN:
        assert fit["fit_peak_error"] ** -2 == pytest.approx(information, rel=0.1)
    else:
        assert fit["fit_peak_error"] ** -2 > 2 * information
