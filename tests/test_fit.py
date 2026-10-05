import dataclasses

import numpy as np
import pytest
from astropy.table import Table

from luv_finder import DataHandler, Gaussian, MatchedFilter
from luv_finder.catalogue import catalogue
from luv_finder.data import Chunk, Metadata, sky_offset
from luv_finder.fit import COLUMNS, LINE_FLUX, LineWindow, fit_line, fit_lines
from luv_finder.matchedfilter import build_grid, grid_model, pb_corrected
from luv_finder.model import C_KMS, FWHM_TO_SIGMA
from luv_finder.utils import C, primary_beam

NU0, N_CHAN, DNU = 100e9, 64, 10e6  # Hz: channels of 30 km/s
OFFSETS = ((0.0, 0.0), (30.0, 0.0))  # pointing centres, arcsec; the primary beam's FWHM is 58 arcsec
SOURCE = {"dra": 12.0, "ddec": -5.0, "bmaj": 2.0, "bmin": 1.0, "pa": 40.0}  # PB 0.86 and 0.74
LINE = {"nu0": NU0 + 2.3 * DNU, "width": 300.0, "peak": 0.2}
CONTINUUM = (0.05, 0.01, -0.02)  # Jy, coefficients in the channel index scaled to [-1, 1]
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


def _data(source=SOURCE, noise=False, seed=0, n_row=3000) -> DataHandler:
    """Two pointings of the source, random uv within 30 klambda (resolution 4.9 arcsec), weights 0.5-2, 5% flagged."""
    rng = np.random.default_rng(seed)
    spectrum = _spectrum(continuum=CONTINUUM, **LINE)
    chunks = []
    for field, offset in enumerate(OFFSETS):
        u, v = rng.uniform(-3e4, 3e4, (2, n_row)) * C / NU0
        w_row = rng.uniform(0.5, 2.0, n_row)
        flag = rng.random((N_CHAN, n_row)) < 0.05
        rows = np.arange(n_row)
        chunk = Chunk(field, 0, np.asarray(offset), FREQ, u, v, rows * 1.0, rows, np.zeros(flag.shape), w_row, flag)
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
    window = LineWindow(data.chunks, data.metadata.dish_diameter)

    def direct(spectral, spatial):
        spectrum = window.design(*spectral[-2:]) @ spectral[:-2]
        return sum(np.sum(c.w * np.abs(c.vis - _model(c, spectrum, *spatial)) ** 2) for c in data.chunks)

    own = sum(np.sum(c.w * np.abs(c.vis) ** 2) for c in data.chunks)
    for spectral, spatial in [
        ([0.2, 0.05, 0.01, -0.02, LINE["nu0"], 300.0], (12.0, -5.0, 2.0, 1.0, 40.0)),
        ([0.1, 0.0, 0.03, 0.01, LINE["nu0"] + 3 * DNU, 200.0], (10.0, -3.0, 0.5, 1.5, 100.0)),
    ]:
        assert window.chi2(np.array(spectral), spatial) + own == pytest.approx(direct(spectral, spatial), rel=1e-10)


def test_noise_free_fit_recovers_every_parameter():
    fit = fit_line(_data(), *START)
    assert fit["fit_converged"] and not fit["fit_point"] and fit["fit_n_pointings"] == 2
    for key, pull in _pulls(fit, _truth()).items():
        assert abs(pull) < 0.01, key
    assert abs(fit["fit_pa"] - SOURCE["pa"]) < 0.01 * fit["fit_pa_error"]
    assert fit["fit_chi2_reduced"] < 1e-8
    np.testing.assert_allclose(
        sky_offset(np.radians([fit["fit_ra"], fit["fit_dec"]]), (0.0, 0.0)), [12.0, -5.0], atol=1e-3
    )


def test_noisy_fit_is_within_its_errors():
    fit = fit_line(_data(noise=True), *START)
    assert fit["fit_converged"] and not fit["fit_point"]
    for key, pull in _pulls(fit, _truth()).items():
        assert abs(pull) < 3, key
    assert abs(fit["fit_pa"] - SOURCE["pa"]) < 3 * fit["fit_pa_error"]
    assert 0.9 < fit["fit_chi2_reduced"] < 1.1


def test_point_source():
    point = {**SOURCE, "bmaj": 0.0, "bmin": 0.0}
    fit = fit_line(_data(point, noise=True), *START, point=True)
    assert fit["fit_point"] and fit["fit_converged"]
    assert fit["fit_bmaj"] == fit["fit_bmin"] == 0 and np.isnan(fit["fit_pa"])
    truth = {k: v for k, v in _truth(point).items() if k not in ("fit_bmaj", "fit_bmin")}
    for key, pull in _pulls(fit, truth).items():
        assert abs(pull) < 3, key
    # a Gaussian fit of a point source collapses both axes to zero and falls back
    assert fit_line(_data(point), *START)["fit_point"]


def test_point_fit_of_a_resolved_source_misses_flux():
    fit = fit_line(_data(noise=True), *START, point=True)
    assert fit["fit_line_flux"] < _truth()["fit_line_flux"] - 5 * fit["fit_line_flux_error"]


def test_size_at_its_bound_falls_back_to_a_point():
    fit = fit_line(_data(noise=True), *START, size_max=0.5)
    assert fit["fit_point"] and fit["fit_bmaj"] == 0


def test_fixed_position():
    fit = fit_line(_data(), SOURCE["dra"], SOURCE["ddec"], *START[2:], fixed_position=True)
    assert (fit["fit_dra"], fit["fit_ddec"]) == (SOURCE["dra"], SOURCE["ddec"])
    assert np.isnan(fit["fit_dra_error"]) and fit["fit_converged"]
    assert fit["fit_bmaj"] == pytest.approx(SOURCE["bmaj"], abs=1e-3)


def test_fit_lines_survives_a_failed_line():
    cat = Table(
        {"dra": [START[0], 0.0], "ddec": [START[1], 0.0], "freq_ghz": [START[2], 50.0], "width": [200.0, 200.0]}
    )
    with pytest.warns(UserWarning, match="row 1"):
        fitted = fit_lines(_data(noise=True), cat)
    assert set(COLUMNS) <= set(fitted.colnames) and fitted["fit_line_flux"].unit == "Jy km / s"
    assert fitted["fit_converged"].tolist() == [True, False]
    assert np.isfinite(fitted["fit_line_flux"][0]) and np.isnan(fitted["fit_line_flux"][1])


def test_fit_lines_of_the_fixture(data, truth, tmp_path):
    """Catalogue the fixture blind, then fit its detected lines: both injected lines come back."""
    lattice = data.metadata.minresolution() / 2 * np.arange(-18, 19)
    mf = MatchedFilter(data, grid_model(build_grid(data, {"dra": lattice, "ddec": lattice})))
    mf.run(jackknife=True)
    cat = fit_lines(data, catalogue(pb_corrected(mf.result, data), ref=data.metadata.ref))
    detected = cat[cat["detected"]]
    assert np.all(np.isfinite(detected["fit_line_flux"]))
    for src in truth["sources"]:
        near = np.hypot(detected["fit_dra"] - src["position_model"][0], detected["fit_ddec"] - src["position_model"][1])
        row = detected[np.argmin(near)]
        assert near.min() < 3 * np.hypot(row["fit_dra_error"], row["fit_ddec_error"])
        assert abs(row["fit_freq_ghz"] - src["line"]["mean"]) < 3 * row["fit_freq_ghz_error"]
    path = tmp_path / "fitted.ecsv"
    cat.write(path)
    back = Table.read(path)
    for name in COLUMNS:
        np.testing.assert_array_equal(back[name], cat[name])
