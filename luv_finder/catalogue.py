"""From a matched-filter search to a catalogue of line candidates.

A :class:`~luv_finder.matchedfilter.SearchResult`, PB-corrected and for a mosaic combined, is not
yet a catalogue: one source lights up many neighbouring grid points, and how often noise alone
reaches a given S/N is not known. Both are measured on the search's jackknife, one realisation of
the data's noise without the sky. :func:`catalogue` takes these steps:

1. The S/N is the filter's, which accounts for the noise correlation between channels (ALMA's
   Hanning response) measured on the jackknife, so the jackknife's S/N has unit variance.
   :func:`jackknife_spread` reports how far its spread is from 1, as a check; a search run with
   ``channel_correlation=None`` on Hanning data gives about 1.4-1.6. The tests below compare the
   data with the jackknife, so they would be unbiased either way.
2. :func:`noise_correlation` measures the correlation function rho of the response under the null
   over positions, channels and templates. A matched source's expected response has the same
   shape, ``E[r(p)] = (S/N)_0 rho(p - s)`` (Vio & Andreani 2021): spatially the dirty beam,
   sidelobes included, spectrally about sqrt(2) times the template width, and between Gaussian
   line templates of FWHM W1 and W2 ``sqrt(2 W1 W2 / (W1^2 + W2^2))``.
3. :func:`groups` merges the local maxima of the best-template S/N, brightest first, that lie
   within the half-power ellipsoid of rho around a brighter one: one group per source. The data,
   the jackknife and the negated data are grouped alike.
4. :func:`likelihood` compares the cumulative group counts of the data and the jackknife,
   ``Lambda(gamma) = N_data(>= gamma) / N_jk(>= gamma)`` (van Marrewijk et al. 2025, A&A 695,
   A204), and :func:`fidelity` their counts per S/N bin, ``1 - N_jk / N_data`` fitted with an
   error function (Walter et al. 2016, ApJ 833, 67, with the jackknife in place of the negative
   peaks). The jackknife's positive groups are the noise reference of both; the data's negative
   groups are kept for comparison only.
5. One table row per data group, ``detected`` where both pass their thresholds. Nothing is
   clipped: the candidates are ``cat[cat["detected"]]``.

Arrays are numpy; NaN marks positions and channels a search does not cover.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
from astropy.table import Table
from scipy import fft, ndimage, optimize, special

from .data import sky_direction
from .matchedfilter import TEMPLATE_KEYS, SearchResult
from .model import C_KMS

#: Channels of the spatial correlation are Fourier transformed in chunks whose transform stays near this size.
CHUNK_BYTES = 2**28

#: 1.4826 x the median absolute deviation is the standard deviation of a Gaussian.
MAD_TO_SIGMA = 1.4826


def _jackknife(result: SearchResult) -> SearchResult:
    if result.jackknife is None:
        raise ValueError(
            "the search has no jackknife to measure the noise on; run it with MatchedFilter.run(jackknife=True)"
        )
    return result.jackknife


def jackknife_spread(result: SearchResult) -> np.ndarray:
    """Spread of the jackknife S/N per template, ``(n_template,)``: 1 if the channel noise is independent.

    The spread is 1.4826 times the median absolute deviation about the median, over every finite
    position and channel, so a few outliers do not set it. Above 1 the search assumed less channel
    correlation than the noise has (for instance ``channel_correlation=None`` on ALMA's Hanning
    response), and its S/N is overestimated by that factor.

    Raises
    ------
    ValueError
        If ``result`` has no jackknife.
    """
    snr = _jackknife(result).snr
    spread = []
    for t in range(snr.shape[2]):
        x = snr[:, :, t, :]
        x = x[np.isfinite(x)]
        spread.append(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))
    return np.array(spread)


@dataclass(frozen=True, eq=False)
class NoiseCorrelation:
    """Correlation function of the matched-filter response under the null (see :func:`noise_correlation`).

    Attributes
    ----------
    spatial : (n_template, n_lag_ra, n_lag_dec) correlation over position lags, averaged over channels
    lags_ra, lags_dec : arcsec east and north, ascending, the axes of ``spatial``
    spectral : (n_template, n_lag_chan) correlation over channel lags 0, 1, ...
    templates : (n_template, n_template) correlation between templates at one position and channel
    hwhm_major, hwhm_minor : (n_template,) arcsec, semi-axes of the half-power region of ``spatial``
    pa : (n_template,) deg east of north of its major axis, in [0, 180)
    spectral_hwhm : (n_template,) channels, half width at half maximum of ``spectral``
    spectral_hwhm_kms : (n_template,) the same in km/s at the median frequency
    sidelobe : (n_template,) largest ``|rho|`` of ``spatial`` outside its main lobe
    """

    spatial: np.ndarray
    lags_ra: np.ndarray
    lags_dec: np.ndarray
    spectral: np.ndarray
    templates: np.ndarray
    hwhm_major: np.ndarray
    hwhm_minor: np.ndarray
    pa: np.ndarray
    spectral_hwhm: np.ndarray
    spectral_hwhm_kms: np.ndarray
    sidelobe: np.ndarray


def _channel_width(freqs: np.ndarray) -> float:
    """Median channel width in Hz; infinite for a single channel."""
    return float(np.median(np.abs(np.diff(freqs)))) if len(freqs) > 1 else np.inf


def _lattice(axis, max_lag: float) -> tuple[float, int]:
    """Signed step of a uniform position axis in arcsec and how many steps fit within ``max_lag``."""
    a = np.asarray(axis, dtype=float)
    if a.size < 2:
        return 1.0, 0
    step = (a[-1] - a[0]) / (a.size - 1)
    if not np.allclose(np.diff(a), step, rtol=1e-6, atol=0):
        raise ValueError("the search positions are not on a uniform lattice")
    return step, min(int(max_lag / abs(step) + 1e-9), a.size - 1)


def _windows(freqs: np.ndarray) -> list[slice]:
    """Runs of uniformly spaced channels: the spectral windows, as far as the frequencies tell.

    A channel step unlike both of its neighbours is a gap between windows.
    """
    step = np.diff(freqs)
    if step.size < 2:
        return [slice(None)]
    alike = np.isclose(step[1:], step[:-1], rtol=1e-6, atol=0)
    regular = np.append(alike, False) | np.insert(alike, 0, False)
    edges = [0, *(np.flatnonzero(~regular) + 1).tolist(), len(freqs)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def _lag_sums(x: np.ndarray, axes: tuple, lags: tuple) -> np.ndarray:
    """``sum_p x(p) x(p + l)`` for every lag ``|l_k| <= lags[k]`` along ``axes`` (sorted), summed over the other axes.

    The result is centred on lag 0. The transform is zero-padded to ``n + lag``, so no pair wraps around.
    """
    shape = [fft.next_fast_len(x.shape[a] + n) for a, n in zip(axes, lags, strict=True)]
    f = fft.rfftn(x, s=shape, axes=axes)
    power = np.sum(f.real**2 + f.imag**2, axis=tuple(a for a in range(x.ndim) if a not in axes))
    return fft.irfftn(power, s=shape)[np.ix_(*(np.arange(-n, n + 1) for n in lags))]


def _autocorrelation(x: np.ndarray, m: np.ndarray, axes: tuple, lags: tuple, chunks: list[slice]) -> np.ndarray:
    """NaN-aware autocorrelation of ``x`` (zero where the mask ``m`` is 0), normalised to 1 at lag 0.

    The lag sums of ``x`` are divided by those of ``m``, the number of pairs at each lag, so
    masked positions neither count as zeros nor bias the edges. Channel ``chunks`` of the
    ``(n_dra, n_ddec, n_chan)`` arrays are transformed one at a time and their sums added; a lag
    no pair covers is NaN.
    """
    num, den = (sum(_lag_sums(a[:, :, c], axes, lags) for c in chunks) for a in (x, m))
    rho = np.where(den > 0.5, num / np.where(den > 0.5, den, 1.0), np.nan)
    return rho / rho[tuple(lags)]


def _half_power(rho: np.ndarray, lags_ra: np.ndarray, lags_dec: np.ndarray) -> tuple[float, float, float]:
    """Semi-axes (arcsec) and PA (deg east of north) of the connected region ``rho >= 0.5`` around lag 0.

    For a uniform ellipse the variance along a semi-axis ``a`` is ``a^2 / 4``.
    """
    labels = ndimage.label(rho >= 0.5)[0]
    i, j = np.nonzero(labels == labels[len(lags_ra) // 2, len(lags_dec) // 2])
    variance, vectors = np.linalg.eigh(np.cov(np.stack([lags_ra[i], lags_dec[j]]), bias=True))
    minor, major = 2 * np.sqrt(np.maximum(variance, 0.0))
    return major, minor, np.degrees(np.arctan2(*vectors[:, 1])) % 180


def _sidelobe(rho: np.ndarray) -> float:
    """Largest ``|rho|`` outside the main lobe, the connected region ``rho > 0`` around lag 0."""
    labels = ndimage.label(rho > 0)[0]
    outside = rho[(labels != labels[rho.shape[0] // 2, rho.shape[1] // 2]) & np.isfinite(rho)]
    return float(np.abs(outside).max()) if outside.size else 0.0


def _half_width(rho: np.ndarray) -> float:
    """Lag of the first 0.5 crossing of ``rho`` (lags 0, 1, ...), linearly interpolated; NaN if it stays above."""
    below = np.flatnonzero(rho < 0.5)
    if not below.size:
        return np.nan
    k = below[0]
    return k - 1 + (rho[k - 1] - 0.5) / (rho[k - 1] - rho[k])


def noise_correlation(result: SearchResult, max_lag: float = 30.0, max_lag_chan: int | None = None) -> NoiseCorrelation:
    """Measure the correlation function of the response under the null on the search's jackknife.

    The jackknife S/N is standardised per template (zero mean, unit variance over its finite
    values), so the result does not depend on the jackknife's overall spread. NaN
    positions and channels are zero-filled and every lag's sum of products is divided by its number
    of unmasked pairs, both from zero-padded FFTs, so mosaic edges and holes do not bias it.
    Spectral lags are taken within runs of uniformly spaced channels only, never across the gap
    between two windows.

    Parameters
    ----------
    result : SearchResult
        With a jackknife, on a uniform position lattice.
    max_lag : float
        Largest position lag in arcsec, along each axis.
    max_lag_chan : int, optional
        Largest channel lag; by default 4 times the widest template's FWHM in channels at the
        median frequency.

    Raises
    ------
    ValueError
        If ``result`` has no jackknife or its positions are not on a uniform lattice.
    """
    snr = _jackknife(result).snr
    n_dra, n_ddec, n_template, n_chan = snr.shape
    (step_ra, lag_ra), (step_dec, lag_dec) = (_lattice(result.axes[k], max_lag) for k in ("dra", "ddec"))
    nu = np.median(result.freqs)
    dnu = _channel_width(result.freqs)
    if max_lag_chan is None:
        max_lag_chan = int(np.ceil(4 * nu * max(result.axes["width"]) / C_KMS / dnu))
    max_lag_chan = min(max_lag_chan, n_chan - 1)

    # standardised copy, (n_template, n_dra, n_ddec, n_chan), zero where masked
    z = np.array(np.moveaxis(snr, 2, 0), dtype=float, order="C")
    finite = np.isfinite(z)
    for zt, ft in zip(z, finite, strict=True):
        zt -= zt.mean(where=ft)
        zt /= zt.std(where=ft)
        zt[~ft] = 0.0

    per_chunk = max(1, CHUNK_BYTES // (32 * n_dra * n_ddec))
    chunks = [slice(c, c + per_chunk) for c in range(0, n_chan, per_chunk)]
    windows = _windows(result.freqs)
    spatial, spectral = [], []
    for x, ft in zip(z, finite, strict=True):
        m = ft.astype(float)
        rho = _autocorrelation(x, m, (0, 1), (lag_ra, lag_dec), chunks)
        # index lags follow the axes' order; flip a descending axis so the lags run east and north
        spatial.append(rho[:: int(np.sign(step_ra)), :: int(np.sign(step_dec))])
        spectral.append(_autocorrelation(x, m, (2,), (max_lag_chan,), windows)[max_lag_chan:])

    flat, flat_mask = z.reshape(n_template, -1), finite.reshape(n_template, -1)
    pairs = np.array([[np.count_nonzero(a & b) for b in flat_mask] for a in flat_mask])
    templates = flat @ flat.T / np.maximum(pairs, 1)

    lags_ra = abs(step_ra) * np.arange(-lag_ra, lag_ra + 1)
    lags_dec = abs(step_dec) * np.arange(-lag_dec, lag_dec + 1)
    major, minor, pa = np.array([_half_power(rho, lags_ra, lags_dec) for rho in spatial]).T
    spectral_hwhm = np.array([_half_width(rho) for rho in spectral])
    return NoiseCorrelation(
        spatial=np.array(spatial),
        lags_ra=lags_ra,
        lags_dec=lags_dec,
        spectral=np.array(spectral),
        templates=templates,
        hwhm_major=major,
        hwhm_minor=minor,
        pa=pa,
        spectral_hwhm=spectral_hwhm,
        spectral_hwhm_kms=spectral_hwhm * dnu / nu * C_KMS,
        sidelobe=np.array([_sidelobe(rho) for rho in spatial]),
    )


@dataclass(frozen=True)
class Group:
    """Local maxima of the S/N merged into one candidate (see :func:`groups`).

    Attributes
    ----------
    i, j, t, k : int, indices of its brightest maximum along (dra, ddec, template, channel)
    snr : float, the S/N there
    n_peaks : int, local maxima merged, the brightest included
    """

    i: int
    j: int
    t: int
    k: int
    snr: float
    n_peaks: int


def _maxima(snr: np.ndarray, floor: float) -> tuple[np.ndarray, ...]:
    """Indices (i, j, t, k) and S/N of the 3x3x3 local maxima >= ``floor`` of the best-template S/N, brightest first."""
    best = np.fmax.reduce(snr, axis=2)
    best = np.where(np.isnan(best), -np.inf, best)
    i, j, k = np.nonzero((best == ndimage.maximum_filter(best, size=3, mode="nearest")) & (best >= floor))
    order = np.argsort(-best[i, j, k], kind="stable")
    i, j, k = i[order], j[order], k[order]
    t = np.nanargmax(snr[i, j, :, k], axis=1) if i.size else np.zeros(0, dtype=int)
    return i, j, t, k, best[i, j, k]


def groups(snr: np.ndarray, result: SearchResult, nc: NoiseCorrelation, floor: float = 4.0) -> list[Group]:
    """Group the local maxima of ``snr`` into candidates, one per source, brightest first.

    The maxima are the points at or above ``floor`` that no neighbour in the 3x3x3 box over
    (dra, ddec, channel) exceeds, of the S/N maximised over templates (NaN counts as -inf). Taken
    brightest first, each maximum not yet in a group starts one, its representative r, and takes in
    every fainter free maximum inside r's half-power ellipsoid of the noise correlation,

    ``(d_major / hwhm_major)^2 + (d_minor / hwhm_minor)^2 + (d_chan / spectral_hwhm)^2 <= 1``

    for the template of r: (d_major, d_minor) is the sky offset along the axes of rho, rotated by
    its ``pa``, and d_chan the frequency difference in median channel widths, so maxima in
    different windows are as far apart as their frequencies.

    Parameters
    ----------
    snr : (n_dra, n_ddec, n_template, n_chan) array
        S/N on the grid of ``result``: its data, its jackknife or the negated data.
    result : SearchResult
        Gives the positions and frequencies.
    nc : NoiseCorrelation
        Of ``result`` (see :func:`noise_correlation`).
    floor : float
        Lowest S/N of a maximum.
    """
    i, j, t, k, value = _maxima(snr, floor)
    x, y = np.asarray(result.axes["dra"])[i], np.asarray(result.axes["ddec"])[j]
    chan = result.freqs[k] / _channel_width(result.freqs)
    pa = np.radians(nc.pa)
    owner = np.full(i.size, -1)
    found = []
    for r in range(i.size):
        if owner[r] >= 0:
            continue
        free, tr = np.flatnonzero(owner < 0), t[r]
        dx, dy, dc = x[free] - x[r], y[free] - y[r], chan[free] - chan[r]
        major = dx * np.sin(pa[tr]) + dy * np.cos(pa[tr])
        minor = dx * np.cos(pa[tr]) - dy * np.sin(pa[tr])
        r2 = (major / nc.hwhm_major[tr]) ** 2 + (minor / nc.hwhm_minor[tr]) ** 2 + (dc / nc.spectral_hwhm[tr]) ** 2
        inside = free[r2 <= 1]
        owner[inside] = r
        found.append(Group(int(i[r]), int(j[r]), int(tr), int(k[r]), float(value[r]), inside.size))
    return found


def likelihood(snr_data, snr_noise) -> tuple[np.ndarray, np.ndarray]:
    """Likelihood ratio ``Lambda(gamma) = N_data(>= gamma) / N_noise(>= gamma)`` at every data S/N gamma.

    Empirical cumulative counts of groups, no fit (van Marrewijk et al. 2025).

    Parameters
    ----------
    snr_data, snr_noise : (n,) arrays
        S/N of the data's and the noise's (jackknife's) groups.

    Returns
    -------
    ratio : (n_data,) Lambda; where no noise group reaches gamma, ``N_noise = 1``
    lower_limit : (n_data,) bool, where no noise group reaches gamma, so ``ratio`` is a lower limit
    """
    snr_data, snr_noise = np.asarray(snr_data, dtype=float), np.asarray(snr_noise, dtype=float)
    n_data, n_noise = (x.size - np.searchsorted(np.sort(x), snr_data) for x in (snr_data, snr_noise))
    return n_data / np.maximum(n_noise, 1), n_noise == 0


def fidelity_curve(snr, centre, sigma):
    return 0.5 * special.erf((snr - centre) / sigma) + 0.5


def fidelity(snr_data, snr_noise, floor: float, step: float = 0.25) -> tuple[np.ndarray, tuple, dict]:
    """Fidelity ``1 - N_noise / N_data`` per S/N bin, fitted, at every data S/N (Walter et al. 2016).

    The groups are counted in bins of ``step`` from ``floor``. Over the bins holding data the
    fidelity, clipped to [0, 1], is fitted with ``0.5 erf((snr - C) / sigma) + 0.5``, weighted by
    its Poisson error ``sqrt(max(N_noise, 1) + N_noise^2 / N_data) / N_data``. With fewer than 3
    such bins, or if the fit fails, the fidelity is NaN and a warning says why.

    Returns
    -------
    values : (n_data,) the fitted fidelity at each data S/N
    fit : (C, sigma)
    bins : dict of (n_bin,) arrays over the bins holding data: ``snr`` (centre), ``n_data``,
        ``n_noise``, ``fidelity`` and its ``error``
    """
    snr_data, snr_noise = np.asarray(snr_data, dtype=float), np.asarray(snr_noise, dtype=float)
    top = max(snr_data.max(initial=floor), snr_noise.max(initial=floor))
    edges = floor + step * np.arange(int((top - floor) / step) + 2)
    n_data, n_noise = (np.histogram(x, edges)[0] for x in (snr_data, snr_noise))
    use = n_data > 0
    d, n = n_data[use], n_noise[use]
    bins = {
        "snr": edges[:-1][use] + step / 2,
        "n_data": d,
        "n_noise": n,
        "fidelity": np.clip(1 - n / d, 0, 1),
        "error": np.sqrt(np.maximum(n, 1) + n**2 / d) / d,
    }
    fit = (np.nan, np.nan)
    if use.sum() < 3:
        warnings.warn(f"only {use.sum()} S/N bins hold data groups, too few to fit the fidelity", stacklevel=2)
        return np.full(snr_data.shape, np.nan), fit, bins
    p0 = (bins["snr"][np.argmin(np.abs(bins["fidelity"] - 0.5))], 1.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
            fit = tuple(
                optimize.curve_fit(
                    fidelity_curve,
                    bins["snr"],
                    bins["fidelity"],
                    p0=p0,
                    sigma=bins["error"],
                    absolute_sigma=True,
                    bounds=([-np.inf, 1e-3], np.inf),
                )[0]
            )
    except (RuntimeError, ValueError) as err:
        warnings.warn(f"the fidelity fit failed: {err}", stacklevel=2)
    return fidelity_curve(snr_data, *fit), fit, bins


#: Units of the :func:`catalogue` columns.
UNITS = {
    "dra": "arcsec",
    "ddec": "arcsec",
    "ra": "deg",
    "dec": "deg",
    "freq_ghz": "GHz",
    "bmin": "arcsec",
    "bmaj": "arcsec",
    "pa": "deg",
    "width": "km/s",
    "flux": "Jy",
    "error": "Jy",
    "line_flux": "Jy km/s",
    "line_flux_error": "Jy km/s",
}


def catalogue(result: SearchResult, ref=None, floor: float = 4.0, k: float = 3.0, fidelity_min: float = 0.6) -> Table:
    """Catalogue the line candidates of a search: one row per group of the data, by S/N descending.

    The noise correlation of ``result`` is measured, and the data, the jackknife and the negated
    data are grouped (:func:`groups`) on the filter's S/N (see the module docstring). Each data
    group gets the likelihood ratio and the fidelity against the jackknife's groups
    (:func:`likelihood`, :func:`fidelity`). A group passes the likelihood ratio if
    ``Lambda >= k`` or no jackknife group is as bright: there ``Lambda`` is only a lower limit,
    bounded by the number of brighter data groups. It is ``detected`` if it passes and its
    fidelity is at least ``fidelity_min``, or the likelihood alone if the fidelity could not be
    fitted. Nothing is clipped: the candidates are ``cat[cat["detected"]]``.

    Parameters
    ----------
    result : SearchResult
        PB-corrected (and, for a mosaic, combined) search with its jackknife.
    ref : (RA, Dec) in rad, optional
        The sky frame's reference (:attr:`~luv_finder.data.Metadata.ref`); adds ``ra`` and ``dec``.
    floor : float
        Lowest S/N of a local maximum, and the lower edge of the fidelity bins.
    k : float
        Likelihood-ratio threshold.
    fidelity_min : float
        Fidelity threshold.

    Returns
    -------
    astropy.table.Table
        Columns ``id``, ``dra``, ``ddec`` (arcsec), ``ra``, ``dec`` (deg, with ``ref``),
        ``freq_ghz``, the template's ``bmin``, ``bmaj``, ``pa`` and ``width``, the ``snr``,
        ``flux`` and ``error`` (Jy) and ``line_flux``, ``line_flux_error`` (Jy km/s) at the
        brightest maximum, ``coverage``, ``n_peaks``, ``likelihood``, ``likelihood_lower_limit``,
        ``fidelity`` and ``detected``. ``meta`` holds the ``jackknife_spread``, the noise
        correlation's ``hwhm_major``, ``hwhm_minor``, ``pa``, ``spectral_hwhm`` and ``sidelobe``,
        the thresholds, the fidelity fit (``fidelity_centre``, ``fidelity_sigma``), the lowest
        data S/N with ``Lambda >= k`` (``snr_likelihood``) and where the fitted fidelity reaches
        ``fidelity_min`` (``snr_fidelity``), NaN if none, the S/N of every group of the data, the
        jackknife and the negated data (``snr_data``, ``snr_jackknife``, ``snr_negative``) and the
        binned fidelity (``fidelity_bins``), all as lists and floats.

    Raises
    ------
    ValueError
        If ``result`` has no jackknife.
    """
    nc = noise_correlation(result)
    found = {
        name: groups(values, result, nc, floor)
        for name, values in (("data", result.snr), ("jackknife", result.jackknife.snr), ("negative", -result.snr))
    }
    snr = {name: np.array([g.snr for g in gs]) for name, gs in found.items()}
    ratio, lower = likelihood(snr["data"], snr["jackknife"])
    fid, (centre, sigma), bins = fidelity(snr["data"], snr["jackknife"], floor)
    passes = (ratio >= k) | lower
    detected = passes & ((fid >= fidelity_min) | np.isnan(fid))

    i, j, t, c = (np.array([getattr(g, a) for g in found["data"]], dtype=int) for a in "ijtk")
    templates = np.array(list(product(*(result.axes[a] for a in TEMPLATE_KEYS))))[t]
    line_flux, line_flux_error = (a[i, j, t, c] for a in result.integrated_flux())
    columns = {"id": np.arange(1, i.size + 1), "dra": np.asarray(result.axes["dra"])[i]}
    columns["ddec"] = np.asarray(result.axes["ddec"])[j]
    if ref is not None:
        columns["ra"], columns["dec"] = np.degrees(sky_direction((columns["dra"], columns["ddec"]), ref))
    columns |= {"freq_ghz": result.freqs[c] / 1e9, **dict(zip(TEMPLATE_KEYS, templates.T, strict=True))}
    columns |= {key: getattr(result, key)[i, j, t, c] for key in ("snr", "flux", "error")}
    columns |= {"line_flux": line_flux, "line_flux_error": line_flux_error, "coverage": result.coverage[i, j, c]}
    columns |= {
        "n_peaks": np.array([g.n_peaks for g in found["data"]], dtype=int),
        "likelihood": ratio,
        "likelihood_lower_limit": lower,
        "fidelity": fid,
        "detected": detected,
    }

    meta = {"jackknife_spread": jackknife_spread(result).tolist()}
    meta |= {key: getattr(nc, key).tolist() for key in ("hwhm_major", "hwhm_minor", "pa", "spectral_hwhm", "sidelobe")}
    meta |= {"floor": float(floor), "k": float(k), "fidelity_min": float(fidelity_min)}
    meta |= {"fidelity_centre": float(centre), "fidelity_sigma": float(sigma)}
    meta["snr_likelihood"] = float(snr["data"][ratio >= k].min()) if np.any(ratio >= k) else np.nan
    meta["snr_fidelity"] = float(centre + sigma * special.erfinv(2 * fidelity_min - 1))
    meta |= {f"snr_{name}": values.tolist() for name, values in snr.items()}
    meta["fidelity_bins"] = {key: values.tolist() for key, values in bins.items()}

    cat = Table(columns, meta=meta)
    for name, unit in UNITS.items():
        if name in cat.colnames:
            cat[name].unit = unit
    return cat
