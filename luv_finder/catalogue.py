from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from astropy.table import Table, vstack
from scipy import fft, ndimage, optimize, special, stats

from .data import sky_direction
from .matchedfilter import TEMPLATE_KEYS, SearchResult
from .model import C_KMS
from .utils import BLOCK_BYTES


def _jackknife(result: SearchResult) -> SearchResult:
    if result.jackknife is None:
        raise ValueError(
            "the search has no jackknife to measure the noise on; run it with MatchedFilter.run(jackknife=True)"
        )
    return result.jackknife


def jackknife_spread(result: SearchResult) -> np.ndarray:
    """Robust spread of the jackknife S/N per template, ``(n_template,)``: 1 under H0."""
    snr = _jackknife(result).snr
    return np.array([stats.median_abs_deviation(x[np.isfinite(x)], scale="normal") for x in np.moveaxis(snr, 2, 0)])


@dataclass(frozen=True, eq=False)
class ResponseCorrelation:
    """Correlation function of the matched-filter response under the null (see :func:`response_correlation`)."""

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
    """Runs of uniformly spaced channels: the spectral windows, as far as the frequencies tell."""
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
    """NaN-aware autocorrelation of ``x`` (zero where the mask ``m`` is 0), normalised to 1 at lag 0."""
    num, den = (sum(_lag_sums(a[:, :, c], axes, lags) for c in chunks) for a in (x, m))
    rho = np.where(den > 0.5, num / np.where(den > 0.5, den, 1.0), np.nan)
    return rho / rho[tuple(lags)]


def _half_power(rho: np.ndarray, lags_ra: np.ndarray, lags_dec: np.ndarray) -> tuple[float, float, float]:
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


def response_correlation(
    result: SearchResult, max_lag: float = 30.0, max_lag_chan: int | None = None
) -> ResponseCorrelation:
    """Measure the correlation function of the response under the null on the search's jackknife."""
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

    per_chunk = max(1, BLOCK_BYTES // (32 * n_dra * n_ddec))
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
    return ResponseCorrelation(
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
    """Local maxima of the S/N merged into one candidate (see :func:`groups`)."""

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


def groups(snr: np.ndarray, result: SearchResult, nc: ResponseCorrelation, floor: float = 4.0) -> list[Group]:
    """Group the local maxima of ``snr`` into candidates, one per source, brightest first."""
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
    """``N_data(>= gamma) / N_noise(>= gamma)`` of peak counts at every data S/N (Vio & Andreani 2021).

    Above the brightest noise group the ratio diverges; those groups get its last finite value,
    ``N_data(>= that group) / 1``, as a lower limit (flagged in the second array).
    """
    snr_data, snr_noise = np.asarray(snr_data, dtype=float), np.asarray(snr_noise, dtype=float)
    n_data, n_noise = (x.size - np.searchsorted(np.sort(x), snr_data) for x in (snr_data, snr_noise))
    lower = n_noise == 0
    ratio = n_data / np.maximum(n_noise, 1)
    ratio[lower] = np.sum(snr_data >= snr_noise.max(initial=-np.inf))
    return ratio, lower


def fidelity(snr_data, snr_noise, floor: float, step: float = 0.25) -> tuple[np.ndarray, tuple, dict]:
    """Fidelity ``1 - N_noise / N_data`` per S/N bin, ``N_noise`` from a Gaussian tail fitted to ``snr_noise``.

    The noise groups above ``floor`` are fitted, unbinned, with ``N(>= s) = A erfc(s / sqrt(2) sigma)``
    (González-López et al. 2019), so the fidelity at high S/N comes from the tail and not from a few
    sparse bins, and an excess of data over noise near the floor leaves it alone. Every data group gets
    its bin's fidelity. Returns those, ``(A, sigma)`` and the bins: the binned counts, their fidelity
    with its error and the ``model`` fidelity.
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
    noise = snr_noise[snr_noise >= floor]
    if noise.size < 3:
        warnings.warn(f"only {noise.size} noise groups above S/N {floor:g}, too few to fit the noise", stacklevel=2)
        bins["model"] = np.full(d.shape, np.nan)
        return np.full(snr_data.shape, np.nan), (np.nan, np.nan), bins

    def nll(sigma):
        return np.sum(noise**2) / (2 * sigma**2) + noise.size * (np.log(sigma) + special.log_ndtr(-floor / sigma))

    sigma = optimize.minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x
    above = noise.size * np.exp(special.log_ndtr(-edges / sigma) - special.log_ndtr(-floor / sigma))
    model = np.clip(1 - (above[:-1] - above[1:]) / np.maximum(n_data, 1), 0, 1)
    bins["model"] = model[use]
    amplitude = noise.size / special.erfc(floor / (np.sqrt(2) * sigma))
    return model[np.searchsorted(edges, snr_data, side="right") - 1], (amplitude, sigma), bins


#: How the likelihood ratio and the fidelity combine into ``detected``: "and" (both) or "or" (either).
DETECTION_RULE = "and"

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


#: ``meta`` of :func:`catalogue` that belongs to one search, kept per part by :func:`merge_catalogues`.
SEARCH_META = ("jackknife_spread", "hwhm_major", "hwhm_minor", "pa", "spectral_hwhm", "sidelobe")


def catalogue(
    result: SearchResult,
    ref=None,
    floor: float = 4.0,
    k: float = 3.0,
    fidelity_min: float = 0.6,
    rule: str = DETECTION_RULE,
) -> Table:
    """Catalogue the line candidates of a search: one row per group of the data, by S/N descending.

    The noise correlation of ``result`` is measured, and the data, the jackknife and the negated
    data are grouped (:func:`groups`) on the filter's S/N (see the module docstring). Each data
    group gets the likelihood ratio and the fidelity against the jackknife's groups
    (:func:`likelihood`, :func:`fidelity`). A group passes the likelihood ratio if
    ``Lambda >= k`` or no jackknife group is as bright: there ``Lambda`` diverges, and its lower
    limit is the last finite value, at the brightest jackknife group. It is ``detected`` if it passes
    the likelihood ratio and its fidelity is at least ``fidelity_min`` (``rule="and"``, the default;
    a fidelity that could not be fitted does not count against it), or either (``rule="or"``). Nothing
    is clipped: the candidates are ``cat[cat["detected"]]``.

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
    rule : {"or", "and"}
        How the two tests combine into ``detected``.

    Returns
    -------
    astropy.table.Table
        Columns ``id``, ``dra``, ``ddec`` (arcsec), ``ra``, ``dec`` (deg, with ``ref``),
        ``freq_ghz``, the template's ``bmin``, ``bmaj``, ``pa`` and ``width``, the ``snr``,
        ``flux`` and ``error`` (Jy) and ``line_flux``, ``line_flux_error`` (Jy km/s) at the
        brightest maximum, ``coverage``, ``n_peaks``, ``likelihood``, ``likelihood_lower_limit``,
        ``fidelity`` and ``detected``. ``meta`` holds the ``jackknife_spread``, the noise
        correlation's ``hwhm_major``, ``hwhm_minor``, ``pa``, ``spectral_hwhm`` and ``sidelobe``,
        the thresholds, the jackknife's noise tail (``noise_amplitude``, ``noise_sigma``), the
        lowest data S/N with ``Lambda >= k`` (``snr_likelihood``) and with a fidelity of at least
        ``fidelity_min`` (``snr_fidelity``), NaN if none, the S/N of every group of the data, the
        jackknife and the negated data (``snr_data``, ``snr_jackknife``, ``snr_negative``) and the
        binned fidelity (``fidelity_bins``), all as lists and floats.

    Raises
    ------
    ValueError
        If ``result`` has no jackknife.
    """
    nc = response_correlation(result)
    found = {
        name: groups(values, result, nc, floor)
        for name, values in (("data", result.snr), ("jackknife", result.jackknife.snr), ("negative", -result.snr))
    }
    snr = {name: np.array([g.snr for g in gs]) for name, gs in found.items()}
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
    columns["n_peaks"] = np.array([g.n_peaks for g in found["data"]], dtype=int)
    reliable, meta = _reliability(snr, floor, k, fidelity_min, rule)
    search = {"jackknife_spread": jackknife_spread(result)} | {key: getattr(nc, key) for key in SEARCH_META[1:]}
    cat = Table(columns | reliable, meta={key: value.tolist() for key, value in search.items()} | meta)
    for name, unit in UNITS.items():
        if name in cat.colnames:
            cat[name].unit = unit
    return cat


def _reliability(
    snr: dict, floor: float, k: float, fidelity_min: float, rule: str = DETECTION_RULE
) -> tuple[dict, dict]:
    """Likelihood ratio, fidelity and detection of the data groups against the jackknife's, as columns and meta."""
    ratio, lower = likelihood(snr["data"], snr["jackknife"])
    fid, (amplitude, sigma), bins = fidelity(snr["data"], snr["jackknife"], floor)
    passes = (ratio >= k) | lower
    if rule == "or":
        detected = passes | (fid >= fidelity_min)
    elif rule == "and":
        detected = passes & ((fid >= fidelity_min) | np.isnan(fid))
    else:
        raise ValueError(f'rule is "or" or "and", not {rule!r}')
    columns = {"likelihood": ratio, "likelihood_lower_limit": lower, "fidelity": fid, "detected": detected}
    meta = {"floor": float(floor), "k": float(k), "fidelity_min": float(fidelity_min), "rule": rule}
    meta |= {"noise_amplitude": float(amplitude), "noise_sigma": float(sigma)}
    for key, passed in (("snr_likelihood", ratio >= k), ("snr_fidelity", fid >= fidelity_min)):
        meta[key] = float(snr["data"][passed].min()) if passed.any() else np.nan
    meta |= {f"snr_{name}": values.tolist() for name, values in snr.items()}
    meta["fidelity_bins"] = {key: values.tolist() for key, values in bins.items()}
    return columns, meta


def merge_catalogues(cats: Sequence[Table], reliability: dict | None = None) -> Table:
    """One catalogue of several searches (e.g. one per spectral window), its reliability over their whole volume.

    ``reliability`` (``floor``, ``k``, ``fidelity_min``, ``rule``) overrides the searches' own, so the
    tests can change without a new search.
    """
    given = {key: cats[0].meta[key] for key in ("floor", "k", "fidelity_min", "rule")} | (reliability or {})
    floor, k, fidelity_min, rule = (given[key] for key in ("floor", "k", "fidelity_min", "rule"))
    cat = vstack(list(cats), metadata_conflicts="silent").copy()  # vstack of one table returns it
    snr = {"data": np.asarray(cat["snr"])}
    snr |= {name: np.concatenate([c.meta[f"snr_{name}"] for c in cats]) for name in ("jackknife", "negative")}
    reliable, meta = _reliability(snr, floor, k, fidelity_min, rule)
    for name, values in reliable.items():
        cat[name] = values
    cat.meta = meta | {"parts": [{key: c.meta[key] for key in SEARCH_META} for c in cats]}
    cat.sort("snr", reverse=True)
    cat["id"] = np.arange(1, len(cat) + 1)
    return cat
