from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from astropy.table import Table, vstack
from scipy import fft, ndimage, optimize, stats

from .data import sky_direction
from .matchedfilter import TEMPLATE_KEYS, SearchResult, continuum
from .model import C_KMS, LINE_FLUX
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
    from scipy.spatial import cKDTree

    i, j, t, k, value = _maxima(snr, floor)
    x, y = np.asarray(result.axes["dra"])[i], np.asarray(result.axes["ddec"])[j]
    line = result.freqs.size > 1  # else a continuum: one channel
    chan = result.freqs[k] / _channel_width(result.freqs) if line else np.zeros(i.size)
    pa = np.radians(nc.pa)
    # every ellipsoid lies within sqrt(2) of its centre in units of the widest one
    scale = np.array([np.max(nc.hwhm_major)] * 2 + [np.max(nc.spectral_hwhm) if line else 1.0])
    tree = cKDTree(np.c_[x, y, chan] / scale)
    owner = np.full(i.size, -1)
    found = []
    for r in range(i.size):
        if owner[r] >= 0:
            continue
        near = np.asarray(tree.query_ball_point(np.r_[x[r], y[r], chan[r]] / scale, np.sqrt(2) * (1 + 1e-9)))
        free, tr = np.sort(near[owner[near] < 0]), t[r]
        dx, dy, dc = x[free] - x[r], y[free] - y[r], chan[free] - chan[r]
        major = dx * np.sin(pa[tr]) + dy * np.cos(pa[tr])
        minor = dx * np.cos(pa[tr]) - dy * np.sin(pa[tr])
        spectral = (dc / nc.spectral_hwhm[tr]) ** 2 if line else 0.0
        r2 = (major / nc.hwhm_major[tr]) ** 2 + (minor / nc.hwhm_minor[tr]) ** 2 + spectral
        inside = free[r2 <= 1]
        owner[inside] = r
        found.append(Group(int(i[r]), int(j[r]), int(tr), int(k[r]), float(value[r]), inside.size))
    return found


#: How the likelihood ratio and the fidelity combine into ``detected``: "and" (both) or "or" (either).
DETECTION_RULE = "and"
#: Default reliability tests: the S/N floor of the groups, the likelihood ratio ``k``, the fidelity and the rule.
RELIABILITY = {"floor": 4.0, "k": 3.0, "fidelity_min": 0.6, "rule": DETECTION_RULE}
#: Lowest S/N of the noise peaks the noise model is fitted to (the data's groups start at ``floor``).
NOISE_FLOOR = 2.5
#: Fewest noise peaks above :data:`NOISE_FLOOR` to fit the model to; fewer, and nothing is detected.
MIN_NOISE = 10
#: Width of the Gaussian prior on the exponent ``m``, about the field's dimension minus one.
EXPONENT_PRIOR = 1.0


def peak_exponent(snr_noise, dim: int, floor: float = NOISE_FLOOR) -> float:
    """Exponent ``m`` of the noise peak counts ``N(>= u) ~ u^m exp(-u^2 / 2)`` above ``floor``, by unbinned MAP.

    The expected number of peaks of a smooth Gaussian field of dimension ``dim`` above ``u`` (its Euler
    characteristic, Vio & Andreani 2021) is a polynomial in ``u`` times ``exp(-u^2 / 2)``, of degree
    ``dim - 1``: the prior's centre, :data:`EXPONENT_PRIOR` wide, so a few noise peaks (a continuum)
    cannot drive ``m`` far off and many (a line search) set it. NaN, with a warning, below
    :data:`MIN_NOISE` peaks.
    """
    u = np.asarray(snr_noise, dtype=float)
    u = u[u >= floor]
    if u.size < MIN_NOISE:
        warnings.warn(f"only {u.size} noise peaks above S/N {floor:g}, too few to fit the noise", stacklevel=2)
        return np.nan
    x = np.log(u / floor)

    def nll(m):  # density -dS/du of the survival S(u) = (u / floor)^m exp(-(u^2 - floor^2) / 2)
        return -np.sum(m * x + np.log(u - m / u)) + ((m - dim + 1) / EXPONENT_PRIOR) ** 2 / 2

    return float(optimize.minimize_scalar(nll, bounds=(-5.0, floor**2 * (1 - 1e-6)), method="bounded").x)


def noise_above(u, n: int, m: float, floor: float = NOISE_FLOOR) -> np.ndarray:
    """Expected noise peaks at or above ``u`` of a kernel with ``n`` noise peaks above ``floor``."""
    u = np.asarray(u, dtype=float)
    return n * np.exp(m * np.log(u / floor) - (u**2 - floor**2) / 2)


def likelihood(snr_data, noise) -> np.ndarray:
    """``N_data(>= gamma) / N_noise(>= gamma)`` at every data S/N (Vio & Andreani 2021); ``noise(u)`` the
    expected noise peaks at or above ``u`` (:func:`noise_above`), infinite where it is zero."""
    snr_data = np.asarray(snr_data, dtype=float)
    n_data = snr_data.size - np.searchsorted(np.sort(snr_data), snr_data)
    with np.errstate(divide="ignore"):
        return n_data / noise(snr_data)


def fidelity(snr_data, snr_noise, noise, floor: float, step: float = 0.25) -> tuple[np.ndarray, dict]:
    """Fidelity ``1 - N_noise / N_data`` per S/N bin from ``floor``, ``N_noise`` from the noise model ``noise``.

    Every data group gets its bin's fidelity. The bins hold the counts of the data and of the noise
    groups ``snr_noise``, their binned fidelity with its error (for the eye) and the ``model`` fidelity.
    """
    snr_data, snr_noise = np.asarray(snr_data, dtype=float), np.asarray(snr_noise, dtype=float)
    top = max(snr_data.max(initial=floor), snr_noise.max(initial=floor))
    edges = floor + step * np.arange(int((top - floor) / step) + 2)
    n_data, n_noise = (np.histogram(x, edges)[0] for x in (snr_data, snr_noise))
    expected = -np.diff(noise(edges))
    model = np.clip(1 - expected / np.maximum(n_data, 1), 0, 1)
    use = n_data > 0
    d, n = n_data[use], n_noise[use]
    bins = {
        "snr": edges[:-1][use] + step / 2,
        "n_data": d,
        "n_noise": n,
        "fidelity": np.clip(1 - n / d, 0, 1),
        "error": np.sqrt(np.maximum(n, 1) + n**2 / d) / d,
        "model": model[use],
    }
    return model[np.clip(np.searchsorted(edges, snr_data, side="right") - 1, 0, model.size - 1)], bins


#: S/N from which a line's sidelobes (``sidelobe`` of a few per cent) reach a few sigma elsewhere in the search:
#: flagged by :func:`bright_lines`, not subtracted.
BRIGHT_SNR = 100.0

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
    floor: float = RELIABILITY["floor"],
    k: float = RELIABILITY["k"],
    fidelity_min: float = RELIABILITY["fidelity_min"],
    rule: str = RELIABILITY["rule"],
    band: tuple[float, float] | None = None,
) -> Table:
    """Catalogue the line candidates of a search: one row per group of the data, by S/N descending.

    The noise correlation of ``result`` is measured, and the data, the jackknife and the negated
    data are grouped (:func:`groups`) on the filter's S/N (see the module docstring), the data from
    ``floor``, the jackknife and the negated data from :data:`NOISE_FLOOR`. Each data group gets the
    likelihood ratio and the fidelity against the noise peak counts modelled on the jackknife's groups
    of its own kernel (:func:`peak_exponent`, :func:`likelihood`, :func:`fidelity`). It is ``detected``
    if ``Lambda >= k`` and its fidelity is at least ``fidelity_min`` (``rule="and"``, the default), or
    either (``rule="or"``); without a noise model (fewer than :data:`MIN_NOISE` noise peaks) nothing
    is. Nothing is clipped: the candidates are ``cat[cat["detected"]]``.

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
    band : (low, high) Hz, optional
        Keep only the groups (of the data, the jackknife and the negated data) that peak in
        ``low <= nu < high``: the chunk's own part of a spectrum searched in overlapping chunks.

    Returns
    -------
    astropy.table.Table
        Columns ``id``, ``dra``, ``ddec`` (arcsec), ``ra``, ``dec`` (deg, with ``ref``),
        ``freq_ghz``, the template's ``bmin``, ``bmaj``, ``pa`` and ``width``, the ``snr``,
        ``flux`` and ``error`` (Jy) and ``line_flux``, ``line_flux_error`` (Jy km/s) at the
        brightest maximum, ``coverage``, ``n_peaks``, ``likelihood``, ``fidelity`` and
        ``detected``. ``meta`` holds the ``jackknife_spread``, the noise
        correlation's ``hwhm_major``, ``hwhm_minor``, ``pa``, ``spectral_hwhm`` and ``sidelobe``,
        the thresholds, the noise model (``noise_floor``, ``noise_exponent``, ``kernels``), the
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
        name: groups(values, result, nc, low)
        for name, values, low in (
            ("data", result.snr, floor),
            ("jackknife", result.jackknife.snr, NOISE_FLOOR),
            ("negative", -result.snr, NOISE_FLOOR),
        )
    }
    if band is not None:
        found = {name: [g for g in gs if band[0] <= result.freqs[g.k] < band[1]] for name, gs in found.items()}
    snr = {name: np.array([g.snr for g in gs]) for name, gs in found.items()}
    kernels = {name: result.templates[np.array([g.t for g in gs], dtype=int)] for name, gs in found.items()}
    i, j, t, c = (np.array([getattr(g, a) for g in found["data"]], dtype=int) for a in "ijtk")
    templates = result.templates[t]
    columns = {"id": np.arange(1, i.size + 1), "dra": np.asarray(result.axes["dra"])[i]}
    columns["ddec"] = np.asarray(result.axes["ddec"])[j]
    if ref is not None:
        columns["ra"], columns["dec"] = np.degrees(sky_direction((columns["dra"], columns["ddec"]), ref))
    columns |= {"freq_ghz": result.freqs[c] / 1e9, **dict(zip(TEMPLATE_KEYS, templates.T, strict=True))}
    peak, error = result.snr[i, j, t, c], result.error[i, j, t, c]
    per_width = templates[:, TEMPLATE_KEYS.index("width")] * LINE_FLUX  # line flux per peak flux density
    columns |= {"snr": peak, "flux": peak * error, "error": error}
    columns |= {"line_flux": peak * error * per_width, "line_flux_error": error * per_width}
    columns["coverage"] = result.coverage[i, j, c]
    columns["n_peaks"] = np.array([g.n_peaks for g in found["data"]], dtype=int)
    dim = 3 if result.freqs.size > 1 else 2  # position and frequency, or a continuum's position
    reliable, meta = _reliability(snr, kernels, dim, floor, k, fidelity_min, rule)
    search = {"jackknife_spread": jackknife_spread(result)} | {key: getattr(nc, key) for key in SEARCH_META[1:]}
    cat = Table(columns | reliable, meta={key: value.tolist() for key, value in search.items()} | meta)
    for name, unit in UNITS.items():
        if name in cat.colnames:
            cat[name].unit = unit
    if note := bright_lines(cat):
        warnings.warn(note, stacklevel=2)
    return cat


def continuum_catalogue(windows: SearchResult, ref=None, **reliability) -> Table:
    """Catalogue the continuum sources of a search's continuum per window (``SearchResult.continuum``).

    Grouped and tested as :func:`catalogue` on the windows combined (:func:`~luv_finder.matchedfilter.continuum`,
    ``freq_ghz`` their mean frequency), each source has its in-band spectrum, the flux density and error of every
    window at its position and template (``meta["spectra"]``), and the spectral index ``alpha`` (``S ~ nu^alpha``)
    fitted to it, linearised about ``freq_ghz``.
    """
    total = continuum(windows)
    cat = catalogue(total, ref=ref, **reliability)
    cat.remove_columns(["width", "line_flux", "line_flux_error"])
    i, j = (
        np.abs(np.asarray(windows.axes[k])[None] - np.asarray(cat[k])[:, None]).argmin(axis=1) for k in ("dra", "ddec")
    )
    shapes = np.c_[cat["bmin"], cat["bmaj"], cat["pa"]].astype(float)
    t = np.abs(windows.templates[None, :, :3] - shapes[:, None]).sum(axis=2).argmin(axis=1)
    flux, error = windows.flux[i, j, t], windows.error[i, j, t]
    alpha = np.full((2, len(cat)), np.nan)
    for n, (f, e) in enumerate(zip(flux, error, strict=True)):
        ok = np.isfinite(f) & np.isfinite(e)
        x, w = np.log(windows.freqs[ok] / total.freqs[0]), e[ok] ** -2.0
        if x.size > 1 and np.ptp(x) > 0:
            a = np.linalg.inv([[w.sum(), w @ x], [w @ x, w @ x**2]])  # covariance of (S0, S0 alpha)
            s0, b = a @ [w @ f[ok], w @ (x * f[ok])]
            alpha[:, n] = b / s0, np.sqrt(a[1, 1] - 2 * b / s0 * a[0, 1] + (b / s0) ** 2 * a[0, 0]) / abs(s0)
    cat["alpha"], cat["alpha_error"] = alpha
    cat.meta["spectra"] = {
        str(k): {"freq_ghz": (windows.freqs / 1e9).tolist(), "flux": f.tolist(), "error": e.tolist()}
        for k, f, e in zip(cat["id"], flux, error, strict=True)
    }
    return cat


def bright_lines(cat: Table) -> str:
    """A note on the groups at S/N >= :data:`BRIGHT_SNR`, empty if none."""
    bright = cat[cat["snr"] >= BRIGHT_SNR]
    if not len(bright):
        return ""
    found = ", ".join(f"#{row['id']} (S/N {row['snr']:.0f}, {row['freq_ghz']:.3f} GHz)" for row in bright)
    return (
        f"S/N >= {BRIGHT_SNR:g}: {found}; its sidelobes can pass the tests elsewhere, "
        "so check the groups around it (or subtract it and search the residual)"
    )


def passes(ratio, fid, k: float, fidelity_min: float) -> tuple[np.ndarray, np.ndarray]:
    """Which groups pass the likelihood ratio and which the fidelity (NaN passes neither)."""
    return np.asarray(ratio) >= k, np.asarray(fid) >= fidelity_min


def _reliability(
    snr: dict, kernels: dict, dim: int, floor: float, k: float, fidelity_min: float, rule: str = DETECTION_RULE
) -> tuple[dict, dict]:
    """Likelihood ratio, fidelity and detection of the data groups against the noise model, as columns and meta.

    The noise is the jackknife's peaks above :data:`NOISE_FLOOR`: one exponent for the whole search of
    dimension ``dim`` (:func:`peak_exponent`) and, per kernel (``kernels``: each group's template, a row of
    :data:`TEMPLATE_KEYS`, as ``snr``), the number of its peaks, since kernels of different size make
    different numbers of independent trials (González-López et al. 2019). Each data group is tested
    against its own kernel's model. ``meta["kernels"]`` has each kernel's counts and thresholds.
    """
    if rule not in ("or", "and"):
        raise ValueError(f'rule is "or" or "and", not {rule!r}')
    data, jackknife = snr["data"], snr["jackknife"]
    m = peak_exponent(jackknife, dim)
    ratio, fid = np.full(data.size, np.nan), np.full(data.size, np.nan)
    per_kernel = []
    for kernel in np.unique(np.r_[kernels["data"], kernels["jackknife"]], axis=0):
        d = np.all(kernels["data"] == kernel, axis=1)
        n = int(np.sum(np.all(kernels["jackknife"] == kernel, axis=1)))

        def noise(u, n=n):
            return noise_above(u, n, m)

        ratio[d] = likelihood(data[d], noise)
        fid[d] = fidelity(data[d], [], noise, floor)[0]
        lr_k, fid_k = passes(ratio[d], fid[d], k, fidelity_min)
        per_kernel.append(
            {"template": kernel.tolist(), "n_data": int(d.sum()), "n_jackknife": n,
             "snr_likelihood": float(data[d][lr_k].min()) if lr_k.any() else np.nan,
             "snr_fidelity": float(data[d][fid_k].min()) if fid_k.any() else np.nan}
        )  # fmt: skip
    n_all = [kn["n_jackknife"] for kn in per_kernel]
    bins = fidelity(data, jackknife[jackknife >= floor], lambda u: noise_above(u, sum(n_all), m), floor)[1]
    lr, fid_ok = passes(ratio, fid, k, fidelity_min)
    detected = lr | fid_ok if rule == "or" else lr & fid_ok
    columns = {"likelihood": ratio, "fidelity": fid, "detected": detected}
    meta = {"floor": float(floor), "k": float(k), "fidelity_min": float(fidelity_min), "rule": rule}
    meta |= {"noise_floor": NOISE_FLOOR, "noise_dim": int(dim), "noise_exponent": m}
    for key, passed in (("snr_likelihood", lr), ("snr_fidelity", fid_ok)):
        meta[key] = float(data[passed].min()) if passed.any() else np.nan
    meta |= {f"snr_{name}": values.tolist() for name, values in snr.items()}
    meta |= {f"template_{name}": np.asarray(values).tolist() for name, values in kernels.items() if name != "data"}
    meta["fidelity_bins"] = {key: values.tolist() for key, values in bins.items()}
    meta["kernels"] = per_kernel
    return columns, meta


def merge_catalogues(cats: Sequence[Table], reliability: dict | None = None, beam: dict | None = None) -> Table:
    """One catalogue of several searches (e.g. one per spectral window), its reliability over their whole volume.

    ``reliability`` (``floor``, ``k``, ``fidelity_min``, ``rule``) overrides the searches' own, so the
    tests can change without a new search. With ``beam`` (arcsec per ``unit``), a line found by two
    searches (:func:`duplicates`) counts once: the fainter group gets ``duplicate_of`` and is not detected.
    """
    given = {key: cats[0].meta[key] for key in RELIABILITY} | (reliability or {})
    cat = vstack(list(cats), metadata_conflicts="silent").copy()  # vstack of one table returns it
    order = np.argsort(-np.asarray(cat["snr"]), kind="stable")
    snr = {"data": np.asarray(cat["snr"])[order]}
    snr |= {name: np.concatenate([c.meta[f"snr_{name}"] for c in cats]) for name in ("jackknife", "negative")}
    kernels = {
        name: np.concatenate([np.reshape(c.meta[f"template_{name}"], (-1, len(TEMPLATE_KEYS))) for c in cats])
        for name in ("jackknife", "negative")
    }
    cat = cat[order]
    cat["id"] = np.arange(1, len(cat) + 1)
    one = np.ones(len(cat), dtype=bool)
    if beam is not None:
        cat["duplicate_of"] = duplicates(cat, beam)
        one = np.asarray(cat["duplicate_of"] == 0)
    kernels["data"] = np.c_[tuple(np.asarray(cat[key], dtype=float) for key in TEMPLATE_KEYS)][one]
    data = {**snr, "data": snr["data"][one]}
    reliable, meta = _reliability(data, kernels, cats[0].meta["noise_dim"], *(given[key] for key in RELIABILITY))
    for name, values in reliable.items():
        cat[name] = np.full(len(cat), False if values.dtype == bool else np.nan, dtype=values.dtype)
        cat[name][one] = values
    cat.meta = meta | {"parts": [{key: c.meta[key] for key in SEARCH_META} for c in cats]}
    return cat


def duplicates(cat: Table, beam: dict) -> np.ndarray:
    """Id of a brighter group of the same line in another unit, else 0, for every row of ``cat``.

    Two groups are one line when they are within ``beam[unit]`` (arcsec) of each other and their
    templates overlap, ``|dv| <= (FWHM_a + FWHM_b) / 2``: a line wider than the gap it straddles between
    two windows is found once in each, its halves far apart in frequency. ``cat`` is sorted by S/N.
    """
    from scipy.spatial import cKDTree

    dra, ddec, nu, width = (np.asarray(cat[k], dtype=float) for k in ("dra", "ddec", "freq_ghz", "width"))
    unit = np.asarray(cat["unit"])
    pairs = cKDTree(np.c_[dra, ddec]).query_pairs(max(beam.values()), output_type="ndarray")
    pairs.sort(axis=1)  # a brighter than b
    a, b = pairs.T if pairs.size else (np.zeros(0, int), np.zeros(0, int))
    same = (
        (unit[a] != unit[b])
        & (np.hypot(dra[a] - dra[b], ddec[a] - ddec[b]) <= np.array([beam[u] for u in unit[a]]))
        & (np.abs(nu[b] / nu[a] - 1) * C_KMS <= (width[a] + width[b]) / 2)
    )
    of = np.zeros(len(cat), dtype=int)
    for i, j in sorted(zip(a[same].tolist(), b[same].tolist(), strict=True)):
        if not of[i] and not of[j]:
            of[j] = cat["id"][i]
    return of
