"""Least-squares fit of catalogued lines in the visibilities.

Detection stays a grid search (:mod:`~luv_finder.matchedfilter`, :mod:`~luv_finder.catalogue`); each
catalogued line is then refined by a deterministic fit of the package's source model: an elliptical
Gaussian on the sky (position ``dra``, ``ddec``, axes ``bmaj``, ``bmin`` as sigma in arcsec, ``pa``)
whose spectrum is a Gaussian line on a polynomial continuum of the same shape,

    ``F(nu) = A exp(-(nu - nu0)^2 / (2 s^2)) + P(nu)``,

``s`` the sigma of a FWHM ``width`` in km/s at ``nu0`` and ``P`` of degree ``continuum_order`` in the
channel index scaled to [-1, 1] over the line's window, as in the search. In pointing p the model
visibility is ``PB_p(nu) F(nu) E(u, v) e^{+i phi}`` (:class:`~luv_finder.model.Gaussian`), ``PB_p``
the primary beam at the source, so ``A`` and ``P`` are intrinsic flux densities.

The fit is exact in the visibilities without visiting them one by one. Over every visibility of the
line's window in every pointing that covers it,

    ``chi^2 = sum w |V - model|^2 = const - 2 sum_nu F S + sum_nu F^2 Q``,

with ``S = sum_p PB_p sig_p`` and ``Q = sum_p PB_p^2 W_p`` from the template-weighted spectra at the
trial position and shape (:func:`~luv_finder.matchedfilter.template_spectrum`). So at a fixed
position and shape the spectrum is a weighted least-squares fit to ``S / Q`` with weights ``Q``,
linear in ``A`` and ``P`` and non-linear in (``nu0``, ``width``) only (variable projection); the
position and shape are optimised on that profiled chi^2 by Nelder-Mead, one collapse per covering
pointing per step.

Errors are the inverse Fisher matrix, half the numerical Hessian of chi^2 in every free parameter at
the optimum, with the visibility weights as the noise. Neighbouring channels share noise (ALMA's
Hanning response correlates them by 2/3 and 1/6), which the chi^2 treats as independent: the fit is
still unbiased, but its errors would be too small. They are corrected by the sandwich estimator
``F^-1 (J^T R J) F^-1`` over the channels, ``J`` the whitened derivatives of the spectrum model and
``R`` the channel correlation measured on the jackknife
(:func:`~luv_finder.data.channel_correlation`): a factor per spectral parameter, and one for the
spatial parameters, whose information follows the model spectrum.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from astropy.table import Table
from scipy import linalg, optimize

from .data import Chunk, DataHandler, load, sky_direction
from .data import channel_correlation as measure_correlation
from .matchedfilter import limit_cores, on_device, template_spectrum
from .model import C_KMS, FWHM_TO_SIGMA
from .utils import primary_beam

#: Spatial parameters: arcsec, the axes as sigma, and deg east of north.
SPATIAL = ("dra", "ddec", "bmaj", "bmin", "pa")

#: Line flux per peak flux density and FWHM, as :meth:`~luv_finder.matchedfilter.SearchResult.integrated_flux`.
LINE_FLUX = np.sqrt(2 * np.pi) * FWHM_TO_SIGMA

#: Columns :func:`fit_lines` adds, with their units.
COLUMNS = {
    "fit_dra": "arcsec",
    "fit_dra_error": "arcsec",
    "fit_ddec": "arcsec",
    "fit_ddec_error": "arcsec",
    "fit_ra": "deg",
    "fit_dec": "deg",
    "fit_bmaj": "arcsec",
    "fit_bmaj_error": "arcsec",
    "fit_bmin": "arcsec",
    "fit_bmin_error": "arcsec",
    "fit_pa": "deg",
    "fit_pa_error": "deg",
    "fit_freq_ghz": "GHz",
    "fit_freq_ghz_error": "GHz",
    "fit_width": "km/s",
    "fit_width_error": "km/s",
    "fit_peak": "Jy",
    "fit_peak_error": "Jy",
    "fit_line_flux": "Jy km/s",
    "fit_line_flux_error": "Jy km/s",
    "fit_continuum": "Jy",
    "fit_continuum_error": "Jy",
    "fit_chi2_reduced": None,
    "fit_n_pointings": None,
    "fit_point": None,
    "fit_converged": None,
}

#: Nelder-Mead tolerances: on the parameters in units of its first steps (a quarter of the
#: resolution, 30 deg for ``pa``), and on chi^2.
XATOL, FATOL = 1e-3, 1e-4


class LineWindow:
    """The window holding one line in every pointing that covers it, and the chi^2 of the line model.

    Parameters
    ----------
    chunks : list of Chunk
        The same spectral window of each covering pointing.
    dish_diameter : float
        m, for the primary beam.
    continuum_order : int or None
        Degree of the continuum polynomial; None fits none.
    rho : array, optional
        Channel correlation from lag 0 (see :func:`~luv_finder.data.channel_correlation`); None
        for independent channels.
    """

    def __init__(
        self, chunks: list[Chunk], dish_diameter: float, continuum_order: int | None = 2, rho: np.ndarray | None = None
    ):
        self.rho = None if rho is None or len(rho) == 1 else np.asarray(rho, dtype=float)
        self.freq = np.asarray(chunks[0].freq, dtype=float)
        if not all(np.allclose(c.freq, self.freq, rtol=1e-9, atol=0) for c in chunks[1:]):
            raise ValueError("the pointings' channel frequencies differ; only one spectral setup can be fitted")
        self.offsets = np.array([c.offset for c in chunks], dtype=float)
        self.chunks = [on_device(c) for c in chunks]
        self.dish_diameter = dish_diameter
        n_basis = 0 if continuum_order is None else continuum_order + 1
        self.basis = np.linspace(-1.0, 1.0, self.freq.size)[:, None] ** np.arange(n_basis)
        self._statistics: dict[tuple, tuple] = {}

    def statistics(self, spatial) -> tuple[np.ndarray, np.ndarray]:
        """``S`` and ``Q``, ``(n_chan,)`` each, at ``spatial`` = (dra, ddec, bmaj, bmin, pa); cached."""
        key = tuple(float(p) for p in spatial)
        if key not in self._statistics:
            with ThreadPoolExecutor(len(self.chunks)) as pool:
                spectra = list(pool.map(lambda c: template_spectrum(c, *key), self.chunks))
            sig, weight = (np.array(a) for a in zip(*spectra, strict=True))
            distance = np.hypot(key[0] - self.offsets[:, :1], key[1] - self.offsets[:, 1:])
            pb = primary_beam(distance, self.freq, self.dish_diameter)
            self._statistics[key] = (pb * sig).sum(axis=0), (pb**2 * weight).sum(axis=0)
        return self._statistics[key]

    def design(self, nu0: float, width: float) -> np.ndarray:
        """Columns of ``F``: the unit-peak line at ``nu0`` (Hz) of FWHM ``width`` (km/s) and the continuum basis."""
        sigma = nu0 * width * FWHM_TO_SIGMA / C_KMS
        return np.column_stack([np.exp(-0.5 * ((self.freq - nu0) / sigma) ** 2), self.basis])

    def chi2(self, spectral, spatial) -> float:
        """chi^2 less the data's own ``sum w |V|^2``; ``spectral`` is (A, c_0, ..., nu0, width)."""
        S, Q = self.statistics(spatial)
        F = self.design(*spectral[-2:]) @ spectral[:-2]
        return float(F**2 @ Q - 2 * F @ S)

    def profile(self, spatial, start, bounds) -> tuple[float, np.ndarray, optimize.OptimizeResult]:
        """The best spectrum at ``spatial``: its :meth:`chi2`, (A, c_0, ..., nu0, width) and the non-linear fit.

        ``start`` and ``bounds`` (a pair of arrays) set the search over (nu0, width); A and the
        continuum follow by linear least squares at every (nu0, width).
        """
        S, Q = self.statistics(spatial)
        ok = Q > 0
        root, d = np.sqrt(Q[ok]), S[ok] / Q[ok]

        def linear(theta):
            G = root[:, None] * self.design(*theta)[ok]
            coef = np.linalg.lstsq(G, root * d, rcond=None)[0]
            return coef, root * d - G @ coef

        scale = (self.freq[1] - self.freq[0], 0.1 * start[1])
        nonlinear = optimize.least_squares(lambda t: linear(t)[1], start, bounds=bounds, x_scale=scale, xtol=1e-12)
        coef, residual = linear(nonlinear.x)
        return float(residual @ residual - d**2 @ Q[ok]), np.concatenate([coef, nonlinear.x]), nonlinear


def _window_chunks(data: DataHandler, dra: float, ddec: float, nu: float, pb_limit: float) -> list[Chunk]:
    """The window holding ``nu`` Hz, where it lies farthest from an edge, of each pointing with PB >= ``pb_limit``.

    The primary beam is taken at (``dra``, ``ddec``) and ``nu``.
    """
    best: dict[int, tuple[float, Chunk]] = {}
    for c in data.chunks:
        margin = min(nu - c.freq[0], c.freq[-1] - nu)
        if margin >= 0 and margin > best.get(c.field, (-1.0, None))[0]:
            best[c.field] = (margin, c)
    dish = data.metadata.dish_diameter
    chunks = [
        c
        for _, c in best.values()
        if primary_beam(np.hypot(dra - c.offset[0], ddec - c.offset[1]), nu, dish) >= pb_limit
    ]
    if not chunks:
        raise ValueError(
            f"no pointing covers {nu / 1e9:.4f} GHz at ({dra:.2f}, {ddec:.2f}) arcsec with PB >= {pb_limit}"
        )
    return chunks


def _correlation_scale(window: LineWindow, spectral: np.ndarray, spatial, n_free: int) -> np.ndarray:
    """Factors on the errors of (A, c_0, ..., nu0, width, free spatial...) for the channel correlation.

    Sandwich estimator over the channels: ``sqrt(diag(F^-1 J^T R J F^-1) / diag(F^-1))`` for the
    spectral parameters, ``J`` the derivatives of the spectrum model whitened by ``sqrt(Q)``, and
    ``sqrt(a^T R a / a^T a)`` with ``a = sqrt(Q) F`` for the spatial ones. All ones without ``rho``.
    """
    n_spec = spectral.size
    if window.rho is None:
        return np.ones(n_spec + n_free)
    _, Q = window.statistics(spatial)
    ok = Q > 0
    peak, nu0, width = spectral[0], spectral[-2], spectral[-1]
    dnu, dw = 1e-3 * (window.freq[1] - window.freq[0]), 1e-3 * width
    G = window.design(nu0, width)
    line_nu0 = (window.design(nu0 + dnu, width)[:, 0] - window.design(nu0 - dnu, width)[:, 0]) / (2 * dnu)
    line_w = (window.design(nu0, width + dw)[:, 0] - window.design(nu0, width - dw)[:, 0]) / (2 * dw)
    root = np.sqrt(Q[ok])[:, None]
    J = root * np.column_stack([G, peak * line_nu0, peak * line_w])[ok]
    a = root[:, 0] * (G @ spectral[:-2])[ok]
    full = np.zeros(window.freq.size)
    full[: window.rho.size] = window.rho
    R = linalg.toeplitz(full)[np.ix_(ok, ok)]
    inverse = np.linalg.pinv(J.T @ J)
    corrected = inverse @ (J.T @ R @ J) @ inverse
    spectral_scale = np.sqrt(np.diag(corrected) / np.diag(inverse))
    return np.concatenate([spectral_scale, np.full(n_free, np.sqrt(a @ R @ a / (a @ a)))])


def _hessian(f, x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``x`` with steps ``h``."""
    n = x.size

    def at(*steps):
        y = x.copy()
        for i, s in steps:
            y[i] += s * h[i]
        return f(y)

    f0, H = f(x), np.empty((n, n))
    for i in range(n):
        H[i, i] = (at((i, 1)) - 2 * f0 + at((i, -1))) / h[i] ** 2
        for j in range(i):
            corners = at((i, 1), (j, 1)) - at((i, 1), (j, -1)) - at((i, -1), (j, 1)) + at((i, -1), (j, -1))
            H[i, j] = H[j, i] = corners / (4 * h[i] * h[j])
    return H


def _covariance(H: np.ndarray, droppable: np.ndarray) -> tuple[np.ndarray, bool]:
    """``2 H^-1`` if ``H`` is positive definite, else over all but ``droppable`` (NaN there) if that is.

    Returns the covariance and whether either was; NaN throughout if neither.
    """
    cov = np.full(H.shape, np.nan)
    for keep in (np.ones(len(H), dtype=bool), ~droppable):
        sub = H[np.ix_(keep, keep)]
        diag = np.diag(sub)
        if np.all(diag > 0) and np.linalg.eigvalsh(sub / np.sqrt(np.outer(diag, diag))).min() > 1e-10:
            cov[np.ix_(keep, keep)] = 2 * np.linalg.inv(sub)
            return cov, True
    return cov, False


@dataclass(frozen=True)
class _Fit:
    spectral: np.ndarray  # A, c_0, ..., nu0, width
    spatial: np.ndarray  # SPATIAL
    free: np.ndarray  # bool over SPATIAL
    cov: np.ndarray  # over the spectral, then the free spatial parameters
    chi2_reduced: float
    converged: bool
    at_bound: bool


def _fit(window: LineWindow, spatial, free, step, start, res: float, limits: dict) -> _Fit:
    """Nelder-Mead over the ``free`` spatial parameters on the profiled chi^2, then the Hessian at the optimum.

    ``spatial`` is the start (and the value of the fixed parameters) and ``step`` the first steps
    of the simplex, both over :data:`SPATIAL`; ``start`` (nu0 Hz, width km/s) starts the line,
    ``res`` is the resolution in arcsec and ``limits`` the largest ``offset`` from the start and
    ``size`` (sigma), arcsec.
    """
    dnu = window.freq[1] - window.freq[0]
    kms = C_KMS / start[0]
    bounds = (
        np.array([window.freq[0], dnu * kms]),
        np.array([window.freq[-1], (window.freq[-1] - window.freq[0]) * kms / 2]),
    )
    start = np.clip(start, *bounds)
    radius = np.array([limits["offset"], limits["offset"], limits["size"], limits["size"], np.inf])
    centre = np.array([*spatial[:2], 0.0, 0.0, 0.0])
    # axes below 1% of the resolution change the envelope by < 0.2% on the longest baseline: a point
    collapsed = 0.01 * res
    step = step[free]

    def place(values):
        p = spatial.copy()
        p[free] = values
        return p

    success = True
    if free.any():
        origin, n = spatial[free], free.sum()
        result = optimize.minimize(
            lambda z: window.profile(place(origin + z * step), start, bounds)[0],
            np.zeros(n),
            method="Nelder-Mead",
            bounds=optimize.Bounds(
                ((centre - radius)[free] - origin) / step, ((centre + radius)[free] - origin) / step
            ),
            options={"initial_simplex": np.vstack([np.zeros(n), np.eye(n)]), "xatol": XATOL, "fatol": FATOL},
        )
        spatial, success = place(origin + result.x * step), result.success
    _, best, nonlinear = window.profile(spatial, start, bounds)
    n_spec = best.size

    def chi2(theta):
        return window.chi2(theta[:n_spec], place(theta[n_spec:]))

    # Hessian steps of about one sigma: from the linear fit, the non-linear one and the diagonal
    S, Q = window.statistics(spatial)
    G = window.design(*best[-2:])
    h = np.concatenate(
        [
            np.sqrt(np.diag(np.linalg.pinv(G.T @ (Q[:, None] * G)))),
            np.minimum(np.sqrt(np.diag(np.linalg.pinv(nonlinear.jac.T @ nonlinear.jac))), [2 * dnu, best[-1] / 3]),
            np.array([res / 50, res / 50, res / 50, res / 50, 2.0])[free],
        ]
    )
    h[:n_spec] = np.where(h[:n_spec] > 0, h[:n_spec], [*np.ones(n_spec - 2), dnu / 10, best[-1] / 20])
    theta = np.concatenate([best, spatial[free]])
    h_max = np.array([res / 2, res / 2, res / 2, res / 2, 20.0])[free]
    f0 = chi2(theta)
    for i, k in enumerate(range(n_spec, theta.size)):
        e = np.zeros_like(theta)
        e[k] = h[k]
        curvature = (chi2(theta + e) - 2 * f0 + chi2(theta - e)) / h[k] ** 2
        h[k] = np.clip(np.sqrt(2 / curvature), h[k], h_max[i]) if curvature > 0 else h_max[i]
    pa = np.zeros(theta.size, dtype=bool)
    pa[n_spec:] = np.asarray(SPATIAL)[free] == "pa"
    cov, positive = _covariance(_hessian(chi2, theta, h), pa)
    scale = _correlation_scale(window, best, spatial, int(free.sum()))
    cov = cov * np.outer(scale, scale)

    ok = Q > 0
    chi2_spectrum = float(Q[ok] @ (S[ok] / Q[ok] - (G @ best[:-2])[ok]) ** 2)
    return _Fit(
        spectral=best,
        spatial=spatial,
        free=free,
        cov=cov,
        chi2_reduced=chi2_spectrum / max(int(ok.sum()) - n_spec, 1),
        converged=bool(success and nonlinear.success and positive),
        at_bound=bool(free[2:4].any() and not collapsed < np.abs(spatial[2:4]).max() < 0.99 * limits["size"]),
    )


def _columns(fit: _Fit, window: LineWindow, ref, point: bool) -> dict:
    """The :data:`COLUMNS` of one fit."""
    n_spec = fit.spectral.size
    cov = fit.cov
    error = np.full(len(SPATIAL), np.nan)
    error[fit.free] = np.sqrt(np.diag(cov)[n_spec:])
    (dra, ddec, bmaj, bmin, pa), (e_dra, e_ddec, e_bmaj, e_bmin, e_pa) = fit.spatial, error
    bmaj, bmin = abs(bmaj), abs(bmin)
    if bmin > bmaj:
        bmaj, bmin, e_bmaj, e_bmin, pa = bmin, bmaj, e_bmin, e_bmaj, pa + 90
    peak, coef, nu0, width = fit.spectral[0], fit.spectral[1:-2], fit.spectral[-2], fit.spectral[-1]

    def propagate(gradient):
        used = gradient != 0
        return float(np.sqrt(gradient[used] @ cov[np.ix_(used, used)] @ gradient[used]))

    flux_gradient = np.zeros(len(cov))
    flux_gradient[[0, n_spec - 1]] = width * LINE_FLUX, peak * LINE_FLUX
    span = window.freq[-1] - window.freq[0]
    t, powers = -1 + 2 * (nu0 - window.freq[0]) / span, np.arange(coef.size)
    continuum_gradient = np.zeros(len(cov))
    continuum_gradient[1 : n_spec - 2] = t**powers
    continuum_gradient[n_spec - 2] = coef[1:] @ (powers[1:] * t ** (powers[1:] - 1.0)) * 2 / span
    ra, dec = np.degrees(sky_direction((dra, ddec), ref))
    spectral_error = np.sqrt(np.diag(cov)[:n_spec])
    return {
        "fit_dra": dra,
        "fit_dra_error": e_dra,
        "fit_ddec": ddec,
        "fit_ddec_error": e_ddec,
        "fit_ra": ra,
        "fit_dec": dec,
        "fit_bmaj": bmaj,
        "fit_bmaj_error": e_bmaj,
        "fit_bmin": bmin,
        "fit_bmin_error": e_bmin,
        "fit_pa": np.nan if np.isnan(e_pa) else pa % 180,
        "fit_pa_error": e_pa,
        "fit_freq_ghz": nu0 / 1e9,
        "fit_freq_ghz_error": spectral_error[-2] / 1e9,
        "fit_width": width,
        "fit_width_error": spectral_error[-1],
        "fit_peak": peak,
        "fit_peak_error": spectral_error[0],
        "fit_line_flux": peak * width * LINE_FLUX,
        "fit_line_flux_error": propagate(flux_gradient),
        "fit_continuum": coef @ t**powers if coef.size else np.nan,
        "fit_continuum_error": propagate(continuum_gradient) if coef.size else np.nan,
        "fit_chi2_reduced": fit.chi2_reduced,
        "fit_n_pointings": len(window.chunks),
        "fit_point": point,
        "fit_converged": fit.converged,
    }


def fit_line(
    data: DataHandler,
    dra: float,
    ddec: float,
    freq_ghz: float,
    width: float,
    continuum_order: int | None = 2,
    point: bool = False,
    fixed_position: bool = False,
    pb_limit: float = 0.2,
    max_offset: float | None = None,
    size_max: float | None = None,
    channel_correlation: str | dict | None = "measure",
) -> dict:
    """Fit one line in the visibilities, starting from a catalogue position, frequency and width.

    The fit uses the window holding ``freq_ghz`` (of two overlapping windows, the one it lies
    deeper in) of every pointing whose primary beam at the start is at least ``pb_limit``. Unless
    ``point``, the source is an elliptical Gaussian, started round with axes of a quarter of the
    resolution (at most half ``size_max``); if that fit fails, does not give a positive-definite
    Hessian (``pa`` alone may be unconstrained, for a round source), runs an axis to ``size_max``
    or collapses both to zero (unresolved), it is refitted as a point source.

    Parameters
    ----------
    data : DataHandler
        Every pointing of the search, or at least those covering the line.
    dra, ddec : float
        Start position, arcsec in the sky frame.
    freq_ghz, width : float
        Start line centre (GHz) and FWHM (km/s). The FWHM is fitted between one channel and half
        the window.
    continuum_order : int or None
        Degree of the continuum polynomial over the window; None fits none.
    point : bool
        Fit a point source: position and spectrum only.
    fixed_position : bool
        Keep the position at (``dra``, ``ddec``).
    pb_limit : float
        Pointings whose primary beam at the start is below this are left out.
    max_offset : float, optional
        Largest distance of the fitted position from the start along either axis, arcsec; default
        the resolution.
    size_max : float, optional
        Largest axis (sigma), arcsec; default twice the resolution.
    channel_correlation : "measure", None or dict
        Noise correlation between channels for the errors: "measure" (default) on the jackknife
        of the line's window, None for independent channels, or ``{spw: (1, rho_1, ...)}``.

    Returns
    -------
    dict
        The :data:`COLUMNS`, see :func:`fit_lines`.
    """
    res = data.metadata.minresolution()
    limits = {"offset": res if max_offset is None else max_offset, "size": 2 * res if size_max is None else size_max}
    chunks = _window_chunks(data, dra, ddec, freq_ghz * 1e9, pb_limit)
    if channel_correlation == "measure":
        rho = measure_correlation(chunks)
    else:
        rho = (channel_correlation or {}).get(chunks[0].spw)
    window = LineWindow(chunks, data.metadata.dish_diameter, continuum_order, rho)
    start = np.array([freq_ghz * 1e9, width])
    size = min(res / 4, limits["size"] / 2)
    step = np.array([res / 4, res / 4, size, size, 30.0])
    moving = not fixed_position
    if not point:
        free = np.array([moving, moving, True, True, True])
        fit = _fit(window, np.array([dra, ddec, size, size, 0.0]), free, step, start, res, limits)
        point = not fit.converged or fit.at_bound
    if point:
        free = np.array([moving, moving, False, False, False])
        fit = _fit(window, np.array([dra, ddec, 0.0, 0.0, 0.0]), free, step, start, res, limits)
    return _columns(fit, window, data.metadata.ref, point)


def fit_lines(data: DataHandler | str, cat: Table, rows=None, cores: int | None = None, **kwargs) -> Table:
    """Fit catalogued lines in the visibilities (see the module docstring and :func:`fit_line`).

    Each line starts from its catalogue ``dra``, ``ddec``, ``freq_ghz`` and ``width``. A line whose
    fit raises is warned about and left unfitted; the others are not affected.

    Parameters
    ----------
    data : DataHandler or str
        The searched data, or a measurement set or NPZ to load whole.
    cat : astropy.table.Table
        From :func:`~luv_finder.catalogue.catalogue`.
    rows : array of bool, optional
        Lines to fit; default the ``detected`` ones (all if there is no such column).
    cores : int, optional
        See :func:`~luv_finder.matchedfilter.limit_cores`.
    **kwargs
        Passed to :func:`fit_line`.

    Returns
    -------
    astropy.table.Table
        A copy of ``cat`` with the :data:`COLUMNS`: the fitted position ``fit_dra``, ``fit_ddec``
        (arcsec) and ``fit_ra``, ``fit_dec`` (deg); the source axes ``fit_bmaj`` >= ``fit_bmin``
        (sigma, arcsec, deconvolved; 0 for a point source) and ``fit_pa`` (deg east of north, in
        [0, 180), NaN where unconstrained); the line centre ``fit_freq_ghz``, FWHM ``fit_width``
        (km/s), intrinsic peak flux density ``fit_peak`` (Jy) and line flux ``fit_line_flux`` (Jy
        km/s); the continuum at the line centre ``fit_continuum`` (Jy), each with an ``_error``;
        ``fit_chi2_reduced`` of the combined spectrum ``S / Q`` against the fitted line and
        continuum; ``fit_n_pointings`` fitted together; ``fit_point`` if the source was fitted as a
        point; ``fit_converged`` if the optimisers converged and the Hessian is positive definite.
        Errors are formal, corrected for the channel correlation (see the module docstring), which
        is measured once per spectral window unless ``channel_correlation`` is passed. Unfitted
        lines hold NaN, 0 and False.
    """
    limit_cores(cores)
    if not isinstance(data, DataHandler):
        data = load(data)
    if kwargs.get("channel_correlation", "measure") == "measure":
        kwargs["channel_correlation"] = data.channel_correlation()
    if rows is None:
        rows = cat["detected"] if "detected" in cat.colnames else np.ones(len(cat), dtype=bool)
    out = cat.copy()
    for name, unit in COLUMNS.items():
        dtype = bool if name in ("fit_point", "fit_converged") else int if name == "fit_n_pointings" else float
        out[name] = np.full(len(cat), np.nan if dtype is float else 0, dtype=dtype)
        out[name].unit = unit
    for i in np.flatnonzero(rows):
        start = (float(cat[k][i]) for k in ("dra", "ddec", "freq_ghz", "width"))
        try:
            values = fit_line(data, *start, **kwargs)
        except Exception as err:  # one line's failure must not cost the table
            warnings.warn(f"the fit of catalogue row {i} failed: {err}", stacklevel=2)
            continue
        for name, value in values.items():
            out[name][i] = value
    return out
