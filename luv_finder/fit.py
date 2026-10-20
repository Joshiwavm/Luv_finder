"""Least-squares fit of catalogued lines in the visibilities, by Newton steps on per-channel moments.

The model in pointing p is ``a_p(nu) E(u, v) e^{+i phi}``, ``a_p = PB_p F``: an elliptical Gaussian
of sky covariance ``(s_ee, s_nn, s_en)`` whose spectrum ``F`` is a Gaussian line on a polynomial
continuum, its own in every spectral window that holds the line. Every derivative of ``E e^{+i phi}``
in position and shape is itself times a polynomial
in (u, v), so one pass over the visibilities (:func:`~luv_finder.kernel.point_moments`) gives
chi^2, its gradient and the Gauss-Newton matrix ``J^T W J`` in every parameter exactly. Errors are
``2 (J^T W J)^-1``, corrected for the channel correlation by a sandwich estimator over the channels.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from scipy import linalg, optimize

from .data import Chunk, DataHandler, load, sky_direction
from .data import channel_noise as measure_noise
from .kernel import DATA_POWERS, WEIGHT_POWERS, limit_cores, on_device, point_moments
from .matchedfilter import MASK_FWHM, MASK_SNR
from .model import C_KMS, CONTINUUM_ORDER, FWHM_TO_SIGMA, covariance, rotated
from .utils import primary_beam

#: Line flux per peak flux density and FWHM, as :meth:`~luv_finder.matchedfilter.SearchResult.integrated_flux`.
LINE_FLUX = np.sqrt(2 * np.pi) * FWHM_TO_SIGMA

#: Quantities reported per line, each with an ``_error``, and their units.
REPORTED = {
    "dra": "arcsec",
    "ddec": "arcsec",
    "bmaj": "arcsec",
    "bmin": "arcsec",
    "pa": "deg",
    "freq_ghz": "GHz",
    "width": "km/s",
    "peak": "Jy",
    "line_flux": "Jy km/s",
    "continuum": "Jy",
}

#: Lowest fitted line FWHM, km/s: a line a few channels wide would otherwise collapse below one channel,
#: since the model samples a Gaussian at the channel centres (no Hanning or channel response).
MIN_WIDTH = 70.0

#: Columns :func:`fit_lines` adds, with their units.
COLUMNS = {
    **{f"fit_{k}{e}": u for k, u in REPORTED.items() for e in ("", "_error")},
    "fit_ra": "deg",
    "fit_dec": "deg",
    "fit_chi2_reduced": None,
    "fit_n_pointings": None,
    "fit_converged": None,
}

#: Catalogue columns a fit starts from; the source shape defaults to a point.
START = ("dra", "ddec", "freq_ghz", "width", "bmaj", "bmin", "pa")

_WEIGHT = {p: k for k, p in enumerate(WEIGHT_POWERS)}
#: Weight moments of the data powers, and of the products of two of them.
_SINGLE = np.array([_WEIGHT[p] for p in DATA_POWERS])
_PAIR = np.array([[_WEIGHT[(i + k, j + l)] for k, l in DATA_POWERS] for i, j in DATA_POWERS])
#: d ln(E e^{+i phi}) / d(dra, ddec, s_ee, s_nn, s_en) over the data powers, u and v in cycles per arcsec.
_GAMMA = np.zeros((5, len(DATA_POWERS)), dtype=complex)
for _row, _power, _value in (
    (0, (1, 0), 2j * np.pi),
    (1, (0, 1), 2j * np.pi),
    (2, (2, 0), -2 * np.pi**2),
    (3, (0, 2), -2 * np.pi**2),
    (4, (1, 1), -4 * np.pi**2),
):
    _GAMMA[_row, DATA_POWERS.index(_power)] = _value


def _spectrum(theta, freq, basis):
    """``F``; ``theta`` is dra, ddec (arcsec), s_ee, s_nn, s_en (arcsec^2), nu0 (Hz), width (km/s), peak, c (Jy)."""
    sigma = theta[5] * theta[6] * FWHM_TO_SIGMA / C_KMS
    return theta[7] * jnp.exp(-0.5 * ((freq - theta[5]) / sigma) ** 2) + basis @ theta[8:]


def _beam(theta, freq, offset, dish):
    return primary_beam(jnp.hypot(theta[0] - offset[0], theta[1] - offset[1]), freq, dish, xp=jnp)


def _amplitude(theta, freq, basis, offset, dish):
    """``a = PB F`` of one pointing per channel."""
    return _beam(theta, freq, offset, dish) * _spectrum(theta, freq, basis)


@jax.jit
def _normal(a, da, data, weight):
    """chi^2 (less ``sum w |V|^2``), its gradient and ``2 J^T W J`` of one pointing from its moments."""
    gamma = jnp.zeros((da.shape[1], len(DATA_POWERS)), dtype=complex).at[:5].set(_GAMMA)
    single, pair, total = weight[:, _SINGLE], weight[:, _PAIR], weight[:, 0]
    chi2 = jnp.sum(a**2 * total - 2 * a * data[:, 0].real)
    # sum w (V - M)^* dM and sum w M^* dM, then sum w dM^* dM
    cross = jnp.einsum("n,km,nm->k", a, gamma, data.conj()) + da.T @ data[:, 0].conj()
    model = jnp.einsum("n,km,nm->k", a**2, gamma, single) + da.T @ (a * total)
    left = jnp.einsum("km,nm->nk", gamma.conj(), single)
    right = jnp.einsum("km,nm->nk", gamma, single)
    jtj = (
        jnp.einsum("n,km,nmo,lo->kl", a**2, gamma.conj(), pair, gamma)
        + jnp.einsum("n,nk,nl->kl", a, left, da)
        + jnp.einsum("n,nk,nl->kl", a, da, right)
        + jnp.einsum("n,nk,nl->kl", total, da, da)
    )
    return chi2, 2 * (model - cross).real, 2 * jtj.real


def _moments(theta, chunks) -> list:
    """Every pointing's moments at the position and shape of ``theta``: the one pass over the visibilities."""
    return [point_moments(c, theta[0], theta[1], tuple(theta[2:5])) for c in chunks]


def _system(theta, moments, chunks, spectral, dish):
    """chi^2, gradient and Gauss-Newton matrix, and every window's ``S``, ``Q`` spectra, from ``moments``.

    ``spectral`` maps each chunk's window to its ``(freq, basis)``.
    """
    chi2, grad, hess = 0.0, 0.0, 0.0
    S, Q = {spw: 0.0 for spw in spectral}, {spw: 0.0 for spw in spectral}
    for c, (data, weight) in zip(chunks, moments, strict=True):
        freq, basis = spectral[c.spw]
        args = (theta, freq, basis, c.offset, dish)
        a, da = _amplitude(*args), jax.jacfwd(_amplitude)(*args)
        pb = _beam(theta, freq, c.offset, dish)
        terms = _normal(a, da, data, weight)
        chi2, grad, hess = chi2 + terms[0], grad + terms[1], hess + terms[2]
        S[c.spw], Q[c.spw] = S[c.spw] + pb * data[:, 0].real, Q[c.spw] + pb**2 * weight[:, 0]
    return chi2, grad, hess, S, Q


def _reported(theta, freq, n_continuum, block):
    """The :data:`REPORTED` quantities; the axes (sigma) from the covariance's eigenvalues, 0 where unresolved.

    The continuum is the one of the window ``freq`` at the line, its coefficients ``block`` onward.
    """
    s_ee, s_nn, s_en = theta[2:5]
    half, split = (s_ee + s_nn) / 2, jnp.hypot((s_ee - s_nn) / 2, s_en)
    t = 2 * (theta[5] - freq[0]) / (freq[-1] - freq[0]) - 1
    major, minor = half + split, half - split
    return jnp.stack(
        [
            theta[0],
            theta[1],
            jnp.sqrt(jnp.maximum(major, 0.0)),
            jnp.sqrt(jnp.maximum(minor, 0.0)),
            jnp.degrees(0.5 * jnp.arctan2(2 * s_en, s_nn - s_ee)) % 180,
            theta[5] / 1e9,
            theta[6],
            theta[7],
            theta[7] * theta[6] * LINE_FLUX,
            t ** jnp.arange(n_continuum) @ theta[8 + block : 8 + block + n_continuum],
        ]
    )


def _correlation_scale(theta, windows) -> np.ndarray:
    """Factors on the errors of ``theta`` for the channel correlation: a sandwich over the channels.

    ``windows``: ``(Q, freq, basis, rho)`` of every window; windows are independent of each other.
    """
    if all(rho is None or len(rho) == 1 for *_, rho in windows):
        return np.ones(theta.size)
    rows, model, blocks = [], [], []
    for Q, freq, basis, rho in windows:
        ok, rho = Q > 0, np.ones(1) if rho is None else rho
        root = np.sqrt(Q[ok])
        rows.append(root[:, None] * np.asarray(jax.jacfwd(_spectrum)(theta, freq, basis))[ok, 5:])
        model.append(root * np.asarray(_spectrum(theta, freq, basis))[ok])
        blocks.append(linalg.toeplitz(np.pad(rho, (0, freq.size - len(rho))))[np.ix_(ok, ok)])
    J, a, R = np.vstack(rows), np.concatenate(model), linalg.block_diag(*blocks)
    inverse = np.linalg.pinv(J.T @ J)
    spectral = np.sqrt(np.diag(inverse @ J.T @ R @ J @ inverse) / np.diag(inverse))
    return np.concatenate([np.full(5, np.sqrt(a @ R @ a / (a @ a))), spectral])


def _window_chunks(data: DataHandler, dra: float, ddec: float, nu: float, width: float, pb_limit: float) -> list[Chunk]:
    """Every window within one FWHM (``width`` km/s) of ``nu`` Hz, of each pointing with PB >= ``pb_limit``."""
    dish = data.metadata.dish_diameter
    chunks = [
        c
        for c in data.chunks
        if np.any(np.abs(c.freq / nu - 1) * C_KMS <= width)
        and primary_beam(np.hypot(dra - c.offset[0], ddec - c.offset[1]), nu, dish) >= pb_limit
    ]
    if not chunks:
        raise ValueError(
            f"no pointing covers {nu / 1e9:.4f} GHz at ({dra:.2f}, {ddec:.2f}) arcsec with PB >= {pb_limit}"
        )
    for c in chunks:
        if not np.allclose(c.freq, next(o.freq for o in chunks if o.spw == c.spw), rtol=1e-9, atol=0):
            raise ValueError(f"the pointings' channel frequencies differ in spw {c.spw}")
    return chunks


def home_window(chunks: list[Chunk], nu: float) -> int:
    """The spw among ``chunks`` that holds ``nu`` Hz farthest from its edges (the nearest, if none holds it)."""
    return max(chunks, key=lambda c: min(nu - c.freq[0], c.freq[-1] - nu)).spw


def _without(chunk: Chunk, nu: float, width: float, others) -> Chunk:
    """``chunk`` with the channels of the ``others`` lines (GHz, km/s) given zero weight, as the filter masks them.

    Each other line masks ``MASK_FWHM`` of its FWHM; the fitted line (``nu`` Hz, ``width``) keeps its own.
    """
    v = np.abs(chunk.freq / nu - 1) * C_KMS
    drop = np.zeros(chunk.freq.size, dtype=bool)
    for nu_o, width_o in others:
        drop |= np.abs(chunk.freq / (nu_o * 1e9) - 1) * C_KMS <= MASK_FWHM * width_o
    drop &= v > MASK_FWHM * width
    if not drop.any():
        return chunk
    keep = ~drop[:, None]
    return dataclasses.replace(chunk, X=chunk.X * keep, w=(chunk.w * keep).astype(chunk.w.dtype))


def fit_line(
    data: DataHandler,
    dra: float,
    ddec: float,
    freq_ghz: float,
    width: float,
    bmaj: float = 0.0,
    bmin: float = 0.0,
    pa: float = 0.0,
    continuum_order: int | None = CONTINUUM_ORDER,
    pb_limit: float = 0.2,
    channel_noise: str | dict | None = "measure",
    others=(),
) -> dict:
    """Fit one line, started from the matched filter's position, template shape (sigma, arcsec) and width.

    Every window within one FWHM of the line is fitted (:func:`_window_chunks`): the line is shared, and
    each window has its own continuum. ``others``: (GHz, km/s) of other lines at the position, left out
    of the fit (:func:`_without`).
    """
    nu = freq_ghz * 1e9
    chunks = _window_chunks(data, dra, ddec, nu, max(width, MIN_WIDTH), pb_limit)
    spws = sorted({c.spw for c in chunks})
    if channel_noise == "measure":
        noise = {spw: measure_noise([c for c in chunks if c.spw == spw]) for spw in spws}
    else:
        noise = {spw: (channel_noise or {}).get(spw) for spw in spws}
    chunks = [_without(c, nu, width, others) for c in chunks]
    dish = data.metadata.dish_diameter
    freqs = {spw: np.asarray(next(c.freq for c in chunks if c.spw == spw), dtype=float) for spw in spws}
    n_continuum = 0 if continuum_order is None else continuum_order + 1
    home = home_window(chunks, nu)  # its continuum is the one reported

    with jax.enable_x64(True):
        chunks = [on_device(c) for c in chunks]
        spectral = {}
        for k, spw in enumerate(spws):  # each window's polynomial in its own block of coefficients
            basis = np.zeros((freqs[spw].size, n_continuum * len(spws)))
            basis[:, k * n_continuum : (k + 1) * n_continuum] = np.linspace(-1.0, 1.0, freqs[spw].size)[
                :, None
            ] ** np.arange(n_continuum)
            spectral[spw] = (jnp.asarray(freqs[spw]), jnp.asarray(basis))
        report = (spectral[home][0], n_continuum, spws.index(home) * n_continuum)

        def system(theta, moments):
            chi2, grad, hess, S, Q = _system(jnp.asarray(theta), moments, chunks, spectral, dish)
            spectra = [{spw: np.asarray(x) for spw, x in d.items()} for d in (S, Q)]
            return np.asarray(chi2), np.asarray(grad), np.asarray(hess), *spectra

        def solve(theta, free, at):
            """Newton over ``theta[free]`` in units of the errors, to 1e-3 of them; ``at`` gives chi^2, grad, hess."""
            cache: dict[bytes, tuple] = {}

            def scaled(z):
                if z.tobytes() not in cache:
                    t = theta.copy()
                    t[free] += scale[free] * z
                    value, grad, hess = at(t)[:3]
                    cache.clear()
                    cache[z.tobytes()] = (
                        value,
                        grad[free] * scale[free],
                        hess[np.ix_(free, free)] * np.outer(scale, scale)[np.ix_(free, free)],
                    )
                return cache[z.tobytes()]

            result = optimize.minimize(
                lambda z: float(scaled(z)[0]),
                np.zeros(scale[free].size),
                jac=lambda z: scaled(z)[1],
                hess=lambda z: scaled(z)[2],
                method="trust-exact",
                options={"gtol": 1e-3, "initial_trust_radius": 10.0},
            )
            theta = theta.copy()
            theta[free] += scale[free] * result.x
            return theta, result.success

        SPATIAL, SPECTRAL, WIDTH = slice(0, 5), slice(5, None), 6

        def profiled(theta):
            """One pass at theta's position and shape, the spectrum fitted exactly on its moments."""
            same = np.array_equal(theta[SPATIAL], best["theta"][SPATIAL])
            moments = best["moments"] if same else _moments(theta, chunks)
            free = np.arange(theta.size)[SPECTRAL]
            theta, ok = solve(
                np.concatenate([theta[SPATIAL], best["theta"][SPECTRAL]]), free, lambda t: system(t, moments)
            )
            if theta[WIDTH] < MIN_WIDTH:  # on its bound: the rest of the spectrum refitted at MIN_WIDTH
                theta[WIDTH] = MIN_WIDTH
                theta, ok = solve(theta, free[free != WIDTH], lambda t: system(t, moments))
            value, grad = system(theta, moments)[:2]
            best.update(theta=theta, ok=ok, moments=moments)
            return value, grad

        # start: the peak and continuum, linear in the model, by one Newton step from zero
        theta0 = np.array(
            [
                dra,
                ddec,
                *covariance(bmaj, bmin, pa),
                freq_ghz * 1e9,
                max(width, MIN_WIDTH),
                0.0,
                *np.zeros(n_continuum * len(spws)),
            ]
        )
        moments = _moments(theta0, chunks)
        _, grad, hess, _, _ = system(theta0, moments)
        theta0[7:] -= np.linalg.solve(hess[7:, 7:], grad[7:])
        scale = 1 / np.sqrt(np.diag(system(theta0, moments)[2]) / 2)
        best = {"theta": theta0, "moments": moments}

        # position and shape as dra, ddec, major and minor variance and pa: sizes >= 0 are bounds
        y0 = np.array([dra, ddec, bmaj**2, bmin**2, pa])
        y_scale = np.array([*scale[:4], 10.0])

        def to_theta(y):
            return jnp.stack([y[0], y[1], *rotated(y[2], y[3], y[4], xp=jnp)])

        def objective(z):
            y = jnp.asarray(y0 + y_scale * z)
            t = theta0.copy()
            t[SPATIAL] = np.asarray(to_theta(y))
            value, grad = profiled(t)
            return float(value), np.asarray(grad[SPATIAL] @ jax.jacfwd(to_theta)(y)) * y_scale

        result = optimize.minimize(
            objective,
            np.zeros(5),
            jac=True,
            method="L-BFGS-B",
            bounds=[(None, None), (None, None), (-y0[2] / y_scale[2], None), (-y0[3] / y_scale[3], None), (None, None)],
            options={"gtol": 1e-3},
        )
        objective(result.x)
        converged = result.success and best["ok"]
        _, _, hess, S, Q = system(best["theta"], best["moments"])
        best = best["theta"]
        rho = {spw: None if noise[spw] is None else noise[spw].rho for spw in spws}
        factor = _correlation_scale(jnp.asarray(best), [(Q[spw], *spectral[spw], rho[spw]) for spw in spws])
        cov = 2 * np.linalg.inv(hess) * np.outer(factor, factor)
        values = np.array(_reported(jnp.asarray(best), *report))
        J = np.asarray(jax.jacfwd(_reported)(jnp.asarray(best), *report))
        F = {spw: np.asarray(_spectrum(jnp.asarray(best), *spectral[spw])) for spw in spws}

    errors = np.sqrt(np.diag(J @ cov @ J.T))
    # an axis on its bound (variance below 1e-6 of its scale) has a one-sided error; a point has no angle
    on_bound = np.sort((y0 + y_scale * result.x)[2:4])[::-1] < 1e-6 * y_scale[2:4].min()
    errors[2:4][on_bound] = np.nan
    values[4], errors[4] = (np.nan, np.nan) if on_bound.all() else (values[4], errors[4])
    if values[6] <= MIN_WIDTH:  # the width on its bound
        errors[6] = np.nan
    # every window's spectrum, in frequency order
    order = np.argsort(np.concatenate([freqs[spw] for spw in spws]))
    freq, S, Q, F = (np.concatenate([d[spw] for spw in spws])[order] for d in (freqs, S, Q, F))
    ok = Q > 0
    ra, dec = np.degrees(sky_direction(values[:2], data.metadata.ref))
    out = {f"fit_{k}": v for k, v in zip(REPORTED, values, strict=True)}
    out |= {f"fit_{k}_error": e if np.isfinite(e) else np.nan for k, e in zip(REPORTED, errors, strict=True)}
    return out | {
        "fit_ra": ra,
        "fit_dec": dec,
        "fit_chi2_reduced": float(Q[ok] @ (S[ok] / Q[ok] - F[ok]) ** 2)
        / max(int(ok.sum()) - 3 - n_continuum * len(spws), 1),
        "fit_n_pointings": len({c.field for c in chunks}),
        "fit_converged": bool(converged),
        "spectrum": {
            "freq_ghz": freq / 1e9,
            "flux": np.where(ok, S / np.where(ok, Q, 1.0), np.nan),
            "error": np.where(ok, 1 / np.sqrt(np.where(ok, Q, 1.0)), np.nan),
            "model": F,
        },
    }


def empty_fit(cat: Table) -> Table:
    """A copy of ``cat`` with the :data:`COLUMNS`, not yet fitted (NaN, 0, False)."""
    out = cat.copy()
    for name, unit in COLUMNS.items():
        dtype = bool if name == "fit_converged" else int if name == "fit_n_pointings" else float
        out[name] = np.full(len(cat), np.nan if dtype is float else 0, dtype=dtype)
        out[name].unit = unit
    return out


def fit_lines(data: DataHandler | str, cat: Table, rows=None, cores: int | None = None, **kwargs) -> Table:
    """A copy of ``cat`` with the :data:`COLUMNS` of every ``rows`` line (default the detected); failures warn.

    ``meta["spectra"]`` holds each fitted line's PB-corrected spectrum at its fitted position and the best-fit model.
    The other rows of ``cat`` with S/N >= ``MASK_SNR`` within a resolution element are left out of each fit
    (their channels get no weight), as the filter leaves them out of the continuum.
    """
    limit_cores(cores)
    if not isinstance(data, DataHandler):
        data = load(data)
    if kwargs.get("channel_noise", "measure") == "measure":
        kwargs["channel_noise"] = data.channel_noise()
    if rows is None:
        rows = cat["detected"] if "detected" in cat.colnames else np.ones(len(cat), dtype=bool)
    out = empty_fit(cat)
    pos, snr = np.c_[cat["dra"], cat["ddec"]].astype(float), np.asarray(cat["snr"], dtype=float)
    lines = np.c_[cat["freq_ghz"], cat["width"]].astype(float)
    for i in np.flatnonzero(rows):
        start = {k: float(cat[k][i]) for k in START if k in cat.colnames}
        near = (np.hypot(*(pos - pos[i]).T) <= data.metadata.minresolution()) & (snr >= MASK_SNR)
        near[i] = False
        try:
            values = fit_line(data, **start, others=[tuple(x) for x in lines[near]], **kwargs)
        except Exception as err:  # one line's failure must not cost the table
            warnings.warn(f"the fit of catalogue row {i} failed: {err}", stacklevel=2)
            continue
        spectrum = values.pop("spectrum")
        out.meta.setdefault("spectra", {})[str(int(cat["id"][i]) if "id" in cat.colnames else i)] = {
            k: np.asarray(v).tolist() for k, v in spectrum.items()
        }
        for name, value in values.items():
            out[name][i] = value
    return out
