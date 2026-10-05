"""Grid-search matched filter for spectral lines in the UV plane, evaluated with JAX.

For one spectral window the response at every grid point comes from three steps:

1. Spatial collapse, per channel: the weighted visibilities phase-shifted onto each trial
   position and summed, ``sig = sum_j t Re[X e^{-i phi}]``, with the per-channel weight
   ``W = sum_j w t^2`` and template amplitude ``a = sum_j w t A``. ``t`` is 1 ("natural") or the
   source envelope A ("template"). On a ``dra x ddec`` grid the phase factorises, so all positions
   of a channel are one complex matrix product; across uniformly spaced channels the phase
   factors follow by recurrence instead of new exponentials.
2. Continuum, per position: ``d = sig / W`` is the dirty spectrum at the trial position, with
   variance ``1 / W``. A continuum source there is smooth in frequency however far it lies from
   the phase centre, whereas a fit per visibility, as CASA's uvcontsub, fails off-centre: the
   source's phase winds across the window. A polynomial ``P`` in the channel index, of degree
   ``continuum_order``, is therefore fitted per window and position jointly with every line
   template of step 3, by weighted least squares. Unlike subtracting a continuum fitted first,
   the joint fit is linear in the data and exactly normalised, and it needs no line-free
   channels: the trial line's own channels are in its model.
3. Spectral collapse, per lag: the Gaussian line profile ``S_i`` centred on channel ``i`` makes
   the template ``g_i = q S_i`` with ``q = a / W``, giving ``num = sum g_i sig`` and
   ``var = sum g_i^2 W``. The continuum is profiled out with ``M = P^T W P`` and
   ``c_i = P^T W g_i``: ``num - c_i^T M^-1 P^T sig`` and ``var - c_i^T M^-1 c_i`` replace them.
   Normalised per lag over the channels it covers, the response ``num / sqrt(var)`` has unit
   variance under the null at every channel, window edges and flagged channels included, and
   peaks on the line's own channel at its S/N, less the little the continuum fit takes.
   Neighbouring channels share noise (ALMA's Hanning response correlates them by 2/3 and 1/6),
   so the variance is ``h^T C h`` instead, ``h`` the template less its continuum projection and
   ``C`` the channel correlation measured on the jackknife (``channel_correlation``): exact,
   and one number per template and lag, since ``h`` does not depend on the position.

The S/N times ``1 / sqrt(var)`` is the best-fit peak flux density of the template line, the
continuum being a nuisance; :class:`SearchResult` holds both. :func:`pb_corrected` divides them by
a pointing's primary beam and :func:`combine_pointings` merges the pointings of a mosaic by
inverse variance; :func:`search_pointings` does both for every field of a dataset.
:func:`template_spectrum` is step 1 at a single position and source shape, which
:mod:`luv_finder.fit` uses to fit the catalogued lines.

All of it runs in float64 (``jax.enable_x64``), scoped to the search.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .data import Chunk, DataHandler, fields_in, load
from .model import C_KMS, FWHM_TO_SIGMA, Gaussian, Model, envelope
from .utils import ARCSEC, C, primary_beam, primary_beam_radius

#: Axes of a line template; the templates of a search are their product in this order.
TEMPLATE_KEYS = ("bmin", "bmaj", "pa", "width")

#: Search axes, in the order of the response rows. ``total_flux`` is deliberately absent: the
#: kernel normalisation is scale-invariant, so varying it duplicates grid points.
GRID_KEYS = ("dra", "ddec", *TEMPLATE_KEYS)

#: How visibilities are weighted when a channel is collapsed: by noise only
#: ("natural", optimal for a point source) or also by the source envelope A(u, v)
#: ("template", optimal for a resolved source of the trial size).
WEIGHTINGS = ("natural", "template")

#: The :class:`~luv_finder.data.Chunk` arrays :func:`_collapse` reads, in its argument order.
COLLAPSE_ARRAYS = ("u", "v", "X", "w_row", "flag", "freq")

#: Position phases are evaluated in blocks of ``dra`` rows whose complex arrays stay near this size.
BLOCK_BYTES = 2**31


def default_cores() -> int:
    """Cores for a search on a shared machine: a quarter of them, never more than all but two."""
    n = os.cpu_count() or 1
    return max(1, min(n // 4, n - 2))


def limit_cores(cores: int | None = None) -> None:
    """Pin this process, and so JAX's and BLAS's threads, to ``cores`` cores (default :func:`default_cores`).

    Linux only; elsewhere this does nothing.
    """
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: cores or default_cores()])


@partial(jax.jit, static_argnames="template")
def _collapse(u, v, X, w_row, flag, freq, dfreq, dra, ddec, bmaj, bmin, pa, template):
    """Scan the channels: ``sig (n_chan, n_shape or 1, n_dra, n_ddec)``, ``W`` and ``a (n_chan, n_shape)``.

    ``dra``/``ddec`` are arcsec from the field centre; ``bmaj``/``bmin`` rad and ``pa`` deg, one
    entry per trial shape.
    """
    k0 = -2j * jnp.pi * freq[0] / C * ARCSEC
    dk = -2j * jnp.pi * dfreq / C * ARCSEC
    pu, du = jnp.exp(k0 * dra[:, None] * u), jnp.exp(dk * dra[:, None] * u)
    pv, dv = jnp.exp(k0 * ddec[:, None] * v), jnp.exp(dk * ddec[:, None] * v)

    def channel(phases, inputs):
        pu, pv = phases
        nu, x, f = inputs
        w = jnp.where(f, 0.0, w_row)
        shape = envelope(u * nu / C, v * nu / C, bmaj[:, None], bmin[:, None], pa[:, None], xp=jnp)
        if template:
            weight = shape**2 @ w
            amplitude = weight
            sig = ((pu[None] * (shape * x)[:, None, :]) @ pv.T).real
        else:
            weight = jnp.broadcast_to(w.sum(), bmaj.shape)
            amplitude = shape @ w
            sig = ((pu * x) @ pv.T).real[None]
        return (pu * du, pv * dv), (sig, weight, amplitude)

    return jax.lax.scan(channel, (pu, pv), (freq, X, flag))[1]


@partial(jax.jit, static_argnames="order")
def _spectral(sig, weight, amplitude, freq, widths, order=None, rho=None):
    """Line templates centred on every channel: S/N ``(n_shape, n_width, n_dra, n_ddec, n_chan)``.

    Also returns the error of the template's peak flux density ``(n_shape, n_width, n_chan)``,
    NaN where no channel weighs in; the S/N times it is the best-fit peak flux density. With
    ``order``, every template is fitted jointly with a polynomial continuum of that degree in the
    channel index scaled to [-1, 1] (see the module docstring), and a lag whose template the
    polynomial reproduces is uncovered. With ``rho`` (the channel correlation from lag 0, see
    :func:`~luv_finder.data.channel_correlation`) the S/N and the error use the exact variance
    ``h^T C h`` of the estimator under correlated channels; the flux estimate does not change.
    """
    ok = weight > 0
    q = jnp.where(ok, amplitude / jnp.where(ok, weight, 1.0), 0.0)
    spread = freq[None, :, None] * widths[:, None, None] * FWHM_TO_SIGMA / C_KMS
    profile = jnp.exp(-0.5 * ((freq[None, None, :] - freq[None, :, None]) / spread) ** 2)
    num = jnp.einsum("wim,msab->swabi", profile, q[:, :, None, None] * sig)
    var = jnp.einsum("wim,ms->swi", profile**2, q**2 * weight)
    covered = var > 0
    if order is not None:
        basis = jnp.linspace(-1.0, 1.0, freq.size)[:, None] ** jnp.arange(order + 1)
        eye = jnp.eye(order + 1)
        gram = jnp.einsum("mj,mk,ms->sjk", basis, basis, weight)
        trace = jnp.trace(gram, axis1=1, axis2=2)[:, None, None]
        gram = jnp.where(trace > 0, gram + 1e-12 * trace * eye, eye)  # the identity for a flagged window
        c = jnp.einsum("wim,ms,mj->swij", profile, q * weight, basis)
        mc = jnp.linalg.solve(gram[:, None, None], c[..., None])[..., 0]
        b = jnp.einsum("mj,msab->sabj", basis, sig)
        num = num - jnp.einsum("swij,sabj->swabi", mc, jnp.broadcast_to(b, (weight.shape[1], *b.shape[1:])))
        fitted = var - jnp.einsum("swij,swij->swi", c, mc)
        covered = fitted > 1e-9 * var  # beyond round-off
        var = fitted
    safe = jnp.where(covered, var, 1.0)
    if rho is None or rho.size == 1:
        snr_scale, error = 1 / jnp.sqrt(safe), 1 / jnp.sqrt(safe)
    else:
        # num = h^T sig with h the template less its continuum projection; under channel
        # correlation Cov(sig_m, sig_n) = rho_|m-n| sqrt(W_m W_n), so Var(num) = h^T C h
        h = profile[None] * q.T[:, None, None, :]
        if order is not None:
            h = h - jnp.einsum("swij,mj->swim", mc, basis)
        h = h * jnp.sqrt(weight.T)[:, None, None, :]
        var_c = jnp.sum(h * h, axis=-1)
        for lag in range(1, rho.size):
            var_c = var_c + 2 * rho[lag] * jnp.sum(h[..., :-lag] * h[..., lag:], axis=-1)
        var_c = jnp.where(covered, var_c, 1.0)
        snr_scale, error = 1 / jnp.sqrt(var_c), jnp.sqrt(var_c) / safe
    error = jnp.where(covered, error, jnp.nan)
    return num * jnp.where(covered, snr_scale, 0.0)[:, :, None, None, :], error


def _one_field(data: DataHandler) -> None:
    if len(data.fields) > 1:
        raise ValueError(
            f"data hold fields {data.fields}; the matched filter runs on one field at a time. "
            "Load one with DataHandler(..., fields=[i]) or luv-find --field i."
        )


def _blocks(chunk: Chunk, dra, ddec, shapes: tuple, template: bool):
    """Run :func:`_collapse` on one window in blocks of ``dra`` rows; yields (rows, (sig, weight, amplitude)).

    Must be iterated inside ``jax.enable_x64(True)``.
    """
    dfreq = np.diff(chunk.freq)
    if dfreq.size and not np.allclose(dfreq, dfreq[0], rtol=1e-9):
        raise ValueError(f"channels of field {chunk.field} spw {chunk.spw} are not uniformly spaced")
    arrays = [jnp.asarray(getattr(chunk, k)) for k in COLLAPSE_ARRAYS]
    dra = np.asarray(dra) - chunk.offset[0]
    ddec = jnp.asarray(np.asarray(ddec) - chunk.offset[1])
    n_complex = 2 + (len(shapes[0]) if template else 1)
    block = max(1, min(len(dra), BLOCK_BYTES // (16 * n_complex * len(chunk.u))))
    for start in range(0, len(dra), block):
        rows = dra[start : start + block]
        # pad the last block to the same shape, so the compiled scan is reused
        padded = jnp.asarray(np.pad(rows, (0, block - len(rows)), mode="edge"))
        step = dfreq[0] if dfreq.size else 0.0
        shape = (jnp.asarray(s) for s in shapes)
        yield len(rows), _collapse(*arrays, step, padded, ddec, *shape, template=template)


def on_device(chunk: Chunk) -> Chunk:
    """``chunk`` with the arrays :func:`_collapse` reads held by JAX in float64.

    Repeated :func:`template_spectrum` calls on it then skip copying them, which otherwise costs as
    much as collapsing one position.
    """
    with jax.enable_x64(True):
        return dataclasses.replace(chunk, **{k: jnp.asarray(getattr(chunk, k)) for k in COLLAPSE_ARRAYS})


def template_spectrum(chunk: Chunk, dra: float, ddec: float, bmaj: float, bmin: float, pa: float):
    """Template-weighted spectrum of one window at one position and source shape.

    The spatial collapse of a ``"template"``-weighted search at the single grid point (``dra``,
    ``ddec``), arcsec in the sky frame, for the envelope E of a Gaussian of axes (sigma) ``bmaj``,
    ``bmin`` arcsec with its major axis at ``pa`` deg. For a source ``V = F(nu) E e^{+i phi}`` (as
    :class:`~luv_finder.model.Gaussian`), ``chi^2 = const - 2 sum F sig + sum F^2 W``.

    Returns
    -------
    sig : (n_chan,) ``sum_j w E Re[V e^{-i phi}]``
    W : (n_chan,) ``sum_j w E^2``
    """
    shapes = (np.array([bmaj * ARCSEC]), np.array([bmin * ARCSEC]), np.array([float(pa)]))
    with jax.enable_x64(True):
        [(_, (sig, weight, _))] = _blocks(chunk, [dra], [ddec], shapes, template=True)
        return np.asarray(sig)[:, 0, 0, 0], np.asarray(weight)[:, 0]


def _beam(dra, ddec, offset, freqs, dish_diameter) -> np.ndarray:
    """Primary beam of the pointing at ``offset`` on the ``dra x ddec`` grid, ``(n_dra, n_ddec, n_chan)``."""
    distance = np.hypot(np.asarray(dra)[:, None] - offset[0], np.asarray(ddec)[None, :] - offset[1])
    return primary_beam(distance[:, :, None], freqs, dish_diameter)


def _same(a, b) -> bool:
    return np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=1e-9, atol=0)


def _check_freqs(freqs: list) -> None:
    if not all(_same(f, freqs[0]) for f in freqs[1:]):
        raise ValueError("the pointings' channel frequencies differ; only one spectral setup can be combined")


def _union_lattice(axes: list, name: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Union of the pointings' position axes, from the lowest to the highest value, and each axis' indices in it.

    The axes must share one lattice anchored at the reference, as :func:`build_grid` places them:
    the same step, and every value a whole number of steps from 0. Where an axis has a point, the
    union holds that exact value.
    """
    axes = [np.asarray(a, dtype=float) for a in axes]
    gaps = [np.diff(np.sort(a)) for a in axes if a.size > 1] or [np.diff(np.unique(np.concatenate(axes)))]
    gaps = np.concatenate(gaps)
    if not gaps.size:
        return axes[0][:1], [np.zeros(a.size, dtype=int) for a in axes]
    step = gaps.min()
    k = [a / step for a in axes]
    spacing = [np.diff(np.sort(x)).min() for x in k if x.size > 1]
    if not (np.allclose(spacing, 1.0) and all(np.allclose(x, np.round(x), rtol=0, atol=1e-6) for x in k)):
        raise ValueError(
            f"the pointings' {name} axes do not lie on one lattice: each must step by the same {step:.6g} arcsec, "
            "with every value a whole number of steps from the reference (as build_grid places them)"
        )
    k = [np.round(x).astype(int) for x in k]
    lo = min(x.min() for x in k)
    union = step * np.arange(lo, max(x.max() for x in k) + 1)
    for a, x in zip(axes, k, strict=True):
        union[x - lo] = a
    return union, [x - lo for x in k]


class _Grid:
    """Grid-point bookkeeping of a search, for a class with ``axes``, ``freqs`` (Hz) and ``response``."""

    @property
    def grid_params(self) -> list[dict]:
        """Every grid point as ``{"src_00_<key>": value}``, in the order of the response rows."""
        return [
            {f"src_00_{k}": v for k, v in zip(GRID_KEYS, combo, strict=True)}
            for combo in product(*(self.axes[k] for k in GRID_KEYS))
        ]

    @property
    def best_index(self) -> int:
        """Row of the highest response; NaN (masked) values are ignored."""
        return int(np.nanargmax(np.fmax.reduce(self.response, axis=1)))

    @property
    def best_params(self) -> dict:
        return self.grid_params[self.best_index]

    def frequencies(self) -> np.ndarray:
        """Channel frequencies in GHz."""
        return self.freqs / 1e9


@dataclass(eq=False)
class SearchResult(_Grid):
    """Matched-filter search of one pointing or of a mosaic.

    Grid points are the product of the :data:`GRID_KEYS` axes; the arrays hold them as
    ``(n_dra, n_ddec, n_template, n_chan)``, the templates being the product of the
    :data:`TEMPLATE_KEYS` axes in that order. NaN marks a position or channel left uncovered
    (see :func:`pb_corrected`).

    Attributes
    ----------
    axes : dict of the :data:`GRID_KEYS` axes, lists of floats
    freqs : (n_chan,) Hz
    snr : matched-filter S/N
    flux : best-fit peak flux density of the template line, Jy (PB-corrected if ``coverage`` is)
    error : 1-sigma error of ``flux``, Jy
    coverage : (n_dra, n_ddec, n_chan) relative sensitivity: 1 for an uncorrected pointing, its
        primary beam once corrected, ``sqrt(sum_p PB_p**2)`` for a combination; 0 where uncovered
    jackknife : the same search of the jackknifed data, if it was run
    """

    axes: dict
    freqs: np.ndarray
    snr: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    coverage: np.ndarray
    jackknife: SearchResult | None = None

    @property
    def response(self) -> np.ndarray:
        """S/N as ``(n_grid, n_chan)``, one row per :attr:`grid_params` entry."""
        return self.snr.reshape(-1, self.snr.shape[-1])

    @property
    def response_jackknife(self) -> np.ndarray | None:
        return None if self.jackknife is None else self.jackknife.response

    def integrated_flux(self) -> tuple[np.ndarray, np.ndarray]:
        """Line flux and its error in Jy km/s: ``flux`` and ``error`` times ``width * sqrt(2 pi) / 2.355``."""
        widths = np.array([combo[-1] for combo in product(*(self.axes[k] for k in TEMPLATE_KEYS))])
        scale = widths[:, None] * np.sqrt(2 * np.pi) * FWHM_TO_SIGMA
        return self.flux * scale, self.error * scale


def pb_corrected(result: SearchResult, data: DataHandler, pb_limit: float = 0.2) -> SearchResult:
    """Correct the search of one pointing for its primary beam.

    ``result`` comes from :meth:`MatchedFilter.run` on ``data``. ``flux`` and ``error`` are divided
    by the pointing's primary beam PB at every position and channel, so ``flux`` estimates the
    intrinsic peak flux density; ``snr`` does not change. Where PB < ``pb_limit`` all three are
    NaN and ``coverage`` is 0; elsewhere ``coverage`` is PB. The jackknife is corrected alike.
    """
    _one_field(data)
    pb = _beam(
        result.axes["dra"], result.axes["ddec"], data.chunks[0].offset, result.freqs, data.metadata.dish_diameter
    )
    keep = pb >= pb_limit
    coverage = np.where(keep, pb, 0.0)
    scale = np.where(keep, pb, np.nan)[:, :, None]

    def correct(r: SearchResult) -> SearchResult:
        return dataclasses.replace(
            r,
            snr=np.where(keep[:, :, None], r.snr, np.nan),
            flux=r.flux / scale,
            error=r.error / scale,
            coverage=coverage,
            jackknife=None if r.jackknife is None else correct(r.jackknife),
        )

    return correct(result)


def combine_pointings(results: Sequence[SearchResult]) -> SearchResult:
    """Combine the PB-corrected searches of a mosaic's pointings on the union of their grids.

    Each pointing weighs in with ``w = 1 / error**2`` (0 where it is NaN), its observation weight
    times PB^2: ``flux = sum w F / sum w``, ``error = 1 / sqrt(sum w)`` and ``snr = flux / error``,
    so a source's S/N is ``sqrt(sum_p snr_p**2)``. All three are NaN where no pointing covers a
    position, and ``coverage`` is ``sqrt(sum_p coverage_p**2)``. The jackknives are combined the
    same way when every result has one.

    The results must share their templates and channels, and their positions one lattice
    anchored at the reference (see :func:`build_grid`); otherwise this raises ``ValueError``.
    """
    first = results[0]
    for r in results[1:]:
        for k in TEMPLATE_KEYS:
            if not _same(r.axes[k], first.axes[k]):
                raise ValueError(
                    f"the pointings were searched with different {k} templates: {first.axes[k]}, {r.axes[k]}"
                )
    _check_freqs([r.freqs for r in results])
    dra, ira = _union_lattice([r.axes["dra"] for r in results], "dra")
    ddec, idec = _union_lattice([r.axes["ddec"] for r in results], "ddec")

    wsum = np.zeros((len(dra), len(ddec), *first.snr.shape[2:]))
    wflux = np.zeros_like(wsum)
    coverage = np.zeros((len(dra), len(ddec), len(first.freqs)))
    for r, i, j in zip(results, ira, idec, strict=True):
        at = np.ix_(i, j)
        w = np.nan_to_num(r.error**-2.0)
        wsum[at] += w
        wflux[at] += w * np.nan_to_num(r.flux)
        coverage[at] += r.coverage**2
    wsum = np.where(wsum > 0, wsum, np.nan)
    flux, error = wflux / wsum, 1 / np.sqrt(wsum)

    jackknives = [r.jackknife for r in results]
    return SearchResult(
        axes={"dra": dra.tolist(), "ddec": ddec.tolist(), **{k: first.axes[k] for k in TEMPLATE_KEYS}},
        freqs=first.freqs,
        snr=flux / error,
        flux=flux,
        error=error,
        coverage=np.sqrt(coverage),
        jackknife=combine_pointings(jackknives) if all(j is not None for j in jackknives) else None,
    )


@dataclass(frozen=True, eq=False)
class DirtyCube:
    """Natural-weighted dirty cube of one pointing on a ``dra x ddec`` grid (see :func:`dirty_cube`).

    Attributes
    ----------
    cube : (n_chan, n_dra, n_ddec) Jy/beam, the windows in frequency order
    weight : (n_chan,) channel weights: the noise of channel ``i`` is ``1 / sqrt(weight[i])``
    dra, ddec : grid axes, arcsec in the dataset's sky frame
    offset : (2,) arcsec, the pointing centre in that frame
    freqs : (n_chan,) Hz
    dish_diameter : m
    window_sizes : channels per window, in order
    """

    cube: np.ndarray
    weight: np.ndarray
    dra: list[float]
    ddec: list[float]
    offset: np.ndarray
    freqs: np.ndarray
    dish_diameter: float
    window_sizes: list[int]


def dirty_cube(data: DataHandler, dra, ddec, cores: int | None = None) -> DirtyCube:
    """Natural-weighted dirty cube of one field on the ``dra x ddec`` grid (arcsec, sky frame).

    This is the search's spatial collapse for a point source, so it costs as much as one search
    pass; a natural-weighted search gives it for free with ``MatchedFilter.run(cube=True)``.
    """
    _one_field(data)
    limit_cores(cores)
    point = (np.zeros(1), np.zeros(1), np.zeros(1))
    with jax.enable_x64(True):
        windows = [[(n, *collapsed[:2]) for n, collapsed in _blocks(c, dra, ddec, point, False)] for c in data.chunks]
    return _dirty_cube(data, dra, ddec, windows)


def _dirty_cube(data: DataHandler, dra, ddec, windows: list[list]) -> DirtyCube:
    """The dirty cube of ``data`` from every window's natural-weighted :func:`_collapse` blocks.

    ``windows`` holds per window the blocks as ``(rows, sig, weight)``, in :func:`_blocks` order.
    """
    cube, weights = [], []
    for blocks in windows:
        sig = np.concatenate([np.asarray(s)[:, 0, :n] for n, s, _ in blocks], axis=1)
        weight = np.asarray(blocks[0][2])[:, 0]
        w = weight[:, None, None]
        cube.append(np.divide(sig, w, out=np.zeros_like(sig), where=w > 0))
        weights.append(weight)
    return DirtyCube(
        cube=np.concatenate(cube),
        weight=np.concatenate(weights),
        dra=np.asarray(dra, dtype=float).tolist(),
        ddec=np.asarray(ddec, dtype=float).tolist(),
        offset=np.asarray(data.chunks[0].offset),
        freqs=data.freqs,
        dish_diameter=data.metadata.dish_diameter,
        window_sizes=[len(c.freq) for c in data.chunks],
    )


def _moments(cube: np.ndarray, ivar: np.ndarray, window_sizes: list[int]):
    """Moment-8, continuum and continuum noise of a ``(n_chan, n_dra, n_ddec)`` cube in Jy/beam.

    ``ivar`` is the inverse variance of the cube's values, broadcastable against it with all
    channels along its first axis, and 0 where a value is not covered; ``window_sizes`` are the
    channels per window. Returns ``(moment8, continuum, sigma)``:

    * ``continuum``: the inverse-variance weighted mean over channels, with noise ``sigma``, which
      has the shape of ``ivar`` summed over channels;
    * ``moment8``: the largest S/N over channels after subtracting each window's own weighted
      continuum, so a continuum source does not dominate it.

    All are NaN where no channel covers a pixel.
    """

    def mean(part, w):
        total = w.sum(axis=0)
        return (w * part).sum(axis=0) / np.where(total > 0, total, np.nan), total

    continuum, total = mean(cube, ivar)
    edges = np.cumsum(window_sizes)[:-1]
    windows = zip(np.split(cube, edges), np.split(ivar, edges), strict=True)
    line = np.concatenate([part - mean(part, w)[0] for part, w in windows])
    moment8 = np.fmax.reduce(np.where(ivar > 0, line * np.sqrt(ivar), np.nan), axis=0)
    return moment8, continuum, 1 / np.sqrt(np.where(total > 0, total, np.nan))


def dirty_maps(data: DataHandler, dra, ddec, cores: int | None = None):
    """Moment-8 and continuum dirty maps of one field on the ``dra x ddec`` grid.

    Returns ``(moment8, continuum, sigma)``:

    * ``continuum``: the weighted mean of the dirty cube over every channel, Jy/beam, with noise
      ``sigma``;
    * ``moment8``: for every position the largest channel value over its noise, in S/N, after
      subtracting each window's own weighted-mean continuum, so a continuum source does not
      dominate it (``max_nu [I / sigma_nu]``, as in alma-data-prep's moment-8).
    """
    c = dirty_cube(data, dra, ddec, cores)
    moment8, continuum, sigma = _moments(c.cube, c.weight[:, None, None], c.window_sizes)
    return moment8, continuum, sigma.item()


def mosaic_dirty_maps(cubes: Sequence[DirtyCube], pb_limit: float = 0.2):
    """Moment-8 and continuum dirty maps of a mosaic, on the union of the pointings' grids.

    Per channel the pointings' dirty cubes form a linear mosaic, the PB-corrected sky
    ``I = sum_p PB_p w_p I_p / sum_p PB_p^2 w_p`` with inverse variance ``sum_p PB_p^2 w_p``, where
    ``w_p`` is pointing p's channel weight; a pointing counts only where its PB >= ``pb_limit``.
    The grids must share one lattice, as for :func:`combine_pointings`.

    Returns ``(dra, ddec, moment8, continuum, sigma)``: the union axes, the moment-8 in S/N and the
    continuum in Jy/beam as in :func:`dirty_maps`, and the continuum noise per pixel. The maps are
    NaN where no pointing covers a pixel.
    """
    first = cubes[0]
    _check_freqs([c.freqs for c in cubes])
    if any(c.window_sizes != first.window_sizes for c in cubes):
        raise ValueError("the pointings' spectral windows differ; only one spectral setup can be combined")
    dra, ira = _union_lattice([c.dra for c in cubes], "dra")
    ddec, idec = _union_lattice([c.ddec for c in cubes], "ddec")

    num = np.zeros((len(first.freqs), len(dra), len(ddec)))
    ivar = np.zeros_like(num)
    for c, i, j in zip(cubes, ira, idec, strict=True):
        pb = np.moveaxis(_beam(c.dra, c.ddec, c.offset, c.freqs, c.dish_diameter), -1, 0)
        w = np.where(pb >= pb_limit, pb, 0.0) * c.weight[:, None, None]
        at = (slice(None), *np.ix_(i, j))
        num[at] += w * c.cube
        ivar[at] += w * pb
    cube = np.divide(num, ivar, out=np.zeros_like(num), where=ivar > 0)
    return (dra, ddec, *_moments(cube, ivar, first.window_sizes))


class MatchedFilter(_Grid):
    """Evaluate a model kernel against the data over a parameter grid.

    :meth:`run` fills ``result``, a :class:`SearchResult`. Its ``response`` (also forwarded as
    ``MatchedFilter.response``) is in signal-to-noise units: one row per grid point
    (``grid_params``, the product of the :data:`GRID_KEYS` axes in that order), one column per
    channel, unit variance under the null hypothesis. A line matching the template at that grid
    point peaks on its own channel with a value equal to its S/N, which a continuum fitted
    alongside lowers a little.

    Parameters
    ----------
    data : DataHandler
        One field; pointings of a mosaic are searched one at a time on a common grid and joined
        with :func:`pb_corrected` and :func:`combine_pointings`.
    mod : Model
        With one component whose ``grid`` dict sets the search axes; an axis the grid does
        not name takes the component's own value.
    weighting : {"natural", "template"}
        See :data:`WEIGHTINGS`. With "natural" the trial source size cancels.
    continuum_order : int or None
        Degree of the polynomial in frequency fitted, per window and trial position, jointly with
        every line template (see the module docstring); None fits no continuum. The dirty
        ``cube`` of :meth:`run` keeps the continuum.
    channel_correlation : "measure", None or dict
        Noise correlation between channels, per spectral window. "measure" (default) measures it
        on the data's jackknife when :meth:`run` starts (see
        :func:`~luv_finder.data.channel_correlation`); None assumes independent channels; a dict
        ``{spw: (1, rho_1, ...)}`` gives it. The S/N and errors account for it exactly, so the
        response has unit variance under the null either way. :meth:`run` leaves the values used
        in ``self.correlation``.
    """

    def __init__(
        self,
        data: DataHandler,
        mod: Model,
        weighting: str = "natural",
        continuum_order: int | None = 2,
        channel_correlation: str | dict | None = "measure",
    ):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        if isinstance(channel_correlation, str) and channel_correlation != "measure":
            raise ValueError(f'channel_correlation must be "measure", None or a dict, got {channel_correlation!r}')
        _one_field(data)
        self.data = data
        self.weighting = weighting
        self.continuum_order = continuum_order
        self.channel_correlation = channel_correlation
        self.correlation: dict[int, np.ndarray] = {}
        self.axes = self._grid_axes(mod)
        self.result: SearchResult | None = None
        self.cube: DirtyCube | None = None

    @staticmethod
    def _grid_axes(mod: Model) -> dict[str, list[float]]:
        grid = {k.split("_", 2)[-1]: v for k, v in mod.grid.items()}
        unknown = sorted(set(grid) - set(GRID_KEYS))
        if unknown:
            raise ValueError(
                f"grid keys {unknown} are not searchable. Searchable keys are {list(GRID_KEYS)}. "
                "total_flux in particular cancels in the kernel normalisation, so varying it "
                "only duplicates grid points."
            )
        own = {k.lstrip("_"): v for k, v in vars(mod.component(0)).items()}
        return {k: np.atleast_1d(grid.get(k, own[k])).astype(float).tolist() for k in GRID_KEYS}

    @property
    def freqs(self) -> np.ndarray:
        """Channel frequencies in Hz."""
        return self.data.freqs

    @property
    def response(self) -> np.ndarray | None:
        return None if self.result is None else self.result.response

    @property
    def response_jackknife(self) -> np.ndarray | None:
        return None if self.result is None else self.result.response_jackknife

    def run(self, jackknife: bool = False, cores: int | None = None, cube: bool = False) -> None:
        """Fill ``result`` and, with ``jackknife``, ``result.jackknife`` on the same grid.

        With ``cube`` (natural weighting only) also fill ``cube``, the data's :class:`DirtyCube`
        on the search grid, from the spatial collapse the search evaluates anyway. The process is
        first pinned to ``cores`` cores (see :func:`limit_cores`).
        """
        if cube and self.weighting != "natural":
            raise ValueError("cube=True needs natural weighting: template weighting does not collapse to a dirty cube")
        limit_cores(cores)
        if self.channel_correlation == "measure":
            self.correlation = self.data.channel_correlation()
        else:
            self.correlation = {k: np.asarray(v, dtype=float) for k, v in (self.channel_correlation or {}).items()}
        self.result, self.cube = self._search(self.data, cube)
        if jackknife:
            self.result.jackknife = self._search(self.data.jackknife())[0]

    def _search(self, data: DataHandler, cube: bool = False) -> tuple[SearchResult, DirtyCube | None]:
        axes = {k: np.asarray(v) for k, v in self.axes.items()}
        bmin, bmaj, pa = (np.asarray(s) for s in zip(*product(axes["bmin"], axes["bmaj"], axes["pa"]), strict=True))
        shapes = (bmaj * ARCSEC, bmin * ARCSEC, pa)
        with jax.enable_x64(True):
            windows = [self._window(c, axes, shapes, cube) for c in tqdm(data.chunks, desc="Windows", leave=False)]
        snr, error, blocks = zip(*windows, strict=True)
        # (n_shape, n_width, n_dra, n_ddec, n_chan) and (n_shape, n_width, n_chan)
        snr, error = np.concatenate(snr, axis=-1), np.concatenate(error, axis=-1)
        snr = snr.transpose(2, 3, 0, 1, 4).reshape(*snr.shape[2:4], -1, snr.shape[-1])
        error = np.broadcast_to(error.reshape(-1, error.shape[-1]), snr.shape)
        coverage = np.broadcast_to(1.0, (*snr.shape[:2], snr.shape[-1]))
        result = SearchResult(dict(self.axes), data.freqs, snr, snr * error, error, coverage)
        return result, _dirty_cube(data, self.axes["dra"], self.axes["ddec"], blocks) if cube else None

    def _window(self, chunk: Chunk, axes: dict, shapes: tuple, cube: bool) -> tuple:
        """S/N and error of one window and, with ``cube``, its collapsed blocks for :func:`_dirty_cube`."""
        snr, blocks = [], []
        widths, freq = jnp.asarray(axes["width"]), jnp.asarray(chunk.freq)
        rho = self.correlation.get(chunk.spw)
        rho = None if rho is None or rho.size == 1 else jnp.asarray(rho)
        for n, collapsed in _blocks(chunk, axes["dra"], axes["ddec"], shapes, self.weighting == "template"):
            block, error = _spectral(*collapsed, freq, widths, order=self.continuum_order, rho=rho)
            snr.append(np.asarray(block)[:, :, :n])
            if cube:
                blocks.append((n, *collapsed[:2]))
        return np.concatenate(snr, axis=2), np.asarray(error), blocks

    def plot_response(
        self, filename: str = "plots/filter_response.png", show: bool = False, vline: float | None = None
    ):
        """Save the best grid point's S/N spectrum (see :mod:`luv_finder.plotting`)."""
        from .plotting import response_check

        plots_dir, name = os.path.split(filename)
        path = response_check(self, plots_dir=plots_dir or ".", name=name, line_ghz=vline)
        if show:  # pragma: no cover - interactive only
            import matplotlib.pyplot as plt

            plt.show()
        return path


def build_grid(data: DataHandler, cfg: dict | None) -> dict:
    """Grid ranges from YAML, positions in the dataset's sky frame (arcsec from the reference).

    Positions default to multiples of half the resolution, counted from the reference
    direction and covering ``fov_fraction`` (default 0.4) of the primary-beam FWHM on either
    side of the loaded field. With ``pb_limit`` they instead reach the radius where the primary
    beam of the lowest channel, the widest, falls to ``pb_limit``, so every channel's
    ``PB >= pb_limit`` region is covered (see :func:`pb_corrected`).
    The lattice comes from dataset-level metadata, so every pointing of a mosaic gets the same
    points. Sizes default to a tenth of the resolution, the position angle to 0, the line widths
    to 100, 200, 300 and 400 km/s.
    """
    fov = data.metadata.primarybeamsize()
    res = data.metadata.minresolution()
    cfg = dict(cfg or {})
    if "total_flux" in cfg:
        raise ValueError(
            "total_flux is not a search axis: it cancels in the matched-filter kernel "
            "normalisation, so every value gives an identical response. Remove it from the grid file."
        )

    def rng(key, default):
        v = cfg.get(key, default)
        if isinstance(v, dict):
            return np.arange(v["start"], v["stop"], v["step"])
        return np.atleast_1d(v)

    if "pb_limit" in cfg:
        half = primary_beam_radius(cfg["pb_limit"], data.freqs.min(), data.metadata.dish_diameter)
    else:
        half = cfg.get("fov_fraction", 0.4) * fov
    step = res / 2

    def lattice(centre):
        return step * np.arange(np.ceil((centre - half) / step), np.floor((centre + half) / step) + 1)

    dra, ddec = data.chunks[0].offset
    return {
        "dra": rng("dra", lattice(dra)),
        "ddec": rng("ddec", lattice(ddec)),
        "bmin": rng("bmin", res / 10),
        "bmaj": rng("bmaj", res / 10),
        "pa": rng("pa", 0.0),
        "width": rng("width", [100.0, 200.0, 300.0, 400.0]),
    }


def grid_model(grid: dict) -> Model:
    """A model of one Gaussian component whose ``grid`` is the search axes ``grid`` (see :func:`build_grid`)."""
    comp = Gaussian()
    comp.grid = grid
    mod = Model()
    mod.addcomponent(comp)
    return mod


def search_pointings(
    path: str,
    grid_cfg: dict | None = None,
    jackknife: bool = True,
    cube: bool = False,
    cores: int | None = None,
    pb_limit: float = 0.2,
    **filter_kwargs,
) -> tuple[SearchResult, list[DirtyCube]]:
    """Search every field of a measurement set or NPZ and combine the pointings on one sky grid.

    Pointings that overlap on the sky are always PB-corrected and combined before anything
    downstream (cataloguing) sees them; a single pointing is PB-corrected too. The fields are
    loaded and searched one at a time, so only one field's visibilities are in memory.

    Parameters
    ----------
    path : str
        Measurement set or NPZ from ``luv-export``.
    grid_cfg : dict, optional
        Grid ranges as for :func:`build_grid`, shared by every pointing. Each pointing's positions
        are those within ``pb_limit`` of its own phase centre unless ``dra``/``ddec`` are given.
    jackknife : bool
        Also search a jackknifed noise realisation of every pointing, combined alike.
    cube : bool
        Also return every pointing's :class:`DirtyCube` (natural weighting only).
    cores : int, optional
        See :func:`limit_cores`.
    pb_limit : float
        Positions and channels where a pointing's primary beam is below this are left out
        (see :func:`pb_corrected`).
    **filter_kwargs
        Passed to :class:`MatchedFilter` (``weighting``, ``continuum_order``, ``channel_correlation``).

    Returns
    -------
    result : SearchResult
        The PB-corrected search, combined by :func:`combine_pointings` if there are several fields.
    cubes : list of DirtyCube
        One per field in field order; empty unless ``cube``.
    """
    results, cubes = [], []
    for field in fields_in(path):
        data = load(path, [field])
        grid = build_grid(data, {**(grid_cfg or {}), "pb_limit": pb_limit})
        mf = MatchedFilter(data, grid_model(grid), **filter_kwargs)
        mf.run(jackknife=jackknife, cores=cores, cube=cube)
        results.append(pb_corrected(mf.result, data, pb_limit))
        if cube:
            cubes.append(mf.cube)
        del data, mf  # free this field's visibilities before loading the next
    return combine_pointings(results) if len(results) > 1 else results[0], cubes
