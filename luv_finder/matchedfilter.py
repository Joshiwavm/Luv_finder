"""Grid-search matched filter for spectral lines in the UV plane, in JAX (float64).

Per window and trial source shape, :mod:`luv_finder.kernel` collapses the visibilities, weighted
by the source's own visibility amplitude, onto every trial position: the dirty spectrum
``d = sig / W`` with variance ``1 / W``. At every position and channel the model is
``d = A g + P c``, the Gaussian line template ``g`` on a polynomial continuum ``P c`` in the
channel index. With ``c`` free, ``A`` is the fit of the template projected off the polynomials,
``h = g - P (P^T W P)^-1 P^T W g``: ``A = h^T W d / h^T W h``. A polynomial cannot mimic a narrow
Gaussian, so no line-free channels are needed; the line loses only the polynomial-like part of
``g``. The error of ``A`` includes the channel correlation measured on the jackknife, so the S/N
has unit variance under the null.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .data import ChannelNoise, Chunk, DataHandler, fields_in, load, pointing
from .kernel import _blocks, limit_cores
from .model import C_KMS, CONTINUUM_ORDER, Gaussian, Model, continuum_basis, covariance, line_profile
from .utils import ARCSEC, primary_beam_at, primary_beam_radius

TEMPLATE_KEYS = ("bmin", "bmaj", "pa", "width")
GRID_KEYS = ("dra", "ddec", *TEMPLATE_KEYS)
#: A line whose response peaks at ``MASK_SNR`` or more at a position is left out of the continuum fit
#: there, within ``MASK_FWHM`` template FWHM of its peak, so it does not bend the continuum of the others.
MASK_SNR, MASK_FWHM = 5.0, 1.0
#: The brightest peaks per position that may mask (a position rarely holds more lines at ``MASK_SNR``).
MASK_PEAKS = 4
#: Arrays of a block's positions x templates x centres that :func:`_spectral` holds at once, at most: it sizes the
#: position blocks, so a window's spectral fit stays within ``BLOCK_BYTES``.
SPECTRAL_ARRAYS = 12
#: Template FWHMs (km/s) a grid searches unless it names its own.
WIDTHS = (100.0, 200.0, 300.0, 400.0)


@partial(jax.jit, static_argnames="order")
def _spectral(sig, weight, freq, centres, widths, rho, order=None):
    """S/N and peak error, both ``(n_shape, n_width, n_dra, n_ddec, n_centre)``.

    With a continuum, the polynomial is fitted jointly with every template, then again at each
    position without the channels of the other lines that peak there at ``MASK_SNR`` or more
    (:func:`_masked`), so a bright line does not bend the continuum of the rest. Only positions
    holding such a line are refitted; elsewhere nothing is masked and the joint fit stands.
    """
    profile = line_profile(freq[None, None, :], centres[None, :, None], widths[:, None, None], jnp)
    num = num0 = jnp.einsum("wim,msab->swabi", profile, sig)
    var = v0 = jnp.einsum("wim,ms->swi", profile**2, weight)
    h = jnp.broadcast_to(profile, (weight.shape[1], *profile.shape))
    covered = var > 0
    if order is not None:
        basis = continuum_basis(freq.size, order + 1, jnp)
        eye = jnp.eye(order + 1)
        gram = jnp.einsum("mj,mk,ms->sjk", basis, basis, weight)
        trace = jnp.trace(gram, axis1=1, axis2=2)[:, None, None]
        gram = jnp.where(trace > 0, gram + 1e-12 * trace * eye, eye)  # the identity for a flagged window
        c = jnp.einsum("wim,ms,mj->swij", profile, weight, basis)
        mc = jnp.linalg.solve(gram[:, None, None], c[..., None])[..., 0]
        num = num - jnp.einsum("swij,sabj->swabi", mc, jnp.einsum("mj,msab->sabj", basis, sig))
        fitted = var - jnp.einsum("swij,swij->swi", c, mc)
        covered = fitted > 1e-9 * var  # beyond round-off
        var = fitted
        h = h - jnp.einsum("swij,mj->swim", mc, basis)
    h = h * jnp.sqrt(weight.T)[:, None, None, :]
    var_c = jnp.sum(h * h, axis=-1)
    for lag in range(1, rho.size):
        var_c = var_c + 2 * rho[lag] * jnp.sum(h[..., :-lag] * h[..., lag:], axis=-1)
    var_c = jnp.where(covered, var_c, 1.0)
    snr = num * jnp.where(covered, 1 / jnp.sqrt(var_c), 0.0)[:, :, None, None, :]
    error = jnp.where(covered, jnp.sqrt(var_c) / jnp.where(covered, var, 1.0), jnp.nan)
    error = jnp.broadcast_to(error[:, :, None, None, :], snr.shape)
    if order is None:
        return snr, error
    fixed = (weight, freq, centres, widths, profile, basis, gram, c, v0, var_c / var, covered)
    peak, which = _peaks(snr)

    def refit(peak, which, sig, num0):
        """The masked fit, twice: the second from the first's peaks, where a peak only a brighter line's bent
        continuum made (at a window's far edge, say) has gone and masks nothing."""
        first = _masked(peak, which, sig, num0, *fixed)
        return _masked(*_peaks(first[0]), sig, num0, *fixed)

    def position(args):
        """One position: refitted only if a line there reaches ``MASK_SNR``, else the joint fit."""
        snr, error, sig, num0, peak, which = args
        one = (peak[:, None, None], which[:, None, None], sig[:, :, None, None], num0[:, :, None, None])
        return jax.lax.cond((peak > 0).any(), lambda: tuple(a[:, :, 0, 0] for a in refit(*one)), lambda: (snr, error))

    def lead(a, at):
        """Positions (axes ``at``, ``at + 1``) first and flat, mapped one at a time: ``cond`` runs one branch."""
        return jnp.moveaxis(a, (at, at + 1), (0, 1)).reshape(-1, *a.shape[:at], *a.shape[at + 2 :])

    n_dra, n_ddec = snr.shape[2:4]
    out = jax.lax.map(position, (*(lead(a, 2) for a in (snr, error, sig, num0)), lead(peak, 1), lead(which, 1)))
    return tuple(jnp.moveaxis(a.reshape(n_dra, n_ddec, *a.shape[1:]), (0, 1), (2, 3)) for a in out)


def _peaks(snr):
    """Best-template S/N where it peaks at ``MASK_SNR`` or more (else 0) and its template, ``(shape, dra, ddec, i)``."""
    best, which = snr.max(axis=1), snr.argmax(axis=1)
    low = jnp.pad(best, ((0, 0), (0, 0), (0, 0), (1, 0)), constant_values=-jnp.inf)[..., :-1]
    high = jnp.pad(best, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=-jnp.inf)[..., 1:]
    return jnp.where((best >= MASK_SNR) & (best >= low) & (best >= high), best, 0.0), which


def _masked(peak, which, sig, num0, weight, freq, centres, widths, profile, basis, gram, c, v0, ratio, covered):
    """The joint fit again, at each position without the channels of its other lines (``MASK_SNR``).

    A line peaking at ``MASK_SNR`` or more at a position (``peak``, its S/N; the ``MASK_PEAKS`` brightest)
    masks ``MASK_FWHM`` of its template's FWHM around the peak, for every template and centre but those
    it is within ``MASK_FWHM`` of: there it is the centre's own line, whose peak noise or a window edge
    can shift. Each template also keeps its own ``MASK_FWHM``, so an isolated line loses nothing. The
    least-squares fit of template and polynomial then runs with the channel weights ``m W``: the full
    sums minus the masked channels' part. The variance is exact for independent channels, scaled by the
    channel correlation's ``ratio`` of the unmasked fit. Where nothing is masked this is the joint fit.
    """
    velocity = jnp.abs(freq[None, :] / centres[:, None] - 1) * C_KMS  # (n_centre, n_chan)
    near = (velocity[None] <= MASK_FWHM * widths[:, None, None]).astype(float)  # (n_width, n_centre, n_chan)
    strength, at = jax.lax.top_k(
        peak, min(MASK_PEAKS, peak.shape[-1])
    )  # the brightest peaks' S/N and centre, (n_shape, n_dra, n_ddec, k)
    reach = near[jnp.take_along_axis(which, at, axis=-1), at]  # the channels each masks, (..., k, n_chan)
    offset = jnp.abs(centres[at][..., None, None, :] / centres[:, None] - 1) * C_KMS  # (..., 1, n_centre, k)
    masks = (strength[..., None, None, :] > 0) & (
        offset > MASK_FWHM * widths[:, None, None]
    )  # (..., n_width, n_centre, k)
    masked = (jnp.einsum("sabwik,sabkm->sabwim", masks.astype(float), reach) > 0) * (1.0 - near)
    # the masked channels' part of every sum of the joint fit, per position and template
    pw = basis[:, :, None] * basis[:, None, :]  # (n_chan, j, k)
    d_gram = jnp.einsum("sabwim,ms,mjk->swabijk", masked, weight, pw)
    d_c = jnp.einsum("sabwim,ms,wim,mj->swabij", masked, weight, profile, basis)
    d_v = jnp.einsum("sabwim,ms,wim->swabi", masked, weight, profile**2)
    d_num = jnp.einsum("sabwim,wim,msab->swabi", masked, profile, sig)
    d_r = jnp.einsum("sabwim,mj,msab->swabij", masked, basis, sig)
    g_m = gram[:, None, None, None, None] - d_gram
    c_m = c[:, :, None, None] - d_c
    r_m = jnp.einsum("mj,msab->sabj", basis, sig)[:, None, :, :, None] - d_r
    coef = jnp.linalg.solve(g_m, c_m[..., None])[..., 0]  # G_m^-1 c_m
    num = num0 - d_num - jnp.einsum("swabij,swabij->swabi", coef, r_m)
    var = v0[:, :, None, None] - d_v - jnp.einsum("swabij,swabij->swabi", coef, c_m)
    var_c = var * ratio[:, :, None, None]
    ok = covered[:, :, None, None] & (var > 1e-9 * v0[:, :, None, None])
    root = jnp.sqrt(jnp.where(ok, var_c, 1.0))
    return jnp.where(ok, num / root, 0.0), jnp.where(ok, root / jnp.where(ok, var, 1.0), jnp.nan)


def window_groups(freqs: dict, widest: float = max(WIDTHS)) -> list[list]:
    """Windows (``{key: freq}``) joined where they overlap or their gap is narrower than ``widest`` km/s.

    A line no wider than the widest template is then searched and fitted in every window that holds it.
    The groups, each a sorted list of keys, come in frequency order.
    """
    groups: list[list] = []
    for lo, hi, k in sorted((f.min(), f.max(), k) for k, f in freqs.items()):
        if groups and lo < groups[-1][1] * (1 + widest / C_KMS):
            groups[-1][1] = max(groups[-1][1], hi)
            groups[-1][2].append(k)
        else:
            groups.append([lo, hi, [k]])
    return [sorted(ks) for *_, ks in groups]


def _reach(chunks: Sequence[Chunk], widest: float) -> list[tuple[float, float]]:
    """Each window's trial centres: up to ``widest`` km/s beyond its edges, within its group (:func:`window_groups`).

    Where windows meet, a template centred near the edge of one sums the channels of both: their estimates
    add by inverse variance, the joint fit of one template with a continuum per window.
    """
    spans = [(0.0, 0.0)] * len(chunks)
    for group in window_groups(dict(enumerate(c.freq for c in chunks)), widest):
        lo, hi = min(chunks[k].freq[0] for k in group), max(chunks[k].freq[-1] for k in group)
        for k in group:
            f0, f1 = chunks[k].freq[0], chunks[k].freq[-1]
            spans[k] = (max(f0 * (1 - widest / C_KMS), lo), min(f1 * (1 + widest / C_KMS), hi))
    return spans


def _centres(chunks: Sequence[Chunk], widest: float) -> np.ndarray:
    """Trial line frequencies: a lattice at the finest channel width from the lowest channel, through every group."""
    low = min(c.freq[0] for c in chunks)
    step = min((np.diff(c.freq).min() for c in chunks if c.freq.size > 1), default=1.0)
    k = [
        np.arange(np.ceil((f0 - low) / step - 1e-6), np.floor((f1 - low) / step + 1e-6) + 1)
        for f0, f1 in _reach(chunks, widest)
    ]
    return low + step * np.unique(np.concatenate(k))


def _one_field(data: DataHandler) -> None:
    if len(data.fields) > 1:
        raise ValueError(
            f"data hold fields {data.fields}; the matched filter runs on one field at a time. "
            "Load one with DataHandler(..., fields=[i]) or luv-find --field i."
        )


def _beam(dra, ddec, offset, freqs, dish_diameter) -> np.ndarray:
    """Primary beam of the pointing at ``offset`` on the ``dra x ddec`` grid, ``(n_dra, n_ddec, n_chan)``."""
    dra, ddec = np.asarray(dra)[:, None, None], np.asarray(ddec)[None, :, None]
    return primary_beam_at(dra, ddec, offset, freqs, dish_diameter)


def _same(a, b) -> bool:
    return np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=1e-9, atol=0)


def _check_freqs(freqs: list) -> None:
    if not all(_same(f, freqs[0]) for f in freqs[1:]):
        raise ValueError("the pointings' channel frequencies differ; only one spectral setup can be combined")


def _union_lattice(axes: list, name: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Union of position axes on one lattice anchored at the reference, and each axis' indices in it."""
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
    def templates(self) -> np.ndarray:
        """Every template, a row of :data:`TEMPLATE_KEYS` values, in the order of the S/N's template axes."""
        return np.array(list(product(*(self.axes[k] for k in TEMPLATE_KEYS))), dtype=float)

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
    """Matched-filter search of one pointing or of a mosaic."""

    axes: dict
    freqs: np.ndarray
    snr: np.ndarray
    error: np.ndarray
    coverage: np.ndarray
    jackknife: SearchResult | None = None

    @property
    def flux(self) -> np.ndarray:
        """Peak flux density, Jy: ``snr * error``."""
        return self.snr * self.error

    @property
    def response(self) -> np.ndarray:
        """S/N as ``(n_grid, n_chan)``, one row per :attr:`grid_params` entry."""
        return self.snr.reshape(-1, self.snr.shape[-1])

    @property
    def response_jackknife(self) -> np.ndarray | None:
        return None if self.jackknife is None else self.jackknife.response


def _estimate(w: np.ndarray, wflux: np.ndarray, empty: float) -> tuple[np.ndarray, np.ndarray]:
    """S/N and error from inverse-variance sums ``w`` and ``w flux``, in their place; ``empty`` S/N where ``w == 0``."""
    empties = w <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        np.sqrt(w, out=w)
        np.divide(wflux, w, out=wflux)
        np.divide(1.0, w, out=w)
    np.copyto(wflux, empty, where=empties)
    np.copyto(w, np.nan, where=empties)
    return wflux, w


def _correct(result: SearchResult, data: DataHandler, pb_limit: float) -> SearchResult:
    """``result`` of one pointing corrected for its primary beam in place: the error over the PB, the S/N
    unchanged, NaN below ``pb_limit``."""
    _one_field(data)
    pb = _beam(
        result.axes["dra"], result.axes["ddec"], data.chunks[0].offset, result.freqs, data.metadata.dish_diameter
    )
    keep = pb >= pb_limit
    scale = np.where(keep, pb, np.nan)[:, :, None]
    for r in filter(None, (result, result.jackknife)):
        np.copyto(r.snr, np.nan, where=~keep[:, :, None])
        r.error /= scale
        r.coverage = np.where(keep, pb, 0.0)
    return result


def pb_corrected(result: SearchResult, data: DataHandler, pb_limit: float = 0.2) -> SearchResult:
    """The search of one pointing corrected for its primary beam (a copy; ``result`` is left as it is)."""

    def copy(r):
        return (
            None
            if r is None
            else dataclasses.replace(r, snr=r.snr.copy(), error=r.error.copy(), jackknife=copy(r.jackknife))
        )

    return _correct(copy(result), data, pb_limit)


@dataclass(eq=False)
class _Mosaic:
    """Inverse-variance sums of PB-corrected pointings on the union of their lattices, ``coverage`` squared."""

    axes: dict
    freqs: np.ndarray
    w: np.ndarray
    wflux: np.ndarray
    coverage: np.ndarray
    jackknife: _Mosaic | None

    @classmethod
    def of(cls, r: SearchResult, own: bool = False) -> _Mosaic:
        """The sums of one PB-corrected pointing, ``w = 1 / error^2`` and ``w flux = snr / error``; made in
        place of ``r``'s arrays when it is ``own`` (``r`` is then spent)."""
        jackknife = None if r.jackknife is None else cls.of(r.jackknife, own)
        w, wflux = (r.error, r.snr) if own else (r.error.copy(), r.snr.copy())
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(wflux, w, out=wflux)
            np.divide(1.0, w, out=w)
            np.multiply(w, w, out=w)
        return cls(dict(r.axes), r.freqs, np.nan_to_num(w, copy=False), np.nan_to_num(wflux, copy=False),
                   r.coverage**2, jackknife)  # fmt: skip

    def add(self, other: _Mosaic) -> None:
        """Add another pointing in place; the lattice grows only where it reaches beyond."""
        for k in TEMPLATE_KEYS:
            if not _same(other.axes[k], self.axes[k]):
                raise ValueError(
                    f"the pointings were searched with different {k} templates: {self.axes[k]}, {other.axes[k]}"
                )
        _check_freqs([self.freqs, other.freqs])
        ira, idec = self.grow(other.axes["dra"], other.axes["ddec"])
        for name in ("w", "wflux", "coverage"):
            total, part = getattr(self, name), getattr(other, name)
            for k, i in enumerate(ira):  # a row at a time: no pointing-size gather
                total[i, idec] += part[k]
        if self.jackknife is None or other.jackknife is None:
            self.jackknife = None
        else:
            self.jackknife.add(other.jackknife)

    def grow(self, dra, ddec) -> tuple[np.ndarray, np.ndarray]:
        """Extend the lattice over ``dra`` and ``ddec`` (zero sums there), and their indices in it."""
        new_dra, (i, ira) = _union_lattice([self.axes["dra"], dra], "dra")
        new_ddec, (j, idec) = _union_lattice([self.axes["ddec"], ddec], "ddec")
        for name in ("w", "wflux", "coverage"):
            total = getattr(self, name)
            if total.shape[:2] != (new_dra.size, new_ddec.size):
                grown = np.zeros((new_dra.size, new_ddec.size, *total.shape[2:]))
                grown[np.ix_(i, j)] = total
                setattr(self, name, grown)
        self.axes |= {"dra": new_dra.tolist(), "ddec": new_ddec.tolist()}
        if self.jackknife is not None:
            self.jackknife.grow(dra, ddec)
        return ira, idec

    def result(self) -> SearchResult:
        """The combined search, made in place of the sums (the mosaic is spent)."""
        jackknife = None if self.jackknife is None else self.jackknife.result()
        snr, error = _estimate(self.w, self.wflux, np.nan)
        return SearchResult(self.axes, self.freqs, snr, error, np.sqrt(self.coverage), jackknife)


def combine_pointings(results: Iterable[SearchResult], span: tuple | None = None, own: bool = False) -> SearchResult:
    """Combine the PB-corrected searches of a mosaic's pointings on the union of their grids, one at a time.

    ``span`` (``dra``, ``ddec``), the union's axes if known, is allocated with the first pointing, so the
    sums never grow (and are never copied) as the pointings come; ``own`` results are spent on their sums.
    """
    mosaic = None
    for r in results:
        if mosaic is None:
            mosaic = _Mosaic.of(r, own)
            if span is not None:
                mosaic.grow(*span)
        else:
            mosaic.add(_Mosaic.of(r, own))
    return mosaic.result()


class MatchedFilter(_Grid):
    """S/N of one field over the ``Model.grid`` axes; ``channel_noise``: "measure", None or ``{spw: ChannelNoise}``.

    ``band`` (Hz, low and high) keeps the trial centres there, a chunk of a long spectrum: every window is
    still fitted over all its centres, so its continuum and its masks are those of the whole spectrum.
    """

    def __init__(
        self,
        data: DataHandler,
        mod: Model,
        continuum_order: int | None = CONTINUUM_ORDER,
        channel_noise: str | dict | None = "measure",
        band: tuple[float, float] | None = None,
    ):
        if isinstance(channel_noise, str) and channel_noise != "measure":
            raise ValueError(f'channel_noise must be "measure", None or a dict, got {channel_noise!r}')
        _one_field(data)
        self.data = data
        self.continuum_order = continuum_order
        self.channel_noise = channel_noise
        self.noise: dict[int, ChannelNoise] = {}
        self.axes = self._grid_axes(mod)
        self.lattice = _centres(data.chunks, max(self.axes["width"]))
        lo, hi = band or (-np.inf, np.inf)
        self.freqs = self.lattice[(self.lattice >= lo) & (self.lattice <= hi)]
        if not self.freqs.size:
            raise ValueError(f"no trial centre in the band {lo / 1e9:.4f}-{hi / 1e9:.4f} GHz")
        self.result: SearchResult | None = None

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
    def response(self) -> np.ndarray | None:
        return None if self.result is None else self.result.response

    @property
    def response_jackknife(self) -> np.ndarray | None:
        return None if self.result is None else self.result.response_jackknife

    def run(self, jackknife: bool = False, cores: int | None = None) -> None:
        """Fill ``result`` and, with ``jackknife``, ``result.jackknife`` on the same grid, on ``cores`` cores."""
        limit_cores(cores)
        measure = self.channel_noise == "measure"
        noise = self.data.jackknife() if jackknife or measure else None  # made once, for both
        self.noise = self.data.channel_noise(noise) if measure else dict(self.channel_noise or {})
        self.result = self._search(self.data)
        if jackknife:
            self.result.jackknife = self._search(noise)

    def _search(self, data: DataHandler) -> SearchResult:
        axes = {k: np.asarray(v) for k, v in self.axes.items()}
        bmin, bmaj, pa = self.templates[:: len(axes["width"]), :3].T
        cov = covariance(bmaj * ARCSEC, bmin * ARCSEC, pa)
        # windows that overlap in frequency add by inverse variance, (n_dra, n_ddec, n_template, n_centre)
        layout = (len(axes["dra"]), len(axes["ddec"]), bmin.size * len(axes["width"]))
        w = np.zeros((*layout, self.freqs.size))
        wflux = np.zeros_like(w)
        tol = 1e-3 * np.diff(self.lattice).min() if self.lattice.size > 1 else 1.0
        reach = _reach(data.chunks, max(self.axes["width"]))
        with jax.enable_x64(True):
            for c, (f0, f1) in zip(tqdm(data.chunks, desc="Windows", leave=False), reach, strict=True):
                # the window's centres, a contiguous run of the lattice: slices, so no copies
                full = self.lattice[slice(*np.searchsorted(self.lattice, [f0 - tol, f1 + tol], side="right"))]
                keep = slice(*np.searchsorted(full, [self.freqs[0] - tol, self.freqs[-1] + tol], side="right"))
                if keep.start == keep.stop:
                    continue  # none of its centres is in the band
                at = slice(*np.searchsorted(self.freqs, [full[keep][0] - tol, full[keep][-1] + tol], side="right"))
                for rows, *pair in self._window(c, axes, cov, full):
                    # (n_rows, n_ddec, n_template, n_centre), the templates in the order of the grid
                    shape = (*pair[0].shape[2:4], -1, at.stop - at.start)
                    snr, error = (a[..., keep].transpose(2, 3, 0, 1, 4).reshape(shape) for a in pair)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        w[rows, ..., at] += np.nan_to_num(error**-2.0)
                        wflux[rows, ..., at] += np.nan_to_num(snr / error)
        snr, error = _estimate(w, wflux, 0.0)
        coverage = np.broadcast_to(1.0, (*snr.shape[:2], snr.shape[-1]))
        return SearchResult(dict(self.axes), self.freqs, snr, error, coverage)

    def _window(self, chunk: Chunk, axes: dict, cov: tuple, centres: np.ndarray):
        """S/N and error of one window at ``centres`` per block of ``dra`` rows: (rows, snr, error), the arrays
        ``(n_shape, n_width, n_rows, n_ddec, n_centre)``."""
        rho = jnp.asarray(self.noise[chunk.spw].rho if chunk.spw in self.noise else np.ones(1))
        freq, centres, widths = jnp.asarray(chunk.freq), jnp.asarray(centres), jnp.asarray(axes["width"])
        per_row = SPECTRAL_ARRAYS * 8 * len(cov[0]) * widths.size * len(axes["ddec"]) * centres.size
        start = 0
        for n, collapsed in _blocks(chunk, axes["dra"], axes["ddec"], cov, per_row):
            snr, error = _spectral(*collapsed, freq, centres, widths, rho, order=self.continuum_order)
            yield slice(start, start + n), np.asarray(snr)[:, :, :n], np.asarray(error)[:, :, :n]
            start += n

    def plot_response(self, filename: str | None = None, vline: float | None = None):
        """The best grid point's S/N spectrum, saved to ``filename`` if given (:mod:`luv_finder.plotting`)."""
        from .plotting import response_check

        plots_dir, name = os.path.split(filename) if filename else (None, "filter_response")
        return response_check(self, plots_dir=(plots_dir or ".") if filename else None, name=name, line_ghz=vline)


def grid_axis(cfg: dict | None, key: str, default) -> np.ndarray:
    """One search axis of a grid YAML: a value, a list, or ``{start, stop, step}``; ``default`` if absent."""
    v = (cfg or {}).get(key, default)
    if isinstance(v, dict):
        return np.arange(v["start"], v["stop"], v["step"])
    return np.atleast_1d(v)


def build_grid(data: DataHandler, cfg: dict | None) -> dict:
    """Grid ranges from YAML, positions in the dataset's sky frame (arcsec from the reference)."""
    return _grid(data.metadata, data.chunks[0].offset, data.freqs.min(), cfg)


def _grid(metadata, offset, low: float, cfg: dict | None) -> dict:
    """:func:`build_grid` of a pointing at ``offset`` whose lowest frequency is ``low`` Hz."""
    res = metadata.minresolution()
    cfg = dict(cfg or {})
    unknown = sorted(set(cfg) - {*GRID_KEYS, "pb_limit"})
    if unknown:
        raise ValueError(
            f"grid keys {unknown} are not {[*GRID_KEYS, 'pb_limit']}; total_flux in particular cancels in the "
            "kernel normalisation, so every value gives an identical response"
        )

    half = primary_beam_radius(cfg.get("pb_limit", 0.2), low, metadata.dish_diameter)
    step = res / 2

    def lattice(centre):
        return step * np.arange(np.ceil((centre - half) / step), np.floor((centre + half) / step) + 1)

    dra, ddec = offset
    return {
        "dra": grid_axis(cfg, "dra", lattice(dra)),
        "ddec": grid_axis(cfg, "ddec", lattice(ddec)),
        "bmin": grid_axis(cfg, "bmin", res / 10),
        "bmaj": grid_axis(cfg, "bmaj", res / 10),
        "pa": grid_axis(cfg, "pa", 0.0),
        "width": grid_axis(cfg, "width", WIDTHS),
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
    cores: int | None = None,
    pb_limit: float = 0.2,
    **filter_kwargs,
) -> SearchResult:
    """Search every field of a measurement set or NPZ and combine the pointings on one sky grid as they come."""

    cfg = {**(grid_cfg or {}), "pb_limit": pb_limit}
    grids = [_grid(*pointing(path, field), cfg) for field in fields_in(path)]
    # the union of the pointings' lattices, allocated once (one pointing keeps its own axes)
    span = tuple(_union_lattice([g[k] for g in grids], k)[0] for k in ("dra", "ddec")) if len(grids) > 1 else None

    def pointings():
        for field in fields_in(path):
            data = load(path, [field])
            mf = MatchedFilter(data, grid_model(build_grid(data, cfg)), **filter_kwargs)
            mf.run(jackknife=jackknife, cores=cores)
            result = _correct(mf.result, data, pb_limit)  # in place: the search is ours
            del data, mf  # free this field before loading the next
            yield result

    return combine_pointings(pointings(), span, own=True)
