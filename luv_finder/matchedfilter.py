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
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .data import Chunk, DataHandler, fields_in, load
from .kernel import _blocks, limit_cores
from .model import C_KMS, FWHM_TO_SIGMA, Gaussian, Model, covariance
from .utils import ARCSEC, primary_beam, primary_beam_radius

TEMPLATE_KEYS = ("bmin", "bmaj", "pa", "width")
GRID_KEYS = ("dra", "ddec", *TEMPLATE_KEYS)


@partial(jax.jit, static_argnames="order")
def _spectral(sig, weight, freq, widths, rho, order=None):
    """S/N ``(n_shape, n_width, n_dra, n_ddec, n_chan)`` and peak error ``(n_shape, n_width, n_chan)``."""
    spread = freq[None, :, None] * widths[:, None, None] * FWHM_TO_SIGMA / C_KMS
    profile = jnp.exp(-0.5 * ((freq[None, None, :] - freq[None, :, None]) / spread) ** 2)
    num = jnp.einsum("wim,msab->swabi", profile, sig)
    var = jnp.einsum("wim,ms->swi", profile**2, weight)
    h = jnp.broadcast_to(profile, (weight.shape[1], *profile.shape))
    covered = var > 0
    if order is not None:
        basis = jnp.linspace(-1.0, 1.0, freq.size)[:, None] ** jnp.arange(order + 1)
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
    error = jnp.where(covered, jnp.sqrt(var_c) / jnp.where(covered, var, 1.0), jnp.nan)
    return num * jnp.where(covered, 1 / jnp.sqrt(var_c), 0.0)[:, :, None, None, :], error


def _one_field(data: DataHandler) -> None:
    if len(data.fields) > 1:
        raise ValueError(
            f"data hold fields {data.fields}; the matched filter runs on one field at a time. "
            "Load one with DataHandler(..., fields=[i]) or luv-find --field i."
        )


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
    """Correct the search of one pointing for its primary beam."""
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
    """Combine the PB-corrected searches of a mosaic's pointings on the union of their grids."""
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


class MatchedFilter(_Grid):
    """S/N of one field over the ``Model.grid`` axes; ``channel_correlation``: "measure", None or ``{spw: rho}``."""

    def __init__(
        self,
        data: DataHandler,
        mod: Model,
        continuum_order: int | None = 2,
        channel_correlation: str | dict | None = "measure",
    ):
        if isinstance(channel_correlation, str) and channel_correlation != "measure":
            raise ValueError(f'channel_correlation must be "measure", None or a dict, got {channel_correlation!r}')
        _one_field(data)
        self.data = data
        self.continuum_order = continuum_order
        self.channel_correlation = channel_correlation
        self.correlation: dict[int, np.ndarray] = {}
        self.axes = self._grid_axes(mod)
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
    def freqs(self) -> np.ndarray:
        """Channel frequencies in Hz."""
        return self.data.freqs

    @property
    def response(self) -> np.ndarray | None:
        return None if self.result is None else self.result.response

    @property
    def response_jackknife(self) -> np.ndarray | None:
        return None if self.result is None else self.result.response_jackknife

    def run(self, jackknife: bool = False, cores: int | None = None) -> None:
        """Fill ``result`` and, with ``jackknife``, ``result.jackknife`` on the same grid, on ``cores`` cores."""
        limit_cores(cores)
        if self.channel_correlation == "measure":
            self.correlation = self.data.channel_correlation()
        else:
            self.correlation = {k: np.asarray(v, dtype=float) for k, v in (self.channel_correlation or {}).items()}
        self.result = self._search(self.data)
        if jackknife:
            self.result.jackknife = self._search(self.data.jackknife())

    def _search(self, data: DataHandler) -> SearchResult:
        axes = {k: np.asarray(v) for k, v in self.axes.items()}
        bmin, bmaj, pa = (np.asarray(s) for s in zip(*product(axes["bmin"], axes["bmaj"], axes["pa"]), strict=True))
        cov = covariance(bmaj * ARCSEC, bmin * ARCSEC, pa)
        with jax.enable_x64(True):
            windows = [self._window(c, axes, cov) for c in tqdm(data.chunks, desc="Windows", leave=False)]
        snr, error = zip(*windows, strict=True)
        # (n_shape, n_width, n_dra, n_ddec, n_chan) and (n_shape, n_width, n_chan)
        snr, error = np.concatenate(snr, axis=-1), np.concatenate(error, axis=-1)
        snr = snr.transpose(2, 3, 0, 1, 4).reshape(*snr.shape[2:4], -1, snr.shape[-1])
        error = np.broadcast_to(error.reshape(-1, error.shape[-1]), snr.shape)
        coverage = np.broadcast_to(1.0, (*snr.shape[:2], snr.shape[-1]))
        return SearchResult(dict(self.axes), data.freqs, snr, snr * error, error, coverage)

    def _window(self, chunk: Chunk, axes: dict, cov: tuple) -> tuple:
        """S/N and error of one window."""
        snr = []
        widths, freq = jnp.asarray(axes["width"]), jnp.asarray(chunk.freq)
        rho = jnp.asarray(self.correlation.get(chunk.spw, np.ones(1)))
        for n, collapsed in _blocks(chunk, axes["dra"], axes["ddec"], cov):
            block, error = _spectral(*collapsed, freq, widths, rho, order=self.continuum_order)
            snr.append(np.asarray(block)[:, :, :n])
        return np.concatenate(snr, axis=2), np.asarray(error)

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
    """Grid ranges from YAML, positions in the dataset's sky frame (arcsec from the reference)."""
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

    half = primary_beam_radius(cfg.get("pb_limit", 0.2), data.freqs.min(), data.metadata.dish_diameter)
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
    cores: int | None = None,
    pb_limit: float = 0.2,
    **filter_kwargs,
) -> SearchResult:
    """Search every field of a measurement set or NPZ and combine the pointings on one sky grid."""
    results = []
    for field in fields_in(path):
        data = load(path, [field])
        grid = build_grid(data, {**(grid_cfg or {}), "pb_limit": pb_limit})
        mf = MatchedFilter(data, grid_model(grid), **filter_kwargs)
        mf.run(jackknife=jackknife, cores=cores)
        results.append(pb_corrected(mf.result, data, pb_limit))
        del data, mf  # free this field's visibilities before loading the next
    return combine_pointings(results) if len(results) > 1 else results[0]
