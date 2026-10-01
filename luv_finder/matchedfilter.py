"""Grid-search matched filter for spectral lines in the UV plane, evaluated with JAX.

For one spectral window the response at every grid point comes from two collapses:

1. Spatial, per channel: the weighted visibilities phase-shifted onto each trial position and
   summed, ``sig = sum_j t Re[X e^{-i phi}]``, with the per-channel weight ``W = sum_j w t^2``
   and template amplitude ``a = sum_j w t A``. ``t`` is 1 ("natural") or the source envelope
   A ("template"). On a ``dra x ddec`` grid the phase factorises, so all positions of a channel
   are one complex matrix product; across uniformly spaced channels the phase factors follow
   by recurrence instead of new exponentials.
2. Spectral, per lag: the Gaussian line profile ``S_i`` centred on channel ``i``,
   ``sum S_i q sig / sqrt(sum S_i^2 q^2 W)`` with ``q = a / W``. Each lag is normalised over the
   channels it covers, so the response has unit variance under the null at every channel,
   window edges and flagged channels included, and peaks at the line's S/N on the line's own
   channel.

All of it runs in float64 (``jax.enable_x64``), scoped to the search.
"""

from __future__ import annotations

import os
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .data import ARCSEC, C, Chunk, DataHandler
from .model import C_KMS, FWHM_TO_SIGMA, Model, envelope

#: Search axes, in the order of the response rows. ``total_flux`` is deliberately absent: the
#: kernel normalisation is scale-invariant, so varying it duplicates grid points.
GRID_KEYS = ("dra", "ddec", "bmin", "bmaj", "pa", "width")

#: How visibilities are weighted when a channel is collapsed: by noise only
#: ("natural", optimal for a point source) or also by the source envelope A(u, v)
#: ("template", optimal for a resolved source of the trial size).
WEIGHTINGS = ("natural", "template")

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


@jax.jit
def _spectral(sig, weight, amplitude, freq, widths):
    """Line templates centred on every channel: response ``(n_shape, n_width, n_dra, n_ddec, n_chan)``."""
    ok = weight > 0
    q = jnp.where(ok, amplitude / jnp.where(ok, weight, 1.0), 0.0)
    sigma = freq[None, :, None] * widths[:, None, None] * FWHM_TO_SIGMA / C_KMS
    profile = jnp.exp(-0.5 * ((freq[None, None, :] - freq[None, :, None]) / sigma) ** 2)
    num = jnp.einsum("wim,msab->swabi", profile, q[:, :, None, None] * sig)
    var = jnp.einsum("wim,ms->swi", profile**2, q**2 * weight)
    norm = jnp.where(var > 0, 1 / jnp.sqrt(jnp.where(var > 0, var, 1.0)), 0.0)
    return num * norm[:, :, None, None, :]


class MatchedFilter:
    """Evaluate a model kernel against the data over a parameter grid.

    ``response`` is in signal-to-noise units: one row per grid point (``grid_params``, the
    product of the :data:`GRID_KEYS` axes in that order), one column per channel, unit
    variance under the null hypothesis. A line matching the template at that grid point peaks
    on its own channel with a value equal to its S/N.

    Parameters
    ----------
    data : DataHandler
        One field; pointings of a mosaic are searched one at a time on a common grid.
    mod : Model
        With one component whose ``grid`` dict sets the search axes; an axis the grid does
        not name takes the component's own value.
    weighting : {"natural", "template"}
        See :data:`WEIGHTINGS`. With "natural" the trial source size cancels.
    """

    def __init__(self, data: DataHandler, mod: Model, weighting: str = "natural"):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        if len(data.fields) > 1:
            raise ValueError(
                f"data hold fields {data.fields}; the matched filter runs on one field at a time. "
                "Load one with DataHandler(..., fields=[i]) or luv-find --field i."
            )
        self.data = data
        self.weighting = weighting
        self.axes = self._grid_axes(mod)
        self.grid_params = [
            {f"src_00_{k}": v for k, v in zip(GRID_KEYS, combo, strict=True)} for combo in product(*self.axes.values())
        ]
        self.response = None
        self.response_jackknife = None

    @staticmethod
    def _grid_axes(mod: Model) -> dict[str, np.ndarray]:
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

    def run(self, jackknife: bool = False, cores: int | None = None) -> None:
        """Fill ``response`` and, with ``jackknife``, ``response_jackknife`` on the same grid.

        The process is first pinned to ``cores`` cores (see :func:`limit_cores`).
        """
        limit_cores(cores)
        self.response = self._search(self.data)
        if jackknife:
            self.response_jackknife = self._search(self.data.jackknife())

    def _search(self, data: DataHandler) -> np.ndarray:
        axes = {k: np.asarray(v) for k, v in self.axes.items()}
        bmin, bmaj, pa = (np.asarray(s) for s in zip(*product(axes["bmin"], axes["bmaj"], axes["pa"]), strict=True))
        shapes = (bmaj * ARCSEC, bmin * ARCSEC, pa)
        with jax.enable_x64(True):
            windows = [self._window(c, axes, shapes) for c in tqdm(data.chunks, desc="Windows", leave=False)]
        response = np.concatenate(windows, axis=-1)  # (n_shape, n_width, n_dra, n_ddec, n_chan)
        n = [len(axes[k]) for k in GRID_KEYS]
        response = response.reshape(n[2], n[3], n[4], n[5], n[0], n[1], -1)
        return response.transpose(4, 5, 0, 1, 2, 3, 6).reshape(-1, response.shape[-1])

    def _window(self, chunk: Chunk, axes: dict, shapes: tuple) -> np.ndarray:
        dfreq = np.diff(chunk.freq)
        if dfreq.size and not np.allclose(dfreq, dfreq[0], rtol=1e-9):
            raise ValueError(f"channels of field {chunk.field} spw {chunk.spw} are not uniformly spaced")
        arrays = [jnp.asarray(a) for a in (chunk.u, chunk.v, chunk.X, chunk.w_row, chunk.flag, chunk.freq)]
        dra, ddec = axes["dra"] - chunk.offset[0], jnp.asarray(axes["ddec"] - chunk.offset[1])
        n_complex = 2 + (len(shapes[0]) if self.weighting == "template" else 1)
        block = max(1, min(len(dra), BLOCK_BYTES // (16 * n_complex * len(chunk.u))))
        out = []
        for start in range(0, len(dra), block):
            rows = dra[start : start + block]
            # pad the last block to the same shape, so the compiled scan is reused
            padded = jnp.asarray(np.pad(rows, (0, block - len(rows)), mode="edge"))
            sig, weight, amplitude = _collapse(
                *arrays,
                dfreq[0] if dfreq.size else 0.0,
                padded,
                ddec,
                *(jnp.asarray(s) for s in shapes),
                template=self.weighting == "template",
            )
            response = _spectral(sig, weight, amplitude, arrays[-1], jnp.asarray(axes["width"]))
            out.append(np.asarray(response)[:, :, : len(rows)])
        return np.concatenate(out, axis=2)

    @property
    def best_index(self) -> int:
        return int(np.argmax(np.max(self.response, axis=1)))

    @property
    def best_params(self) -> dict:
        return self.grid_params[self.best_index]

    def frequencies(self) -> np.ndarray:
        """Channel frequencies in GHz."""
        return self.data.freqs / 1e9

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
