"""Spatial collapse of one window's visibilities onto trial positions and source shapes, in JAX."""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

from .data import Chunk
from .model import covariance, envelope
from .utils import ARCSEC, BLOCK_BYTES, C

#: The :class:`~luv_finder.data.Chunk` arrays :func:`_collapse` reads, in its argument order.
COLLAPSE_ARRAYS = ("u", "v", "X", "w_row", "flag", "freq")


def default_cores() -> int:
    n = os.cpu_count() or 1
    return max(1, min(n // 4, n - 2))


def limit_cores(cores: int | None = None) -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: cores or default_cores()])


@jax.jit
def _collapse(u, v, X, w_row, flag, freq, dfreq, dra, ddec, cov):
    """Scan the channels: ``sig (n_chan, n_shape, n_dra, n_ddec)`` and ``W (n_chan, n_shape)``."""
    k0 = -2j * jnp.pi * freq[0] / C * ARCSEC
    dk = -2j * jnp.pi * dfreq / C * ARCSEC
    pu, du = jnp.exp(k0 * dra[:, None] * u), jnp.exp(dk * dra[:, None] * u)
    pv, dv = jnp.exp(k0 * ddec[:, None] * v), jnp.exp(dk * ddec[:, None] * v)
    cov = tuple(c[:, None] for c in cov)

    def channel(phases, inputs):
        pu, pv = phases
        nu, x, f = inputs
        w = jnp.where(f, 0.0, w_row)
        shape = envelope(u * nu / C, v * nu / C, cov, xp=jnp)
        sig = ((pu[None] * (shape * x)[:, None, :]) @ pv.T).real
        return (pu * du, pv * dv), (sig, shape**2 @ w)

    return jax.lax.scan(channel, (pu, pv), (freq, X, flag))[1]


def _blocks(chunk: Chunk, dra, ddec, cov: tuple):
    """Run :func:`_collapse` on one window in blocks of ``dra`` rows; yields (rows, (sig, weight))."""
    dfreq = np.diff(chunk.freq)
    if dfreq.size and not np.allclose(dfreq, dfreq[0], rtol=1e-9):
        raise ValueError(f"channels of field {chunk.field} spw {chunk.spw} are not uniformly spaced")
    arrays = [jnp.asarray(getattr(chunk, k)) for k in COLLAPSE_ARRAYS]
    dra = np.asarray(dra) - chunk.offset[0]
    ddec = jnp.asarray(np.asarray(ddec) - chunk.offset[1])
    cov = tuple(jnp.asarray(c) for c in cov)
    block = max(1, min(len(dra), BLOCK_BYTES // (16 * (2 + len(cov[0])) * len(chunk.u))))
    step = dfreq[0] if dfreq.size else 0.0
    for start in range(0, len(dra), block):
        rows = dra[start : start + block]
        # pad the last block to the same shape, so the compiled scan is reused
        padded = jnp.asarray(np.pad(rows, (0, block - len(rows)), mode="edge"))
        yield len(rows), _collapse(*arrays, step, padded, ddec, cov)


def on_device(chunk: Chunk) -> Chunk:
    """``chunk`` with the arrays :func:`_collapse` reads held by JAX in float64."""
    with jax.enable_x64(True):
        return dataclasses.replace(chunk, **{k: jnp.asarray(getattr(chunk, k)) for k in COLLAPSE_ARRAYS})


def template_spectrum(chunk: Chunk, dra: float, ddec: float, bmaj: float, bmin: float, pa: float):
    """Template-weighted spectrum of one window at one position and source shape."""
    cov = tuple(np.atleast_1d(c) for c in covariance(bmaj * ARCSEC, bmin * ARCSEC, float(pa)))
    with jax.enable_x64(True):
        [(_, (sig, weight))] = _blocks(chunk, [dra], [ddec], cov)
        return np.asarray(sig)[:, 0, 0, 0], np.asarray(weight)[:, 0]
