"""Spatial collapse of one window's visibilities onto trial positions and source shapes, in JAX."""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

from .data import Chunk
from .model import envelope
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


#: Powers ``(i, j)`` of ``u^i v^j`` in the data moments (degree <= 2) and the weight moments (degree <= 4).
DATA_POWERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
WEIGHT_POWERS = tuple((i, d - i) for d in range(5) for i in range(d, -1, -1))


@jax.jit
def _moments(u, v, X, w_row, flag, freq, dra, ddec, cov):
    """Per channel ``sum E X e^{-i phi} u^i v^j`` and ``sum w E^2 u^i v^j``, u and v in cycles per arcsec."""

    def channel(_, inputs):
        nu, x, f = inputs
        uw, vw = u * nu / C * ARCSEC, v * nu / C * ARCSEC
        shape = envelope(uw, vw, cov, xp=jnp)
        data = shape * x * jnp.exp(-2j * jnp.pi * (uw * dra + vw * ddec))
        weight = jnp.where(f, 0.0, w_row) * shape**2
        up, vp = [jnp.ones_like(uw)], [jnp.ones_like(vw)]
        for _ in range(4):
            up.append(up[-1] * uw)
            vp.append(vp[-1] * vw)
        return None, (
            jnp.stack([data @ (up[i] * vp[j]) for i, j in DATA_POWERS]),
            jnp.stack([weight @ (up[i] * vp[j]) for i, j in WEIGHT_POWERS]),
        )

    return jax.lax.scan(channel, None, (freq, X, flag))[1]


def point_moments(chunk: Chunk, dra: float, ddec: float, cov) -> tuple:
    """:func:`_moments` of one window at one position (arcsec, sky frame) and sky covariance (arcsec^2)."""
    arrays = [getattr(chunk, k) for k in COLLAPSE_ARRAYS]
    return _moments(*arrays, dra - chunk.offset[0], ddec - chunk.offset[1], cov)
