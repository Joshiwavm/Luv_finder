"""Natural-weighted dirty cubes and moment-8/continuum maps on a regular grid, by NUFFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax_finufft import nufft1

from .data import DataHandler, fields_in, load
from .kernel import limit_cores
from .matchedfilter import _beam, _check_freqs, _one_field, _union_lattice, build_grid
from .utils import ARCSEC, C


@dataclass(frozen=True, eq=False)
class DirtyCube:
    """Dirty cube of one pointing, ``(n_chan, n_dra, n_ddec)`` Jy/beam; channel ``i`` has noise ``weight[i]**-0.5``."""

    cube: np.ndarray
    weight: np.ndarray
    dra: list[float]
    ddec: list[float]
    offset: np.ndarray
    freqs: np.ndarray
    dish_diameter: float
    window_sizes: list[int]


def _regular(axis) -> tuple[float, float, int]:
    a = np.asarray(axis, dtype=float)
    step = (a[-1] - a[0]) / (a.size - 1) if a.size > 1 else 1.0
    if not np.allclose(a, a[0] + step * np.arange(a.size), rtol=0, atol=1e-9 * max(abs(step), 1.0)):
        raise ValueError("a dirty cube needs uniformly spaced dra and ddec axes")
    return a[0], step, a.size


@partial(jax.jit, static_argnames=("n_dra", "n_ddec"))
def _window(u, v, X, freq, ra0, step_ra, n_dra, dec0, step_dec, n_ddec):
    """``sum_j Re[X e^{-i phi}]`` per channel on the grid, ``(n_chan, n_dra, n_ddec)``."""
    centre_ra, centre_dec = ra0 + step_ra * (n_dra // 2), dec0 + step_dec * (n_ddec // 2)

    def channel(_, inputs):
        nu, x = inputs
        uw, vw = u * nu / C * ARCSEC, v * nu / C * ARCSEC
        source = x * jnp.exp(-2j * jnp.pi * (uw * centre_ra + vw * centre_dec))
        px = jnp.mod(2 * jnp.pi * uw * step_ra + jnp.pi, 2 * jnp.pi) - jnp.pi
        py = jnp.mod(2 * jnp.pi * vw * step_dec + jnp.pi, 2 * jnp.pi) - jnp.pi
        return None, nufft1((n_dra, n_ddec), source, px, py, iflag=-1, eps=1e-12).real

    return jax.lax.scan(channel, None, (freq, X))[1]


def dirty_cube(data: DataHandler, dra, ddec, cores: int | None = None) -> DirtyCube:
    """Natural-weighted dirty cube of one field on the regular ``dra x ddec`` grid (arcsec, sky frame)."""
    _one_field(data)
    limit_cores(cores)
    (ra0, step_ra, n_dra), (dec0, step_dec, n_ddec) = _regular(dra), _regular(ddec)
    cube, weights = [], []
    with jax.enable_x64(True):
        for c in data.chunks:
            ra, dec = ra0 - c.offset[0], dec0 - c.offset[1]
            arrays = (
                jnp.asarray(c.u, dtype=float),
                jnp.asarray(c.v, dtype=float),
                jnp.asarray(c.X, dtype=complex),
                jnp.asarray(c.freq, dtype=float),
            )
            sig = np.asarray(_window(*arrays, ra, step_ra, n_dra, dec, step_dec, n_ddec))
            weight = c.w.sum(axis=1)
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
    """Moment-8 (S/N, each window's continuum removed), continuum and its noise of a cube in Jy/beam."""

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
    """Moment-8 (S/N), continuum (Jy/beam) and continuum noise of one field on the ``dra x ddec`` grid."""
    c = dirty_cube(data, dra, ddec, cores)
    moment8, continuum, sigma = _moments(c.cube, c.weight[:, None, None], c.window_sizes)
    return moment8, continuum, sigma.item()


def pointing_cubes(path: str, grid_cfg: dict | None = None, pb_limit: float = 0.2, cores: int | None = None):
    """The dirty cube of every field of a measurement set or NPZ, each on its search grid."""
    cubes = []
    for field in fields_in(path):
        data = load(path, [field])
        grid = build_grid(data, {**(grid_cfg or {}), "pb_limit": pb_limit})
        cubes.append(dirty_cube(data, grid["dra"], grid["ddec"], cores))
    return cubes


def mosaic_dirty_maps(cubes: Sequence[DirtyCube], pb_limit: float = 0.2):
    """Linear mosaic of the pointings' cubes (PB >= ``pb_limit``): ``(dra, ddec, moment8, continuum, sigma)``."""
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
