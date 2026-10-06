import dataclasses

import jax
import numpy as np
import pytest
from test_matchedfilter import _chunk, _metadata

from luv_finder import DataHandler, Gaussian
from luv_finder.imaging import dirty_cube, dirty_maps, mosaic_dirty_maps, pointing_cubes
from luv_finder.kernel import _blocks
from luv_finder.matchedfilter import build_grid
from luv_finder.utils import primary_beam


def test_dirty_cube_of_a_point_source_is_its_flux():
    """Natural weighting: the dirty map of a point source peaks at its flux, in every channel."""
    chunk = _chunk(20, 500, offset=(3.0, 1.0))
    data = DataHandler(
        chunks=[dataclasses.replace(chunk, X=2.0 * np.conj(chunk.phase(5.0, -3.0)))], metadata=_metadata(chunk)
    )
    c = dirty_cube(data, [-10.0, -5.0, 0.0, 5.0], [-3.0, 0.0, 3.0, 6.0])
    assert c.cube.shape == (20, 4, 4)
    assert np.allclose(c.cube[:, 3, 0], 2.0)
    assert np.allclose(c.weight, 500.0)
    assert (c.dra, c.ddec, c.window_sizes) == ([-10.0, -5.0, 0.0, 5.0], [-3.0, 0.0, 3.0, 6.0], [20])
    assert np.array_equal(c.offset, (3.0, 1.0)) and np.array_equal(c.freqs, chunk.freq)


def test_dirty_cube_is_the_exact_dft():
    """The NUFFT cube equals the direct sum over visibilities, with flags, offsets and two windows."""
    r = np.random.default_rng(5)
    chunks = []
    for spw, nchan in ((0, 30), (1, 20)):
        chunk = _chunk(nchan, 300, seed=spw, spw=spw, offset=(4.0, -2.0), freq0=40e9 + spw * 2e9)
        flag = r.random(chunk.flag.shape) < 0.1
        flag[3] = True
        vis = r.normal(size=flag.shape) + 1j * r.normal(size=flag.shape)
        chunks.append(dataclasses.replace(chunk, X=np.where(flag, 0, vis), flag=flag))
    data = DataHandler(chunks=chunks, metadata=_metadata(chunks[0]))
    dra, ddec = [-3.0, 0.0, 3.0, 6.0], [1.0, 3.5, 6.0]
    c = dirty_cube(data, dra, ddec)
    point = (np.zeros(1),) * 3
    with jax.enable_x64(True):
        collapsed = [
            np.asarray(a)
            for chunk in chunks
            for _, (sig, w) in _blocks(chunk, dra, ddec, point)
            for a in (sig[:, 0], w[:, 0])
        ]
    sig, w = np.concatenate(collapsed[::2]), np.concatenate(collapsed[1::2])[:, None, None]
    assert np.allclose(c.cube, np.divide(sig, w, out=np.zeros_like(sig), where=w > 0), rtol=0, atol=1e-12)
    assert c.window_sizes == [30, 20]
    with pytest.raises(ValueError, match="uniformly spaced"):
        dirty_cube(data, [0.0, 1.0, 3.0], ddec)


def test_pointing_cubes_are_each_fields_cube_on_its_grid(two_field_npz):
    cubes = pointing_cubes(str(two_field_npz))
    for field, cube in enumerate(cubes):
        data = DataHandler.from_npz(str(two_field_npz), [field])
        grid = build_grid(data, {"pb_limit": 0.2})
        assert np.allclose(cube.cube, dirty_cube(data, grid["dra"], grid["ddec"]).cube, rtol=0, atol=1e-12)
    assert cubes[1].offset[0] - cubes[0].offset[0] == 10.0


def test_dirty_maps_separate_continuum_and_line():
    """The continuum map shows the continuum source; the moment-8 the line, not the continuum."""
    chunk = _chunk(40, 800)
    continuum = 0.5 * np.conj(chunk.phase(20.0, -15.0))
    line = Gaussian(dra=-10.0, ddec=8.0, nu_center=chunk.freq[20], width=300.0, total_flux=4.0).profile(chunk)
    data = DataHandler(chunks=[dataclasses.replace(chunk, X=continuum + line)], metadata=_metadata(chunk))
    axis = np.arange(-30.0, 31.0, 5.0)
    moment8, cont, sigma = dirty_maps(data, axis, axis)
    assert sigma == pytest.approx(1 / np.sqrt(40 * 800))
    assert np.unravel_index(np.argmax(cont), cont.shape) == (10, 3)  # (+20, -15)
    assert np.unravel_index(np.argmax(moment8), moment8.shape) == (4, 8)  # (-10, +8)
    assert moment8[10, 3] < 0.2 * moment8.max()


def test_mosaic_dirty_maps_restore_a_continuum_source():
    """Two pointings see a continuum source through their beams; the linear mosaic restores its flux."""
    pos, flux, ivar, cubes = (20.0, 0.0), 2.0, 0.0, []
    pointings = (
        ((0.0, 0.0), 500, [-20.0, 0.0, 20.0], [0.0, 20.0]),
        ((60.0, 0.0), 300, np.arange(20.0, 241.0, 20.0).tolist(), [-20.0, 0.0]),
    )
    for field, (offset, nvis, dra, ddec) in enumerate(pointings):
        chunk = _chunk(20, nvis, seed=field, field=field, offset=offset)
        pb = primary_beam(np.hypot(pos[0] - offset[0], pos[1] - offset[1]), chunk.freq, 12.0)
        X = flux * pb[:, None] * np.conj(chunk.phase(*pos))
        data = DataHandler(chunks=[dataclasses.replace(chunk, X=X)], metadata=_metadata(chunk))
        cubes.append(dirty_cube(data, dra, ddec))
        ivar += np.sum(pb**2 * nvis)
    dra, ddec, moment8, continuum, sigma = mosaic_dirty_maps(cubes, pb_limit=0.2)
    assert dra.tolist() == np.arange(-20.0, 241.0, 20.0).tolist()
    assert ddec.tolist() == [-20.0, 0.0, 20.0]
    assert continuum[2, 1] == pytest.approx(flux, rel=1e-9)
    assert sigma[2, 1] == pytest.approx(1 / np.sqrt(ivar), rel=1e-9)
    assert abs(moment8[2, 1]) < 1e-6  # no line, so the continuum is all subtracted
    # (-20, -20) is on neither grid; (240, 0) only on the second, beyond its pb_limit
    for i, j in ((0, 0), (13, 1)):
        assert np.isnan(moment8[i, j]) and np.isnan(continuum[i, j]) and np.isnan(sigma[i, j])
    assert np.all(np.isfinite(continuum[:5, 1]))
