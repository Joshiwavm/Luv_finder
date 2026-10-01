"""Grid-search matched filter for spectral lines in the UV plane."""

from __future__ import annotations

import copy
import multiprocessing
import os
from itertools import product

import astropy.units as u
import numpy as np
from astropy.constants import c
from tqdm import tqdm

from .data import Chunk, DataHandler
from .model import Model

#: Search axes that a grid may vary. ``total_flux`` is deliberately absent: the
#: kernel normalisation is scale-invariant, so varying it duplicates grid points.
GRID_KEYS = ("dra", "ddec", "bmin", "bmaj", "width")

#: How visibilities are weighted when a channel is collapsed: by noise only
#: ("natural", optimal for a point source) or also by the source envelope A(u, v)
#: ("template", optimal for a resolved source of the trial size).
WEIGHTINGS = ("natural", "template")

#: Channels are processed in blocks whose (channels x rows) complex temporaries stay near this size.
BLOCK_BYTES = 256 * 2**20


def nu_center_func(width: float, uvfreq_min: float) -> float:
    """Line centre placed 4 sigma above the lowest channel, for a given width (km/s)."""
    fmin = u.Quantity(uvfreq_min, u.Hz)
    return (fmin + 4 / 2.355 * ((width * u.km / u.s) / c * fmin).to(u.Hz)).value


def _default_pool(pool: int | None) -> int:
    return pool if pool is not None else max(1, int(multiprocessing.cpu_count() * 0.25))


_FINDER: MatchedFilter | None = None


def _init_worker(finder: MatchedFilter) -> None:
    # Set once per worker, so tasks carry only grid points; with fork the data are shared.
    global _FINDER
    _FINDER = finder


def _worker(params: dict):
    return _FINDER.response_at(params), params


class MatchedFilter:
    """Evaluate a model kernel against the data over a parameter grid.

    ``response`` is in signal-to-noise units: one row per grid point, one column
    per channel, unit variance under the null hypothesis. A line matching the
    template at that grid point shows up with a peak equal to its S/N.

    Parameters
    ----------
    data : DataHandler
        One field; pointings of a mosaic are searched one at a time on a common grid.
    mod : Model
        With one component whose ``grid`` dict defines the search ranges.
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
        self.mod = copy.deepcopy(mod)
        self.weighting = weighting
        self.response = None
        self.grid_params = None
        self.response_jackknife = None
        self.grid_params_jackknife = None

    @staticmethod
    def delay_transform(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Circular cross-correlation via FFT."""
        return np.fft.ifft(np.fft.fft(signal, axis=0) * np.fft.fft(kernel, axis=0), axis=0).real

    def response_at(self, params: dict) -> np.ndarray:
        """Signal-to-noise spectrum of one grid point, the windows concatenated in frequency order.

        The kernel is normalised by ``sqrt(k^T N^-1 k)``, so this is the matched-filter
        statistic ``k^T N^-1 d / sqrt(k^T N^-1 k)``: unit variance under the null, and equal to
        the line's S/N when the template matches. That normalisation also makes the result
        independent of the template amplitude.
        """
        comp = self.mod.component(0)
        for key, value in params.items():
            setattr(comp, key.split("_", 2)[-1], value)
        return np.concatenate([self._window_response(chunk, comp) for chunk in self.data.chunks])

    def _window_response(self, chunk: Chunk, comp) -> np.ndarray:
        # the kernel line sits 4 sigma above the window's lowest channel; the delay transform
        # slides it across the window
        comp.nu_center = nu_center_func(comp._width, chunk.freq[0])
        n_chan = len(chunk.freq)
        weight_sum, signal_sum, kernel_sum = np.zeros(n_chan), np.zeros(n_chan), np.zeros(n_chan)
        step = max(1, BLOCK_BYTES // (16 * len(chunk.u)))
        for start in range(0, n_chan, step):
            sl = slice(start, start + step)
            block = chunk.channels(sl)
            # "natural" weights each visibility by its noise weight only, which is optimal for a
            # point source; "template" also weights by the source envelope A(u, v), which is the
            # full-visibility matched filter for a resolved source. Per channel the collapsed data
            # sum(w t V) / sum(w t^2) has variance 1 / sum(w t^2), with t = 1 or A. Shifted onto
            # its own position the model is the real A(u, v) S(nu), so the kernel has no phase.
            envelope = comp.envelope(block)
            taper = envelope if self.weighting == "template" else 1.0
            weighted = block.w * taper
            weight_sum[sl] = np.sum(weighted * taper, axis=1)
            signal_sum[sl] = np.sum(taper * (block.X * block.phase(comp.dra, comp.ddec)).real, axis=1)
            kernel_sum[sl] = np.sum(weighted * envelope, axis=1) * comp.spectrum(block.freq)

        # fully flagged channels carry no weight and contribute nothing
        ok = weight_sum > 0
        signal = np.divide(signal_sum, weight_sum, out=np.zeros(n_chan), where=ok)
        kernel = kernel_sum / np.sqrt(np.sum(kernel_sum[ok] ** 2 / weight_sum[ok]))

        pad = int(4 * comp.width / np.median(np.diff(chunk.freq)) + 0.5)
        signal_p = np.pad(signal, (0, 2 * pad), mode="reflect")
        kernel_p = np.pad(kernel, (0, 2 * pad), mode="reflect")
        return self.delay_transform(signal_p, kernel_p)[pad : pad + n_chan]

    def _expand_grid(self) -> list[dict]:
        unknown = [k for k in self.mod.grid if k.split("_", 2)[-1] not in GRID_KEYS]
        if unknown:
            raise ValueError(
                f"grid keys {unknown} are not searchable. Searchable keys are {list(GRID_KEYS)}. "
                "total_flux in particular cancels in the kernel normalisation, so varying it "
                "only duplicates grid points."
            )
        keys = list(self.mod.grid)
        values = [np.atleast_1d(self.mod.grid[k]).tolist() for k in keys]
        return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]

    def get_response(self, pool: int | None = None, data: DataHandler | None = None):
        """Return (responses[n_grid, n_freq], grid_params) for ``data`` (default: the data)."""
        original = self.data
        self.data = original if data is None else data
        points = self._expand_grid()
        try:
            with multiprocessing.Pool(_default_pool(pool), initializer=_init_worker, initargs=(self,)) as p:
                results = list(tqdm(p.imap_unordered(_worker, points), total=len(points), desc="Grid search"))
        finally:
            self.data = original
        responses, params = zip(*results, strict=True)
        return np.array(responses), list(params)

    def run(self, pool: int | None = None, jackknife: bool = False) -> None:
        self.response, self.grid_params = self.get_response(pool)
        if jackknife:
            jacked = self.data.jackknife()
            self.response_jackknife, self.grid_params_jackknife = self.get_response(pool, data=jacked)

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
