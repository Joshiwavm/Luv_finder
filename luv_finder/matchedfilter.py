"""Grid-search matched filter for spectral lines in the UV plane."""

from __future__ import annotations

import copy
import functools
import multiprocessing
from itertools import product

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from tqdm import tqdm

from .data import DataHandler
from .model import Model

# Empirical correction of the response amplitude (kernel normalisation bias).
# TODO: derive analytically; see the padding/normalisation experiments in the git history of Test.ipynb.
RESPONSE_SCALE = 0.9


def nu_center_func(width: float, uvfreq_min: float) -> float:
    """Line centre placed 4 sigma above the lowest channel, for a given width (km/s)."""
    fmin = u.Quantity(uvfreq_min, u.Hz)
    return (fmin + 4 / 2.355 * ((width * u.km / u.s) / c * fmin).to(u.Hz)).value


def _default_pool(pool: int | None) -> int:
    return pool if pool is not None else max(1, int(multiprocessing.cpu_count() * 0.25))


def _grid_point_response(args):
    """Response spectrum of one grid point (module-level for multiprocessing)."""
    params, finder, n_vis, n_freq = args
    data = finder.data
    model_uv = finder._get_model(params)
    model_uv = data.apply_phase_shift(params["src_00_dra"], params["src_00_ddec"], model_uv)
    data_uv = data.apply_phase_shift(params["src_00_dra"], params["src_00_ddec"], data.uvdata)

    signal = data_uv.UVreals_shifted.reshape(n_freq, n_vis)
    kernel = model_uv.UVreals_shifted.reshape(n_freq, n_vis)
    weight = data_uv.uvwghts.reshape(n_freq, n_vis)

    signal_mean, weight_mean = np.average(signal, weights=weight, axis=1, returned=True)
    kernel_mean = np.average(kernel, weights=weight, axis=1)
    kernel_norm = kernel_mean * weight_mean / np.sqrt(kernel_mean @ (kernel_mean * weight_mean))

    width_hz = (4 / 2.355 * (params["src_00_width"] * u.km / u.s) / c * params["src_00_nu_center"] * u.Hz).to(u.Hz)
    df = np.median(np.diff(np.unique(data.uvdata.uvfreqs))) * u.Hz
    pad = int(width_hz / df + 0.5)

    signal_p = np.pad(signal_mean, (0, 2 * pad), mode="reflect")
    kernel_p = np.pad(kernel_norm, (0, 2 * pad), mode="reflect")
    response = MatchedFilter.delay_transform(signal_p, kernel_p) * RESPONSE_SCALE
    return response[pad:-pad], params


class MatchedFilter:
    """Evaluate a model kernel against the data over a parameter grid.

    Parameters
    ----------
    data : DataHandler
    mod : Model
        With one component whose ``grid`` dict defines the search ranges.
    """

    def __init__(self, data: DataHandler, mod: Model):
        self.data = copy.deepcopy(data)
        self.mod = copy.deepcopy(mod)
        self.response = None
        self.grid_params = None
        self.response_jackknife = None
        self.grid_params_jackknife = None

    @staticmethod
    def delay_transform(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Circular cross-correlation via FFT."""
        return np.fft.ifft(np.fft.fft(signal, axis=0) * np.fft.fft(kernel, axis=0), axis=0).real

    def _get_model(self, theta: dict):
        comp = self.mod.component(0)
        for key, value in theta.items():
            setattr(comp, key.split("_", 2)[-1], value)
        return comp.profile(self.data.uvdata)

    def _expand_grid(self) -> list[dict]:
        variable, fixed, funcs = {}, {}, {}
        for key, val in self.mod.grid.items():
            if callable(val):
                funcs[key] = val
            else:
                arr = np.atleast_1d(val)
                (variable if arr.size > 1 else fixed)[key] = arr if arr.size > 1 else arr.item()

        keys = list(variable)
        points = []
        for combo in product(*(variable[k] for k in keys)):
            p = dict(fixed, **dict(zip(keys, combo, strict=True)))
            for key, fn in funcs.items():
                raw = fn.func if isinstance(fn, functools.partial) else fn
                nargs = raw.__code__.co_argcount
                if isinstance(fn, functools.partial):
                    nargs -= len(fn.args) + len(fn.keywords or {})
                if nargs == 1:
                    width = next((v for k, v in p.items() if k.endswith("width")), None)
                    if width is None:
                        raise ValueError(f"no width parameter available for derived grid key {key}")
                    p[key] = fn(width)
                else:
                    p[key] = fn(p)
            points.append(p)
        return points

    def get_response(self, pool: int | None = None, uvdata=None):
        """Return (responses[n_grid, n_freq], grid_params) for ``uvdata`` (default: the data)."""
        original = self.data.uvdata
        self.data.uvdata = original if uvdata is None else uvdata
        n_vis = self.data.n_visbs(self.data.uvdata)
        n_freq = self.data.n_freqs(self.data.uvdata)
        args = [(p, self, n_vis, n_freq) for p in self._expand_grid()]
        with multiprocessing.Pool(_default_pool(pool)) as p:
            results = list(tqdm(p.imap_unordered(_grid_point_response, args), total=len(args), desc="Grid search"))
        self.data.uvdata = original
        responses, params = zip(*results, strict=True)
        return np.array(responses), list(params)

    def run(self, pool: int | None = None, jackknife: bool = False, jackknife_mode: str = "scan") -> None:
        self.response, self.grid_params = self.get_response(pool)
        if jackknife:
            jacked = self.data.jackknife(self.data.uvdata, mode=jackknife_mode)
            self.response_jackknife, self.grid_params_jackknife = self.get_response(pool, uvdata=jacked)

    @property
    def best_index(self) -> int:
        return int(np.argmax(np.max(self.response, axis=1)))

    @property
    def best_params(self) -> dict:
        return self.grid_params[self.best_index]

    def frequencies(self) -> np.ndarray:
        """Channel frequencies in GHz."""
        n_vis = self.data.n_visbs(self.data.uvdata)
        n_freq = self.data.n_freqs(self.data.uvdata)
        return self.data.uvdata.uvfreqs.reshape(n_freq, n_vis)[:, 0] / 1e9

    def getmodel(self):
        """Model visibilities at the best grid point."""
        return self._get_model(self.best_params)

    def plot_response(
        self, filename: str = "plots/filter_response.png", show: bool = False, vline: float | None = None
    ):
        i = self.best_index
        freqs = self.frequencies()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axhline(0, ls="--", color="gray")
        if vline is not None:
            ax.axvline(vline, ls="--", color="gray")
        ax.plot(freqs, self.response[i], lw=2, label="Data")
        if self.response_jackknife is not None:
            ax.plot(freqs, self.response_jackknife[i], lw=2, ls=":", label="Jackknife")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Response")
        ax.set_title("Best grid point")
        ax.legend()
        text = "\n".join(
            f"{k.split('_', 2)[-1]}={v:.2f}" for k, v in self.best_params.items() if not k.endswith("nu_center")
        )
        ax.text(
            1.05,
            0.5,
            text,
            transform=ax.transAxes,
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()
        fig.savefig(filename)
        if show:
            plt.show()
        plt.close(fig)
