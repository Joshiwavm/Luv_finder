"""Synthetic ALMA observations of spectral-line sources via CASA ``simobserve``."""

from __future__ import annotations

import glob
import os
import shutil

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.constants import c
from astropy.io import fits
from astropy.modeling import models

from . import utils

# ALMA Cycle 10 12m-array reference: 50 uJy continuum rms in 27 min (used for the
# sensitivity <-> integration time scaling below).
_REF_RMS_JY = 0.05e-3
_REF_MIN = 27.0


class MockObservation:
    """Build a line cube, observe it with ``simobserve``, image it, and move the products.

    Parameters
    ----------
    cube_shape : (nchan, ny, nx)
    dv, freq_center, cell : str with units, e.g. "100km/s", "40GHz", "0.25arcsec"
    sensitivity : str or float, per-channel rms in Jy (ignored if integration_time given)
    integration_time : str with units, e.g. "8min"
    sources : list of dict
        ``position`` (dra, ddec) arcsec; optional ``axis_min``/``axis_maj`` arcsec;
        ``line`` {width km/s, mean GHz, snr}; ``continuum`` {snr}.
    """

    def __init__(
        self,
        cube_shape=(30, 256, 256),
        dv="100km/s",
        freq_center="40GHz",
        cell="0.25arcsec",
        sensitivity=None,
        integration_time=None,
        fits_filename=None,
        sources=None,
        direction="J2000 00h00m00.10 -40d00m00.00",
        ptg_file="configs/alma/ptgfile.txt",
        alma_config="configs/alma/alma.cycle10.2.cfg",
        output_folder="output/ms_files",
        support_folder="support",
        seed=242,
    ):
        if not sources:
            raise ValueError("at least one source is required")
        self.cube_shape = tuple(cube_shape)
        self.dv = dv
        self.freq_center = freq_center
        self.cell = cell
        self.fits_filename = fits_filename
        self.sources = sources
        self.direction = direction
        self.ptg_file = ptg_file
        self.alma_config = alma_config
        self.output_folder = output_folder
        self.support_folder = support_folder
        self.seed = seed
        self.cube = None
        self.ms_noiseless = None
        self.ms_noisy = None

        if integration_time is not None:
            self.integration_time = u.Quantity(integration_time).to(u.min).value
            self.sensitivity = self.estimate_sensitivity(self.integration_time)
        elif sensitivity is not None:
            self.sensitivity = u.Quantity(sensitivity).to(u.Jy).value if isinstance(sensitivity, str) else sensitivity
            self.integration_time = self.compute_totaltime()
        else:
            raise ValueError("either sensitivity or integration_time must be given")

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> MockObservation:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg.update(overrides)
        return cls(**cfg)

    # ------------------------------------------------------------ frequencies
    @property
    def nchan(self) -> int:
        return self.cube_shape[0]

    def compute_df(self) -> u.Quantity:
        return (u.Quantity(self.freq_center) * u.Quantity(self.dv) / c).to(u.GHz)

    def _freq_axis(self):
        """(df [GHz], f_start [GHz], dv [km/s])."""
        df = self.compute_df().value
        f_start = u.Quantity(self.freq_center).to(u.GHz).value - self.nchan / 2 * df
        return df, f_start, u.Quantity(self.dv).to(u.km / u.s).value

    def compute_totaltime(self) -> float:
        return (_REF_RMS_JY / (self.sensitivity / np.sqrt(self.nchan))) ** 2 * _REF_MIN

    def estimate_sensitivity(self, totaltime_min: float) -> float:
        return np.sqrt(_REF_MIN * _REF_RMS_JY**2 * self.nchan / totaltime_min)

    @property
    def project_name(self) -> str:
        if self.fits_filename is None:
            parts = []
            for src in self.sources:
                if "line" in src:
                    parts.append(f"line{src['line']['snr']}")
                if "continuum" in src:
                    parts.append(f"cont{src['continuum']['snr']}")
            self.fits_filename = os.path.join(self.support_folder, "_".join(parts) + "_input.fits")
        return os.path.basename(self.fits_filename).split("_input")[0]

    # ------------------------------------------------------------------ cube
    def _source_pixels(self, source):
        nchan, ny, nx = self.cube_shape
        cell = u.Quantity(self.cell).to(u.arcsec).value
        dra, ddec = source.get("position", (0.0, 0.0))
        pos_y = ny / 2 + ddec / cell
        pos_x = nx / 2 + dra / cell
        sig_min = source.get("axis_min", cell) / cell
        sig_maj = source.get("axis_maj", cell) / cell
        return pos_y, pos_x, sig_maj, sig_min

    def create_cube(self) -> np.ndarray:
        nchan, ny, nx = self.cube_shape
        self.cube = np.zeros(self.cube_shape)
        chans = np.arange(nchan)
        Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        df, f_start, dv = self._freq_axis()

        for src in self.sources:
            pos_y, pos_x, sig_maj, sig_min = self._source_pixels(src)
            spatial = models.Gaussian2D(
                amplitude=1 / (2 * np.pi * sig_maj * sig_min),
                x_mean=pos_y,
                y_mean=pos_x,
                x_stddev=sig_maj,
                y_stddev=sig_min,
            )(Y, X)
            if "line" in src:
                width_chan = src["line"]["width"] / dv
                mean_chan = (src["line"]["mean"] - f_start) / df
                flux_line = src["line"]["snr"] * self.sensitivity * np.sqrt(width_chan / 2.355 * 4)
                amp = flux_line / (width_chan / 2.355 * np.sqrt(2 * np.pi))
                spec = models.Gaussian1D(amplitude=amp, mean=mean_chan, stddev=width_chan / 2.355)(chans)
                self.cube += spec[:, None, None] * spatial[None]
                print(f"source {src['position']}: line flux {flux_line * dv:.3e} Jy km/s")
            if "continuum" in src:
                flux_cont = src["continuum"]["snr"] * self.sensitivity / np.sqrt(nchan)
                self.cube += flux_cont * spatial[None]
                print(f"source {src['position']}: continuum {flux_cont:.3e} Jy/chan")
        return self.cube

    def save_cube(self) -> str:
        if self.cube is None:
            raise RuntimeError("call create_cube() first")
        _ = self.project_name  # resolves fits_filename
        os.makedirs(os.path.dirname(self.fits_filename), exist_ok=True)
        fits.writeto(self.fits_filename, self.cube, overwrite=True)
        return self.fits_filename

    # -------------------------------------------------------------- simulate
    def simulate_observation(self) -> None:
        from casatasks import simobserve

        if self.fits_filename is None:
            raise RuntimeError("call save_cube() first")
        os.makedirs(os.path.dirname(self.ptg_file) or ".", exist_ok=True)
        with open(self.ptg_file, "w") as f:
            f.write(self.direction)

        simobserve(
            project=self.project_name,
            skymodel=self.fits_filename,
            setpointings=False,
            ptgfile=self.ptg_file,
            overwrite=True,
            integration="10s",
            totaltime=f"{self.integration_time}min",
            inbright="",
            comp_nchan=self.nchan,
            indirection=self.direction,
            incell=self.cell,
            incenter=str(self.freq_center),
            inwidth=str(self.compute_df()),
            antennalist=self.alma_config,
            seed=self.seed,
            graphics="none",
        )
        cfg = os.path.basename(self.alma_config)
        self.ms_noiseless = f"{self.project_name}/{self.project_name}." + cfg.replace("cfg", "ms")
        self.ms_noisy = f"{self.project_name}/{self.project_name}." + cfg.replace("cfg", "noisy.ms")
        utils.getstatwtweights(self.ms_noisy)

    def run_imaging(self) -> None:
        from casatasks import exportfits, tclean

        for vis in (self.ms_noiseless, self.ms_noisy):
            if vis is None:
                raise RuntimeError("call simulate_observation() first")
            name = vis.replace(".ms", ".im")
            tclean(
                vis=vis,
                imagename=name,
                niter=0,
                imsize=self.cube_shape[1],
                cell=self.cell,
                gridder="standard",
                weighting="natural",
                specmode="cube",
                parallel=False,
            )
            exportfits(imagename=name + ".image", fitsimage=name + ".fits", overwrite=True)

    # ------------------------------------------------------------------ plots
    def plot_results(self, plots_dir: str = "plots", show: bool = False) -> None:
        """Line / continuum maps (noiseless, noisy, SNR) for every source."""
        os.makedirs(plots_dir, exist_ok=True)
        cube_clean = fits.getdata(self.ms_noiseless.replace(".ms", ".im.fits"))[0]
        cube_noisy = fits.getdata(self.ms_noisy.replace(".ms", ".im.fits"))[0]
        nchan, ny, nx = cube_clean.shape
        df, f_start, dv = self._freq_axis()

        def triptych(clean, noisy, tag):
            std = np.nanstd(noisy - clean)
            snr = clean / std if std else clean
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, img, title in zip(
                axes, (clean, noisy, snr), ("noiseless", "noisy", f"SNR (peak {np.nanmax(snr):.1f})"), strict=True
            ):
                im = ax.imshow(img, origin="lower")
                ax.set_title(f"{tag} {title}")
                fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"{self.project_name}_{tag}.png"))
            if show:
                plt.show()
            plt.close(fig)

        for i, src in enumerate(self.sources):
            if "line" in src:
                sig = src["line"]["width"] / dv / 2.355
                mean = (src["line"]["mean"] - f_start) / df
                lo, hi = int(max(0, mean - 3 * sig)), int(min(nchan, mean + 3 * sig + 1))
                triptych(cube_clean[lo:hi].sum(0), cube_noisy[lo:hi].sum(0), f"src{i}_line")
            if "continuum" in src:
                pos_y, pos_x, sig_maj, sig_min = self._source_pixels(src)
                h = int(3 * max(sig_maj, sig_min))
                sl = (
                    slice(max(0, int(pos_y - h)), min(ny, int(pos_y + h))),
                    slice(max(0, int(pos_x - h)), min(nx, int(pos_x + h))),
                )
                triptych(cube_clean.mean(0)[sl], cube_noisy.mean(0)[sl], f"src{i}_cont")

    # ----------------------------------------------------------------- output
    def move_output(self) -> str:
        dest = os.path.join(self.output_folder, self.project_name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.makedirs(self.output_folder, exist_ok=True)
        shutil.move(self.project_name, dest)
        for log in glob.glob("*.log") + glob.glob("*.last"):
            os.remove(log)
        self.ms_noiseless = os.path.join(self.output_folder, self.ms_noiseless)
        self.ms_noisy = os.path.join(self.output_folder, self.ms_noisy)
        return dest

    def run_all(self, plots_dir: str = "plots") -> str:
        self.create_cube()
        self.save_cube()
        self.simulate_observation()
        self.run_imaging()
        self.plot_results(plots_dir)
        return self.move_output()
