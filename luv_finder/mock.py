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
from ._casa import tasks, tools

# Rough 12 m-array anchor (50 uJy in 27 min) used only to pick a starting flux.
# It is wrong for other arrays, so ``calibrate_snr`` measures the achieved S/N
# from the simulated data and corrects it.
_REF_RMS_JY = 0.05e-3
_REF_MIN = 27.0

#: Antenna configurations are resolved from CASA's own data directory, so the
#: repository does not vendor .cfg files. Cycle 13 needs casarundata >= 2026.02.19.
SIMMOS = "alma/simmos"


def resolve_antenna_config(name: str) -> str:
    """Absolute path to a CASA antenna configuration, by bare name or path."""
    if os.path.sep in name and os.path.exists(name):
        return name
    repo = tools().ctsys.resolve(SIMMOS)
    path = os.path.join(repo, name)
    if not os.path.exists(path):
        available = sorted(os.path.basename(p) for p in glob.glob(os.path.join(repo, "*cycle*.cfg")))
        cycles = sorted({n.split("cycle")[1].split(".")[0] for n in available if "cycle" in n}, key=int)
        raise FileNotFoundError(
            f"antenna configuration {name!r} not found in {repo}. "
            f"Cycles available in this casarundata: {', '.join(cycles) or 'none'}. "
            "Update it with: python -c 'from casaconfig import data_update; data_update()'"
        )
    return path


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
    alma_config : str
        Bare name of a CASA antenna configuration, e.g. ``alma.cycle13.3.cfg``,
        resolved from CASA's own data directory.
    calibrate_snr : bool
        Re-simulate once with the flux rescaled so that each line's declared
        ``snr`` is its achieved matched-filter S/N. Doubles the runtime.
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
        ptg_file=None,
        alma_config="alma.cycle13.3.cfg",
        output_folder="output/ms_files",
        support_folder="support",
        seed=242,
        calibrate_snr=True,
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
        self.ptg_file = ptg_file or os.path.join(support_folder, "ptgfile.txt")
        self.calibrate_snr = calibrate_snr
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
        simobserve = tasks().simobserve
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
            antennalist=resolve_antenna_config(self.alma_config),
            seed=self.seed,
            graphics="none",
        )
        cfg = os.path.basename(self.alma_config)
        self.ms_noiseless = f"{self.project_name}/{self.project_name}." + cfg.replace("cfg", "ms")
        self.ms_noisy = f"{self.project_name}/{self.project_name}." + cfg.replace("cfg", "noisy.ms")
        utils.getstatwtweights(self.ms_noisy)

    def run_imaging(self) -> None:
        casatasks = tasks()
        tclean, exportfits = casatasks.tclean, casatasks.exportfits

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

    def achieved_snr(self) -> list[float]:
        """Matched-filter S/N of each line source in the simulated data.

        Uses the noiseless visibilities as the signal and the noisy weights as the
        noise model, so this is the expected S/N rather than one noisy draw.
        """
        from .data import DataHandler

        clean = DataHandler(self.ms_noiseless)
        noisy = DataHandler(self.ms_noisy)
        nf, nv = clean.n_freqs(clean.uvdata), clean.n_visbs(clean.uvdata)
        out = []
        for src in self.sources:
            if "line" not in src:
                continue
            dra, ddec = src.get("position", (0.0, 0.0))
            # model convention flips the sign of dra relative to the image
            shifted = clean.apply_phase_shift(-dra, ddec, clean.uvdata)
            weights = noisy.apply_phase_shift(-dra, ddec, noisy.uvdata).uvwghts
            s_ch, w_ch = np.average(
                shifted.UVreals_shifted.reshape(nf, nv),
                weights=weights.reshape(nf, nv),
                axis=1,
                returned=True,
            )
            out.append(float(np.sqrt(np.sum(s_ch**2 * w_ch))))
        return out

    def _calibrate(self) -> list[float]:
        """Rescale the cube so each line's declared ``snr`` is its achieved S/N.

        S/N is linear in flux and the noise draw is seeded, so a single
        multiplicative correction is exact.
        """
        achieved = self.achieved_snr()
        wanted = [src["line"]["snr"] for src in self.sources if "line" in src]
        if not achieved or min(achieved) <= 0:
            return achieved
        # one global factor: all line sources share the same noise realisation
        factor = float(np.mean([w / a for w, a in zip(wanted, achieved, strict=True)]))
        print(f"calibrating: achieved S/N {[round(a, 2) for a in achieved]} -> scaling flux by {factor:.3f}")
        self.cube *= factor
        self.save_cube()
        self.simulate_observation()
        return self.achieved_snr()

    def run_all(self, plots_dir: str = "plots") -> str:
        self.create_cube()
        self.save_cube()
        self.simulate_observation()
        if self.calibrate_snr:
            final = self._calibrate()
            print(f"achieved matched-filter S/N: {[round(a, 2) for a in final]}")
        self.run_imaging()
        self.plot_results(plots_dir)
        return self.move_output()
