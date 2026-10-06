"""Parametric UV-domain source models.

Parameter naming convention: every component attribute is exposed as
``src_{component_index:02d}_{attribute}`` (e.g. ``src_00_nu_center``). The
matched filter parses this key to route values back onto the component.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.constants import c

C_KMS = c.to(u.km / u.s).value
FWHM_TO_SIGMA = 1 / 2.355


def covariance(bmaj, bmin, pa, xp=np):
    """Sky covariance ``(s_ee, s_nn, s_en)`` of a Gaussian of axes (sigma) ``bmaj``, ``bmin`` at ``pa`` deg E of N."""
    # the major axis points along (sin pa, cos pa) in (east, north)
    sin, cos = xp.sin(xp.deg2rad(pa)), xp.cos(xp.deg2rad(pa))
    return bmaj**2 * sin**2 + bmin**2 * cos**2, bmaj**2 * cos**2 + bmin**2 * sin**2, (bmaj**2 - bmin**2) * sin * cos


def envelope(uw, vw, cov, xp=np):
    """Visibility amplitude at (uw, vw) wavelengths of a unit-flux Gaussian of sky covariance ``cov`` (rad^2)."""
    s_ee, s_nn, s_en = cov
    return xp.exp(-2 * np.pi**2 * (s_ee * uw**2 + s_nn * vw**2 + 2 * s_en * uw * vw))


class Gaussian:
    """2D spatial x 1D spectral Gaussian evaluated directly in the UV plane."""

    positive = True

    def __init__(self, dra=0.0, ddec=0.0, total_flux=1.0, bmin=0.0, bmaj=0.0, pa=0.0, nu_center=0.0, width=100.0):
        self.dra = dra
        self.ddec = ddec
        self._bmin = bmin
        self._bmaj = bmaj
        self.pa = pa
        self.total_flux = total_flux
        self.nu_center = nu_center
        self._width = width
        self.profile = self._uvgauss_1D2D
        self.grid: dict | None = None

    # arcsec <-> rad accessors -------------------------------------------------
    @property
    def bmin(self):
        return np.deg2rad(self._bmin / 3600)

    @bmin.setter
    def bmin(self, v):
        self._bmin = v

    @property
    def bmaj(self):
        return np.deg2rad(self._bmaj / 3600)

    @bmaj.setter
    def bmaj(self, v):
        self._bmaj = v

    @property
    def width(self):
        """Spectral sigma in Hz."""
        return self.nu_center * self._width * FWHM_TO_SIGMA / C_KMS

    @width.setter
    def width(self, v):
        self._width = v

    def envelope(self, chunk) -> np.ndarray:
        """Spatial envelope A(u, v): the source's visibility amplitude, 1 at zero spacing, (n_chan, n_row)."""
        return envelope(*chunk.uv_waves(), covariance(self.bmaj, self.bmin, self.pa))

    def spectrum(self, freq: np.ndarray) -> np.ndarray:
        """Line profile in Jy at ``freq`` (Hz)."""
        flux_hz = self.total_flux * self.nu_center / C_KMS
        amp = flux_hz / (self.width * np.sqrt(2 * np.pi))
        return amp * np.exp(-0.5 * ((freq - self.nu_center) / self.width) ** 2)

    def _uvgauss_1D2D(self, chunk) -> np.ndarray:
        """Model visibilities on the uv coverage of ``chunk``, (n_chan, n_row) complex."""
        return self.spectrum(chunk.freq)[:, None] * self.envelope(chunk) * np.conj(chunk.phase(self.dra, self.ddec))


class Model:
    """Container of source components with a flat, prefixed parameter list."""

    Gaussian = Gaussian

    def __init__(self):
        self.ncomp = 0
        self.params: list[str] = []
        self.profile: list = []
        self.type: list[str] = []
        self.grid: dict = {}

    def addcomponent(self, comp) -> None:
        prefix = f"src_{self.ncomp:02d}_"
        for key in comp.__dict__:
            if key in ("profile", "grid"):
                continue
            self.params.append(prefix + key.lstrip("_"))
        if comp.grid:
            for key, rng in comp.grid.items():
                self.grid[prefix + key] = rng
        self.profile.append(comp.profile)
        self.type.append(comp.__class__.__name__)
        setattr(self, self.type[-1].lower(), comp)
        self.ncomp += 1

    def component(self, idx: int = 0):
        return getattr(self, self.type[idx].lower())
