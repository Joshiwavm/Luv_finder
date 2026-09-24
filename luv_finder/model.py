"""Parametric UV-domain source models.

Parameter naming convention: every component attribute is exposed as
``src_{component_index:02d}_{attribute}`` (e.g. ``src_00_nu_center``). The
matched filter parses this key to route values back onto the component.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.constants import c

C_KMS = c.to(u.km / u.s).value
FWHM_TO_SIGMA = 1 / 2.355


class Gaussian:
    """2D spatial x 1D spectral Gaussian evaluated directly in the UV plane.

    Parameters
    ----------
    dra, ddec : float
        Offsets from the phase centre, arcsec.
    total_flux : float
        Integrated line flux, Jy km/s.
    bmin, bmaj : float
        Source axes (sigma), arcsec.
    nu_center : float
        Line centre, Hz.
    width : float
        Line FWHM, km/s.
    """

    positive = True

    def __init__(self, dra=0.0, ddec=0.0, total_flux=1.0, bmin=0.0, bmaj=0.0, nu_center=0.0, width=100.0):
        self._dra = dra
        self._ddec = ddec
        self._bmin = bmin
        self._bmaj = bmaj
        self.total_flux = total_flux
        self.nu_center = nu_center
        self._width = width
        self.profile = self._uvgauss_1D2D
        self.grid: dict | None = None

    # arcsec <-> rad accessors -------------------------------------------------
    @property
    def dra(self):
        return np.deg2rad(self._dra / 3600)

    @dra.setter
    def dra(self, v):
        self._dra = v

    @property
    def ddec(self):
        return np.deg2rad(self._ddec / 3600)

    @ddec.setter
    def ddec(self, v):
        self._ddec = v

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

    def envelope(self, uvdata: SimpleNamespace) -> np.ndarray:
        """Spatial envelope A(u, v): the source's visibility amplitude, 1 at zero spacing."""
        return np.exp(-2 * np.pi**2 * ((self.bmaj * uvdata.uwaves) ** 2 + (self.bmin * uvdata.vwaves) ** 2))

    def _uvgauss_1D2D(self, uvdata: SimpleNamespace) -> SimpleNamespace:
        uvdata = copy.deepcopy(uvdata)
        flux_hz = self.total_flux * self.nu_center / C_KMS
        amp = flux_hz / (self.width * np.sqrt(2 * np.pi))
        spectral = amp * np.exp(-0.5 * ((uvdata.uvfreqs - self.nu_center) / self.width) ** 2)
        spatial = self.envelope(uvdata) * np.exp(2j * np.pi * (uvdata.uwaves * self.dra + uvdata.vwaves * self.ddec))
        uvdata.UVreals = spatial.real * spectral
        uvdata.UVimags = spatial.imag * spectral
        return uvdata


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
