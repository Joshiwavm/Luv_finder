"""Parametric UV-domain source models.

Parameter naming convention: every component attribute is exposed as
``src_{component_index:02d}_{attribute}`` (e.g. ``src_00_nu_center``). The
matched filter parses this key to route values back onto the component.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.constants import c

from .utils import ARCSEC

C_KMS = c.to(u.km / u.s).value
FWHM_TO_SIGMA = 1 / 2.355
#: Line flux per peak flux density and FWHM of a Gaussian line, ``sqrt(2 pi) / 2.355``.
LINE_FLUX = np.sqrt(2 * np.pi) * FWHM_TO_SIGMA
#: Degree of the continuum polynomial fitted with each line, in the search and the fit.
CONTINUUM_ORDER = 1


def line_profile(freq, centre, width, xp=np):
    """Unit-peak Gaussian line at ``freq`` (Hz), centred on ``centre`` (Hz), of FWHM ``width`` (km/s)."""
    return xp.exp(-0.5 * ((freq - centre) / (centre * width * FWHM_TO_SIGMA / C_KMS)) ** 2)


def continuum_basis(n_chan: int, terms: int, xp=np):
    """The continuum's ``terms`` polynomials, powers of the channel index mapped to [-1, 1]: ``(n_chan, terms)``."""
    return xp.linspace(-1.0, 1.0, n_chan)[:, None] ** xp.arange(terms)


def covariance(bmaj, bmin, pa, xp=np):
    """Sky covariance ``(s_ee, s_nn, s_en)`` of a Gaussian of FWHM ``bmaj``, ``bmin`` at ``pa`` deg E of N."""
    return rotated((bmaj * FWHM_TO_SIGMA) ** 2, (bmin * FWHM_TO_SIGMA) ** 2, pa, xp)


def rotated(major, minor, pa, xp=np):
    """``(s_ee, s_nn, s_en)`` of variances ``major`` along ``pa`` deg east of north and ``minor`` across it."""
    # the major axis points along (sin pa, cos pa) in (east, north)
    sin, cos = xp.sin(xp.deg2rad(pa)), xp.cos(xp.deg2rad(pa))
    return major * sin**2 + minor * cos**2, major * cos**2 + minor * sin**2, (major - minor) * sin * cos


def envelope(uw, vw, cov, xp=np):
    """Visibility amplitude at (uw, vw) wavelengths of a unit-flux Gaussian of sky covariance ``cov`` (rad^2)."""
    s_ee, s_nn, s_en = cov
    return xp.exp(-2 * np.pi**2 * (s_ee * uw**2 + s_nn * vw**2 + 2 * s_en * uw * vw))


class Gaussian:
    """2D spatial x 1D spectral Gaussian evaluated directly in the UV plane.

    Position (arcsec), FWHM axes ``bmaj``, ``bmin`` (arcsec) at ``pa`` (deg E of N), line flux ``total_flux``
    (Jy km/s) at ``nu_center`` (Hz) of FWHM ``width`` (km/s).
    """

    positive = True

    def __init__(self, dra=0.0, ddec=0.0, total_flux=1.0, bmin=0.0, bmaj=0.0, pa=0.0, nu_center=0.0, width=100.0):
        self.dra = dra
        self.ddec = ddec
        self.bmin = bmin
        self.bmaj = bmaj
        self.pa = pa
        self.total_flux = total_flux
        self.nu_center = nu_center
        self.width = width
        self.profile = self._uvgauss_1D2D
        self.grid: dict | None = None

    def envelope(self, chunk) -> np.ndarray:
        """Spatial envelope A(u, v): the source's visibility amplitude, 1 at zero spacing, (n_chan, n_row)."""
        return envelope(*chunk.uv_waves(), covariance(self.bmaj * ARCSEC, self.bmin * ARCSEC, self.pa))

    def spectrum(self, freq: np.ndarray) -> np.ndarray:
        """Line profile in Jy at ``freq`` (Hz)."""
        peak = self.total_flux / (self.width * LINE_FLUX)
        return peak * line_profile(freq, self.nu_center, self.width)

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
