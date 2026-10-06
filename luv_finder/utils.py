"""Shared constants, the analytic ALMA primary beam and the mock-weight helper.

The primary beam follows the ALMA Technical Handbook: the measured FWHM of the 12 m and 7 m antennas is
about 1.13 lambda / D, and CASA models the beam as an Airy pattern scaled so that its FWHM matches.
"""

from __future__ import annotations

import astropy.constants as const
import numpy as np
from scipy.optimize import brentq
from scipy.special import j1, jn_zeros

from ._casa import tools

C = const.c.value
ARCSEC = np.deg2rad(1 / 3600)

#: Arrays processed in blocks stay near this size in memory.
BLOCK_BYTES = 2**31

#: Measured primary-beam FWHM in units of lambda / D (ALMA Technical Handbook).
FWHM_FACTOR = 1.13

#: First zero of J1, the edge of the Airy main lobe.
_X_NULL = jn_zeros(1, 1)[0]


def _airy(x):
    """Airy power pattern ``(2 J1(x) / x)**2``, equal to 1 at ``x = 0``."""
    x = np.asarray(x, dtype=float)
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 1.0, (2 * j1(safe) / safe) ** 2)


#: Argument at which the Airy pattern drops to one half.
X_HALF = brentq(lambda x: _airy(x) - 0.5, 1.0, _X_NULL)


def primary_beam_fwhm(freq_hz, dish_diameter, fwhm_factor=FWHM_FACTOR):
    return fwhm_factor * (C / np.asarray(freq_hz, dtype=float)) / dish_diameter / ARCSEC


def primary_beam(offset_arcsec, freq_hz, dish_diameter, fwhm_factor=FWHM_FACTOR):
    """Primary-beam attenuation in [0, 1] at an angular offset from the pointing centre."""
    fwhm = primary_beam_fwhm(freq_hz, dish_diameter, fwhm_factor)
    return _airy(2 * X_HALF * np.asarray(offset_arcsec, dtype=float) / fwhm)


def primary_beam_radius(level, freq_hz, dish_diameter, fwhm_factor=FWHM_FACTOR):
    """Offset in arcsec at which the main lobe falls to ``level`` (``0 < level <= 1``)."""
    if not 0 < level <= 1:
        raise ValueError(f"level must be in (0, 1], got {level}")
    x = 0.0 if level == 1 else brentq(lambda x: _airy(x) - level, 0.0, _X_NULL)
    return x / (2 * X_HALF) * primary_beam_fwhm(freq_hz, dish_diameter, fwhm_factor)


def getstatwtweights(vis: str, seed: int = 0) -> None:
    """Reset WEIGHT in-place from the scan-jackknifed real-part scatter.

    ``simobserve`` leaves ``WEIGHT = 1`` and writes no ``WEIGHT_SPECTRUM``, so
    this sets one weight per row from the injected noise level. Real data carry
    per-channel weights; mocks built this way do not.
    """
    from .data import pair_weight, target_selection

    rng = np.random.default_rng(seed)
    fields, spws = target_selection(vis)
    for field in fields:
        for spw in spws:
            ms = tools().ms()
            ms.open(vis, nomodify=False)
            ms.selectinit(reset=True)
            ms.selectinit(datadescid=int(spw))
            ms.select({"field_id": int(field)})
            rec = ms.getdata(["data", "weight", "time"])
            uvwght = pair_weight(rec["weight"][0], rec["weight"][1])

            uvreal = (rec["data"][0].real + rec["data"][1].real) / 2.0
            uvtime = np.ones_like(uvreal) * rec["time"].reshape(1, -1)
            scans, idx = np.unique(uvtime, return_inverse=True)
            neg = (idx % 2).astype(bool)
            pos = ~neg
            if len(scans) % 2 == 1:
                pos[idx == idx[0, -1]] = False
            white_noise = np.nanstd(0.5 * (uvreal[pos] - uvreal[neg]))
            noise = rng.normal(white_noise, white_noise / np.sqrt(uvwght.size), size=uvwght.shape)
            wgts = 1 / noise**2
            rec["weight"][0] = wgts / 4
            rec["weight"][1] = wgts / 4
            ms.putdata(rec)
            ms.reset()
            ms.close()
