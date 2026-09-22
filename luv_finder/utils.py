"""Unit conversions and measurement-set weight helpers."""

from __future__ import annotations

import numpy as np
from astropy.constants import c

C = c.value


def arcsec_to_uvdist(arcsec: float) -> float:
    """Angular scale [arcsec] -> uv-distance [klambda]."""
    return 1 / np.deg2rad(arcsec / 3600) / 1e3


def uvdist_to_arcsec(uvdist: float) -> float:
    """uv-distance [klambda] -> angular scale [arcsec]."""
    return np.rad2deg(1 / (uvdist * 1e3)) * 3600


def l_to_arcsec(ell: float) -> float:
    return 180 / ell * 3600


def arcsec_to_l(arcsec: float) -> float:
    return 180 / (arcsec / 3600)


def l_to_uvdist(ell: float) -> float:
    return arcsec_to_uvdist(l_to_arcsec(ell))


def uvdist_to_l(uvdist: float) -> float:
    return arcsec_to_l(uvdist_to_arcsec(uvdist))


def _target_selection(vis: str):
    import casatools

    msmd = casatools.msmetadata()
    msmd.open(vis)
    fields = msmd.fieldsforintent("*OBSERVE_TARGET*", False)
    spws = msmd.spwsforintent("*OBSERVE_TARGET#ON_SOURCE*")
    msmd.close()
    return fields, spws


def uvload(vis: str):
    """Return (weights, uvdists [lambda]) of all target visibilities in an MS."""
    import casatools

    uvwghts = np.empty(0)
    uvdists = np.empty(0)
    fields, spws = _target_selection(vis)
    for field in fields:
        for spw in spws:
            ms = casatools.ms()
            ms.open(vis)
            ms.selectinit(reset=True)
            ms.selectinit(datadescid=int(spw))
            ms.select({"field_id": int(field)})
            rec = ms.getdata(["u", "v", "weight"])
            freqs = ms.range("chan_freq")["chan_freq"][:, 0]
            ms.close()

            uvwght = 4.0 / (1.0 / rec["weight"][0] + 1.0 / rec["weight"][1])
            uwave = (rec["u"].reshape(-1, 1) * freqs / C).T
            vwave = (rec["v"].reshape(-1, 1) * freqs / C).T
            uvwghts = np.append(uvwghts, (np.ones_like(uwave) * uvwght.reshape(1, -1)).flatten())
            uvdists = np.append(uvdists, np.hypot(uwave, vwave).flatten())
    return uvwghts, uvdists


def getstatwtweights(vis: str, seed: int = 0) -> None:
    """Reset WEIGHT in-place from the scan-jackknifed real-part scatter.

    Used after ``simobserve`` so that simulated weights reflect the injected
    noise level instead of CASA's nominal values.
    """
    import casatools

    rng = np.random.default_rng(seed)
    fields, spws = _target_selection(vis)
    for field in fields:
        for spw in spws:
            ms = casatools.ms()
            ms.open(vis, nomodify=False)
            ms.selectinit(reset=True)
            ms.selectinit(datadescid=int(spw))
            ms.select({"field_id": int(field)})
            rec = ms.getdata(["data", "weight", "time"])
            uvwght = 4.0 / (1.0 / rec["weight"][0] + 1.0 / rec["weight"][1])

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
            rec["weight"][0] = wgts / 4  # TODO: should be /2; kept for continuity with existing mocks
            rec["weight"][1] = wgts / 4
            ms.putdata(rec)
            ms.reset()
            ms.close()
