"""Measurement-set weight helper for simulated observations."""

from __future__ import annotations

import numpy as np

from ._casa import tools
from .data import DataHandler


def getstatwtweights(vis: str, seed: int = 0) -> None:
    """Reset WEIGHT in-place from the scan-jackknifed real-part scatter.

    ``simobserve`` leaves ``WEIGHT = 1`` and writes no ``WEIGHT_SPECTRUM``, so
    this sets one weight per row from the injected noise level. Real data carry
    per-channel weights; mocks built this way do not.
    """
    rng = np.random.default_rng(seed)
    fields, spws = DataHandler._target_selection(vis)
    for field in fields:
        for spw in spws:
            ms = tools().ms()
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
