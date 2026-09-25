"""Measurement-set I/O and visibility-domain operations.

CASA (``casatools``) is imported lazily inside the methods that touch a
measurement set, so that the NPZ path (``DataHandler.from_npz``) works in an
environment without CASA.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import astropy.constants as const
import astropy.units as u
import numpy as np

from ._casa import tools

UV_FIELDS = (
    "UVreals",
    "UVimags",
    "uvwghts",
    "uvtimes",
    "uvfreqs",
    "uwaves",
    "vwaves",
)


def _empty_uvdata() -> SimpleNamespace:
    ns = SimpleNamespace(**{k: np.empty(0) for k in UV_FIELDS})
    ns.jacked = False
    return ns


class Metadata:
    """Derived quantities of an observation (primary beam, resolution)."""

    def __init__(self, uvdata: SimpleNamespace, dish_diameter: float | None = None, msfile: str | None = None):
        self.uvdata = uvdata
        self.msfile = msfile
        self._dish_diameter = dish_diameter

    @property
    def dish_diameter(self) -> float:
        """Antenna diameter in metres (read from the MS on first access)."""
        if self._dish_diameter is None:
            if self.msfile is None:
                raise ValueError("dish_diameter unknown and no MS file to read it from")
            msmd = tools().msmetadata()
            msmd.open(self.msfile)
            d = msmd.antennadiameter()
            msmd.close()
            self._dish_diameter = d[next(iter(d))]["value"]
        return self._dish_diameter

    def central_frequency(self) -> float:
        return float(self.uvdata.uvfreqs.mean())

    def primarybeamsize(self, dish_diameter: float | None = None) -> float:
        """Half-power primary beam width in arcsec (1.22 lambda / D)."""
        d = self.dish_diameter if dish_diameter is None else dish_diameter
        wavelength = const.c.value / self.central_frequency()
        return (1.22 * wavelength / d * u.rad).to(u.arcsec).value

    def minresolution(self) -> float:
        """Angular resolution in arcsec from the longest baseline."""
        max_baseline_lambda = np.hypot(self.uvdata.uwaves, self.uvdata.vwaves).max()
        return np.rad2deg(1.0 / max_baseline_lambda) * 3600.0


class DataHandler:
    """Flattened visibilities of a measurement set.

    Attributes
    ----------
    uvdata : SimpleNamespace
        Arrays ``UVreals, UVimags, uvwghts, uvtimes, uvfreqs [Hz],
        uwaves, vwaves [lambda]``, all of length
        ``n_freqs * n_visbs``, channel-major.
    metadata : Metadata
    """

    def __init__(
        self, msfile: str | None = None, uvdata: SimpleNamespace | None = None, dish_diameter: float | None = None
    ):
        self.msfile = msfile
        self.uvdata = uvdata if uvdata is not None else _empty_uvdata()
        if msfile is not None and uvdata is None:
            self.load_data()
        self.metadata = Metadata(self.uvdata, dish_diameter=dish_diameter, msfile=msfile)

    # ------------------------------------------------------------------ shape
    @staticmethod
    def n_freqs(uvdata: SimpleNamespace) -> int:
        return len(np.unique(uvdata.uvfreqs))

    def n_visbs(self, uvdata: SimpleNamespace) -> int:
        return uvdata.UVreals.shape[0] // self.n_freqs(uvdata)

    # -------------------------------------------------------------- CASA I/O
    @staticmethod
    def _target_selection(msfile: str):
        msmd = tools().msmetadata()
        msmd.open(msfile)
        fields = msmd.fieldsforintent("*OBSERVE_TARGET*", False)
        spws = msmd.spwsforintent("*OBSERVE_TARGET*")
        msmd.close()
        return fields, spws

    def load_data(self) -> None:
        """Read all target-field visibilities from ``self.msfile`` into ``uvdata``."""
        mstool = tools().ms
        fields, spws = self._target_selection(self.msfile)
        for field in fields:
            for spw in spws:
                ms = mstool()
                ms.open(self.msfile)
                ms.selectinit(reset=True)
                ms.selectinit(datadescid=int(spw))
                ms.select({"field_id": int(field)})
                rec = ms.getdata(["data", "time", "u", "v", "weight"])
                freqs = ms.range("chan_freq")["chan_freq"][:, 0]
                ms.close()

                uvreal = (rec["data"][0].real + rec["data"][1].real) / 2.0
                uvimag = (rec["data"][0].imag + rec["data"][1].imag) / 2.0
                uvwght = 4.0 / (1.0 / rec["weight"][0] + 1.0 / rec["weight"][1])

                uwave = (rec["u"].reshape(-1, 1) * freqs.reshape(1, -1) / const.c.value).T
                vwave = (rec["v"].reshape(-1, 1) * freqs.reshape(1, -1) / const.c.value).T
                ones = np.ones_like(uwave)

                d = self.uvdata
                d.UVreals = np.append(d.UVreals, uvreal.flatten())
                d.UVimags = np.append(d.UVimags, uvimag.flatten())
                d.uvwghts = np.append(d.uvwghts, (ones * uvwght.reshape(1, -1)).flatten())
                d.uvtimes = np.append(d.uvtimes, (ones * rec["time"].reshape(1, -1)).flatten())
                d.uvfreqs = np.append(d.uvfreqs, (ones * freqs.reshape(-1, 1)).flatten())
                d.uwaves = np.append(d.uwaves, uwave.flatten())
                d.vwaves = np.append(d.vwaves, vwave.flatten())

    # --------------------------------------------------------------- NPZ I/O
    def to_npz(self, path: str) -> None:
        """Save ``uvdata`` plus dish diameter to a compressed NPZ (CASA-free format)."""
        arrays = {k: getattr(self.uvdata, k) for k in UV_FIELDS}
        np.savez_compressed(path, dish_diameter=self.metadata.dish_diameter, **arrays)

    @classmethod
    def from_npz(cls, path: str) -> DataHandler:
        with np.load(path) as f:
            uvdata = SimpleNamespace(**{k: f[k] for k in UV_FIELDS})
            uvdata.jacked = False
            dish = float(f["dish_diameter"])
        return cls(uvdata=uvdata, dish_diameter=dish)

    # ---------------------------------------------------------- operations
    def apply_phase_shift(self, dRA: float, dDec: float, uvdata: SimpleNamespace) -> SimpleNamespace:
        """Phase-shift visibilities by an offset (arcsec); result in ``UVreals_shifted``/``UVimags_shifted``."""
        uvdata = copy.deepcopy(uvdata)
        dra = np.deg2rad(dRA / 3600.0)
        ddec = np.deg2rad(dDec / 3600.0)
        vis = uvdata.UVreals + 1j * uvdata.UVimags
        vis = vis * np.exp(-2j * np.pi * (uvdata.uwaves * dra + uvdata.vwaves * ddec))
        uvdata.UVreals_shifted = vis.real
        uvdata.UVimags_shifted = vis.imag
        return uvdata

    def jackknife(self, uvdata: SimpleNamespace) -> SimpleNamespace:
        """Return a signal-free noise realisation of ``uvdata``.

        Pairs consecutive integrations, differences them and halves the result,
        so any signal constant over a pair cancels while the noise is preserved.
        The output has half the rows.

        The split is by integration, never random: a random sign flip of
        individual visibilities destroys the baseline structure and does not
        give the observation's white-noise level.
        """
        uvdata = copy.deepcopy(uvdata)
        scans, idx = np.unique(uvdata.uvtimes, return_inverse=True)
        neg = (idx % 2).astype(bool)
        pos = ~neg
        if len(scans) % 2 == 1:
            # drop the unpaired last scan
            pos[idx == idx[-1]] = False

        uvdata.UVreals = 0.5 * (uvdata.UVreals[pos] - uvdata.UVreals[neg])
        uvdata.UVimags = 0.5 * (uvdata.UVimags[pos] - uvdata.UVimags[neg])
        for k in ("uvtimes", "uvfreqs", "uvwghts", "uwaves", "vwaves"):
            arr = getattr(uvdata, k)
            setattr(uvdata, k, 0.5 * (arr[pos] + arr[neg]))
        uvdata.jacked = True
        return uvdata
