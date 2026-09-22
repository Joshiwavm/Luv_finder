"""Measurement-set I/O and visibility-domain operations.

CASA (``casatools``) is imported lazily inside the methods that touch a
measurement set, so that the NPZ path (``DataHandler.from_npz``) works in an
environment without CASA.
"""

from __future__ import annotations

import copy
import os
import shutil
from types import SimpleNamespace

import astropy.constants as const
import astropy.units as u
import numpy as np

UV_FIELDS = (
    "UVreals",
    "UVimags",
    "uvdists",
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
            from casatools import msmetadata

            msmd = msmetadata()
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
        max_baseline_lambda = self.uvdata.uvdists.max() * 1e3
        return np.rad2deg(1.0 / max_baseline_lambda) * 3600.0


class DataHandler:
    """Flattened visibilities of a measurement set.

    Attributes
    ----------
    uvdata : SimpleNamespace
        Arrays ``UVreals, UVimags, uvdists [klambda], uvwghts, uvtimes,
        uvfreqs [Hz], uwaves, vwaves [lambda]``, all of length
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

    @staticmethod
    def arcsec_to_uvdist(arcsec: float) -> float:
        """Angular scale in arcsec -> uv-distance in lambda."""
        return 1 / np.deg2rad(arcsec / 3600)

    # -------------------------------------------------------------- CASA I/O
    @staticmethod
    def _target_selection(msfile: str):
        from casatools import msmetadata

        msmd = msmetadata()
        msmd.open(msfile)
        fields = msmd.fieldsforintent("*OBSERVE_TARGET*", False)
        spws = msmd.spwsforintent("*OBSERVE_TARGET*")
        msmd.close()
        return fields, spws

    def load_data(self) -> None:
        """Read all target-field visibilities from ``self.msfile`` into ``uvdata``."""
        from casatools import ms as mstool

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
                d.uvdists = np.append(d.uvdists, np.hypot(uwave, vwave).flatten() * 1e-3)
                d.UVreals = np.append(d.UVreals, uvreal.flatten())
                d.UVimags = np.append(d.UVimags, uvimag.flatten())
                d.uvwghts = np.append(d.uvwghts, (ones * uvwght.reshape(1, -1)).flatten())
                d.uvtimes = np.append(d.uvtimes, (ones * rec["time"].reshape(1, -1)).flatten())
                d.uvfreqs = np.append(d.uvfreqs, (ones * freqs.reshape(-1, 1)).flatten())
                d.uwaves = np.append(d.uwaves, uwave.flatten())
                d.vwaves = np.append(d.vwaves, vwave.flatten())

    def uv_save(self, output_name: str, uvdata: SimpleNamespace | None = None) -> str:
        """Write ``uvdata`` into a copy of ``self.msfile`` named ``<base>_<output_name>.ms``."""
        from casatools import ms as mstool

        uvdata = self.uvdata if uvdata is None else uvdata
        base_dir, base_name = os.path.split(self.msfile)
        stem, ext = os.path.splitext(base_name)
        new_path = os.path.join(base_dir, f"{stem}_{output_name}{ext}")
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        shutil.copytree(self.msfile, new_path)

        fields, spws = self._target_selection(new_path)
        index = 0
        for field in fields:
            for spw in spws:
                ms = mstool()
                ms.open(new_path, nomodify=False)
                ms.selectinit(reset=True)
                ms.selectinit(datadescid=int(spw))
                ms.select({"field_id": int(field)})
                rec = ms.getdata(["data", "time", "weight"])
                n_times = rec["time"].shape[0]
                n_chans = rec["data"].shape[1]
                n = n_times * n_chans

                real = uvdata.UVreals[index : index + n].reshape(n_chans, n_times)
                imag = uvdata.UVimags[index : index + n].reshape(n_chans, n_times)
                wght = uvdata.uvwghts[index : index + n].reshape(n_chans, n_times)
                index += n

                rec["data"][0][:] = real + 1j * imag
                rec["data"][1][:] = real + 1j * imag
                rec["weight"][0] = wght[0, :] / 2
                rec["weight"][1] = wght[0, :] / 2
                ms.putdata(rec)
                ms.reset()
                ms.close()
        return new_path

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

    def jackknife(self, uvdata: SimpleNamespace, mode: str = "scan", seed: int = 42) -> SimpleNamespace:
        """Return a signal-free noise realisation of ``uvdata``.

        Parameters
        ----------
        mode : {"scan", "random"}
            ``"scan"`` pairs consecutive integrations, sign-flips one of each pair
            and averages them (output has half the rows). ``"random"`` sign-flips
            a random half of the visibilities in place (output has the same shape;
            reference: jackknify ``Jack._jack_it``).
        """
        uvdata = copy.deepcopy(uvdata)
        if mode == "random":
            rng = np.random.default_rng(seed)
            flip = np.zeros(uvdata.UVreals.shape[0], dtype=bool)
            flip[: flip.size // 2] = True
            rng.shuffle(flip)
            uvdata.UVreals[flip] *= -1.0
            uvdata.UVimags[flip] *= -1.0
            uvdata.jacked = True
            return uvdata
        if mode != "scan":
            raise ValueError(f"unknown jackknife mode {mode!r}")

        scans, idx = np.unique(uvdata.uvtimes, return_inverse=True)
        neg = (idx % 2).astype(bool)
        pos = ~neg
        if len(scans) % 2 == 1:
            # drop the unpaired last scan
            pos[idx == idx[-1]] = False

        uvdata.UVreals = 0.5 * (uvdata.UVreals[pos] - uvdata.UVreals[neg])
        uvdata.UVimags = 0.5 * (uvdata.UVimags[pos] - uvdata.UVimags[neg])
        for k in ("uvdists", "uvtimes", "uvfreqs", "uvwghts", "uwaves", "vwaves"):
            arr = getattr(uvdata, k)
            setattr(uvdata, k, 0.5 * (arr[pos] + arr[neg]))
        uvdata.jacked = True
        return uvdata
