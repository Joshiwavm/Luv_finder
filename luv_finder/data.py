"""Measurement-set I/O and visibility-domain operations.

Visibilities are held per (field, spectral window) in a :class:`Chunk`: the weighted
visibilities ``X = w * V`` and their flags as ``(n_chan, n_row)`` arrays, and everything that
is constant along an axis (``u``, ``v``, time and weight per row, frequency per channel) once.

Positions are offsets in one sky frame per dataset: arcsec east and north of
:attr:`Metadata.ref`, the model's (dra, ddec) convention. Each chunk carries its field's phase
centre in that frame, so the same (dra, ddec) means the same sky position in every pointing.

CASA (``casatools``) is imported lazily inside the functions that read a measurement set, so
the NPZ path (:meth:`DataHandler.from_npz`) works in an environment without CASA.
"""

from __future__ import annotations

import dataclasses
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import astropy.constants as const
import numpy as np

from ._casa import tools

C = const.c.value
ARCSEC = np.deg2rad(1 / 3600)

#: Arrays stored per chunk in an NPZ, under ``{field}_{spw}_{name}``.
CHUNK_ARRAYS = ("offset", "freq", "u", "v", "time", "baseline", "X", "w_row", "flag")

#: Target size of one block of rows read from a measurement set, in visibility-channels.
READ_BLOCK = 25_000_000


def pair_weight(a, b) -> np.ndarray:
    """Inverse variance of ``(x_a +- x_b) / 2`` from the weights of ``x_a`` and ``x_b``; 0 if either is."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    s = a + b
    return np.divide(4 * a * b, s, out=np.zeros(np.broadcast(a, b).shape), where=s > 0)


def stokes_i(data: np.ndarray, weight: np.ndarray, flag: np.ndarray):
    """Average the parallel hands ``[0, -1]`` of MS columns into (vis, w_row, flag).

    ``data`` and ``flag`` are ``(n_corr, n_chan, n_row)``, ``weight`` is the per-row ``WEIGHT``
    column ``(n_corr, n_row)``. A channel is flagged if either hand is.
    """
    return 0.5 * (data[0] + data[-1]), pair_weight(weight[0], weight[-1]), flag[0] | flag[-1]


def sky_offset(direction, ref) -> np.ndarray:
    """(east, north) offset in arcsec of ``direction`` on the tangent plane at ``ref``, both (RA, Dec) in rad."""
    (ra, dec), (ra0, dec0) = direction, ref
    east = np.cos(dec) * np.sin(ra - ra0)
    north = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
    return np.array([east, north]) / ARCSEC


@dataclass(frozen=True, eq=False)
class Chunk:
    """Visibilities of one (field, spectral window), channel-major.

    ``X = w * V`` is zero where flagged, so a weighted sum over visibilities is a plain sum over
    ``X``. Every row has some unflagged channel and ``w_row > 0``.

    Attributes
    ----------
    offset : (2,) arcsec, the field's phase centre in the dataset's sky frame
    freq : (n_chan,) Hz, ascending
    u, v : (n_row,) m
    time : (n_row,) s
    baseline : (n_row,) int, identifies the antenna pair
    X : (n_chan, n_row) complex, weighted visibilities
    w_row : (n_row,) inverse variance of one visibility
    flag : (n_chan, n_row) bool
    """

    field: int
    spw: int
    offset: np.ndarray
    freq: np.ndarray
    u: np.ndarray
    v: np.ndarray
    time: np.ndarray
    baseline: np.ndarray
    X: np.ndarray
    w_row: np.ndarray
    flag: np.ndarray

    @property
    def w(self) -> np.ndarray:
        """Weight of every visibility, 0 where flagged."""
        return np.where(self.flag, 0.0, self.w_row)

    @property
    def vis(self) -> np.ndarray:
        """Visibilities in Jy, 0 where flagged."""
        return self.X / self.w_row

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, k).nbytes for k in CHUNK_ARRAYS)

    def channels(self, sl: slice) -> Chunk:
        """The channels ``sl`` as a chunk of views, for bounded-memory loops over channel blocks."""
        return dataclasses.replace(self, freq=self.freq[sl], X=self.X[sl], flag=self.flag[sl])

    def uv_waves(self) -> tuple[np.ndarray, np.ndarray]:
        """``u`` and ``v`` in wavelengths, ``(n_chan, n_row)``."""
        k = self.freq[:, None] / C
        return self.u * k, self.v * k

    def phase(self, dra: float, ddec: float) -> np.ndarray:
        """``exp(-2 pi i (u dra + v ddec))``: moves the phase centre to (dra, ddec) arcsec in the sky frame."""
        delay = (self.u * (dra - self.offset[0]) + self.v * (ddec - self.offset[1])) * ARCSEC / C
        return np.exp(-2j * np.pi * np.outer(self.freq, delay))

    def jackknife(self) -> Chunk:
        """A signal-free noise realisation: consecutive integrations differenced and halved.

        Integrations are paired in time order, (0, 1), (2, 3), ..., and each row is matched to
        the row of the same baseline in its partner, so rows missing on either side are skipped
        instead of misaligning the pairs. A signal constant over a pair cancels; the noise stays,
        with weight ``pair_weight`` (2w for equal weights).

        The split is by integration, never random: a random sign flip of individual
        visibilities destroys the baseline structure and does not give the observation's
        white-noise level.
        """
        t = np.unique(self.time, return_inverse=True)[1]
        key = t // 2 * (self.baseline.max() + 1) + self.baseline
        first = t % 2 == 0
        _, ia, ib = np.intersect1d(key[first], key[~first], return_indices=True)
        a, b = np.flatnonzero(first)[ia], np.flatnonzero(~first)[ib]

        w_row = pair_weight(self.w_row[a], self.w_row[b])
        flag = self.flag[:, a] | self.flag[:, b]
        diff = 0.5 * (self.X[:, a] / self.w_row[a] - self.X[:, b] / self.w_row[b])
        return dataclasses.replace(
            self,
            u=0.5 * (self.u[a] + self.u[b]),
            v=0.5 * (self.v[a] + self.v[b]),
            time=0.5 * (self.time[a] + self.time[b]),
            baseline=self.baseline[a],
            X=np.where(flag, 0.0, w_row * diff),
            w_row=w_row,
            flag=flag,
        )


@dataclass(frozen=True)
class Metadata:
    """Quantities of the whole dataset, the same whichever fields or windows are loaded.

    Attributes
    ----------
    dish_diameter : m
    ref : (RA, Dec) in rad that sky offsets are measured from (the first target field)
    central_frequency : Hz
    max_uvwave : longest baseline in wavelengths
    """

    dish_diameter: float
    ref: tuple[float, float]
    central_frequency: float
    max_uvwave: float

    @classmethod
    def from_ms(cls, msfile: str) -> Metadata:
        fields, spws = target_selection(msfile)
        msmd = tools().msmetadata()
        msmd.open(msfile)
        diameters = msmd.antennadiameter()
        dish = diameters[next(iter(diameters))]["value"]
        ref = _phase_centre(msmd, fields[0])
        freqs = np.concatenate([msmd.chanfreqs(int(s)) for s in spws])
        msmd.close()

        tb = tools().table()
        tb.open(msfile)
        sel = tb.query(f"FIELD_ID IN {[int(f) for f in fields]} AND ANTENNA1 != ANTENNA2", columns="UVW")
        uvw = sel.getcol("UVW")
        sel.close()
        tb.close()
        return cls(dish, ref, float(freqs.mean()), float(np.hypot(uvw[0], uvw[1]).max() * freqs.max() / C))

    def primarybeamsize(self, dish_diameter: float | None = None) -> float:
        """Half-power primary beam width in arcsec (1.22 lambda / D)."""
        d = self.dish_diameter if dish_diameter is None else dish_diameter
        return 1.22 * C / self.central_frequency / d / ARCSEC

    def minresolution(self) -> float:
        """Angular resolution in arcsec from the longest baseline."""
        return 1.0 / self.max_uvwave / ARCSEC


def target_selection(msfile: str):
    """Field and spectral-window ids observed with the target intent."""
    msmd = tools().msmetadata()
    msmd.open(msfile)
    fields = msmd.fieldsforintent("*OBSERVE_TARGET*", False)
    spws = msmd.spwsforintent("*OBSERVE_TARGET*")
    msmd.close()
    return fields, spws


def _phase_centre(msmd, field) -> tuple[float, float]:
    centre = msmd.phasecenter(int(field))
    return centre["m0"]["value"], centre["m1"]["value"]


def iter_chunks(msfile: str, ref, fields=None, spws=None) -> Iterator[Chunk]:
    """Read target visibilities one (field, spw) at a time, offsets measured from ``ref``.

    Autocorrelations, flagged rows and rows with no unflagged channel are dropped. Rows are read
    in blocks, so the peak memory is about twice the largest chunk.
    """
    all_fields, all_spws = target_selection(msfile)
    msmd = tools().msmetadata()
    msmd.open(msfile)
    n_ant = msmd.nantennas()
    plan = [
        (
            int(f),
            int(s),
            sky_offset(_phase_centre(msmd, f), ref),
            msmd.chanfreqs(int(s)),
            msmd.datadescids(spw=int(s))[0],
        )
        for f in (all_fields if fields is None else fields)
        for s in (all_spws if spws is None else spws)
    ]
    msmd.close()

    tb = tools().table()
    tb.open(msfile)
    try:
        for field, spw, offset, freq, ddid in plan:
            sel = tb.query(f"FIELD_ID == {field} AND DATA_DESC_ID == {ddid} AND ANTENNA1 != ANTENNA2 AND NOT FLAG_ROW")
            try:
                chunk = _read_chunk(sel, field, spw, offset, freq, n_ant)
            finally:
                sel.close()
            if chunk is not None:
                yield chunk
    finally:
        tb.close()


def _read_chunk(sel, field, spw, offset, freq, n_ant) -> Chunk | None:
    order = np.argsort(freq)
    step = max(1, READ_BLOCK // len(freq))
    parts = []
    for start in range(0, sel.nrows(), step):

        def col(name, start=start):
            return sel.getcol(name, start, step)

        vis, w_row, flag = stokes_i(col("DATA"), col("WEIGHT"), col("FLAG"))
        keep = (w_row > 0) & ~flag.all(axis=0)
        flag = flag[np.ix_(order, keep)]
        uvw = col("UVW")[:, keep]
        parts.append(
            {
                "u": uvw[0],
                "v": uvw[1],
                "time": col("TIME")[keep],
                "baseline": (col("ANTENNA1") * n_ant + col("ANTENNA2"))[keep],
                "X": np.where(flag, 0.0, vis[np.ix_(order, keep)] * w_row[keep]),
                "w_row": w_row[keep],
                "flag": flag,
            }
        )
    if not sum(len(p["u"]) for p in parts):
        return None
    arrays = {k: np.concatenate([p[k] for p in parts], axis=-1) for k in parts[0]}
    return Chunk(field, spw, offset, freq[order], **arrays)


def write_npz(path: str, metadata: Metadata, chunks: Iterable[Chunk]) -> None:
    """Write an NPZ one array at a time, so ``chunks`` may be a generator that never sits in memory whole."""

    def put(name, value):
        with zf.open(name + ".npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array(fh, np.asanyarray(value))

    ids = []
    with zipfile.ZipFile(path, "w", allowZip64=True) as zf:
        for name, value in dataclasses.asdict(metadata).items():
            put(name, value)
        for chunk in chunks:
            for name in CHUNK_ARRAYS:
                put(f"{chunk.field}_{chunk.spw}_{name}", getattr(chunk, name))
            ids.append((chunk.field, chunk.spw))
        put("chunks", np.array(ids, dtype=int).reshape(-1, 2))


class DataHandler:
    """Visibilities of a measurement set, one :class:`Chunk` per (field, spectral window).

    Parameters
    ----------
    msfile : str, optional
        Measurement set to read; ``fields``/``spws`` select a subset of its target data.
    chunks, metadata : optional
        Ready-made data, e.g. from :meth:`from_npz` or a test.
    """

    def __init__(
        self,
        msfile: str | None = None,
        chunks: list[Chunk] | None = None,
        metadata: Metadata | None = None,
        fields=None,
        spws=None,
    ):
        if msfile is not None:
            metadata = Metadata.from_ms(msfile)
            chunks = list(iter_chunks(msfile, metadata.ref, fields, spws))
        self.msfile = msfile
        self.metadata = metadata
        self.chunks = sorted(chunks, key=lambda c: (c.field, c.freq[0]))

    @property
    def fields(self) -> list[int]:
        return sorted({c.field for c in self.chunks})

    @property
    def freqs(self) -> np.ndarray:
        """Channel frequencies in Hz, every chunk in order."""
        return np.concatenate([c.freq for c in self.chunks])

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self.chunks)

    def jackknife(self) -> DataHandler:
        """Signal-free noise realisation of every chunk (see :meth:`Chunk.jackknife`)."""
        return DataHandler(chunks=[c.jackknife() for c in self.chunks], metadata=self.metadata)

    def spectrum(self, dra: float = 0.0, ddec: float = 0.0, model=None) -> np.ndarray:
        """Weighted mean visibility per channel, phase-shifted to (dra, ddec) arcsec.

        With ``model`` (a source component) the mean is over its visibilities on the data's
        uv coverage and weights instead.
        """
        out = []
        for c in self.chunks:
            vis = c.X if model is None else c.w * model.profile(c)
            weight = c.w.sum(axis=1)
            total = (vis * c.phase(dra, ddec)).sum(axis=1)
            out.append(np.divide(total, weight, out=np.zeros_like(total), where=weight > 0))
        return np.concatenate(out)

    # --------------------------------------------------------------- NPZ I/O
    def to_npz(self, path: str) -> None:
        """Save the chunks and metadata to an NPZ (CASA-free format)."""
        write_npz(path, self.metadata, self.chunks)

    @classmethod
    def from_npz(cls, path: str, fields=None) -> DataHandler:
        """Load an NPZ written by :meth:`to_npz` or ``luv-export``; ``fields`` loads only those."""
        with np.load(path) as f:
            metadata = Metadata(
                float(f["dish_diameter"]),
                tuple(f["ref"].tolist()),
                float(f["central_frequency"]),
                float(f["max_uvwave"]),
            )
            chunks = [
                Chunk(int(field), int(spw), **{k: f[f"{field}_{spw}_{k}"] for k in CHUNK_ARRAYS})
                for field, spw in f["chunks"]
                if fields is None or field in fields
            ]
        return cls(chunks=chunks, metadata=metadata)
