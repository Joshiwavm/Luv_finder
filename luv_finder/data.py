"""Measurement-set I/O and visibility-domain operations.

Visibilities are held per (field, spectral window) in a :class:`Chunk`: the weighted
visibilities ``X = w * V`` and the weight of every visibility ``w`` (0 where flagged) as
``(n_chan, n_row)`` arrays, and what is constant along an axis (``u``, ``v``, time, baseline and
scan per row, frequency per channel) once. The weights are corrected on reading, per scan and
channel, by the noise measured in the data (:func:`reweighted`).

Positions are offsets in one sky frame per dataset: arcsec east and north of
:attr:`Metadata.ref`, the model's (dra, ddec) convention. Each chunk carries its field's phase
centre in that frame, so the same (dra, ddec) means the same sky position in every pointing.

CASA (``casatools``) is imported lazily inside the functions that read a measurement set, so
the NPZ path (:meth:`DataHandler.from_npz`) works in an environment without CASA.
"""

from __future__ import annotations

import dataclasses
import warnings
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import NamedTuple

import numpy as np

from ._casa import tools
from .utils import ARCSEC, C, primary_beam_fwhm

#: Arrays stored per chunk in an NPZ, under ``{field}_{spw}_{name}``.
CHUNK_ARRAYS = ("offset", "freq", "u", "v", "time", "baseline", "scan", "X", "w")

#: Target size of one block of rows read from a measurement set, in visibility-channels.
READ_BLOCK = 25_000_000

#: Largest time between two integrations of a baseline that measure the weights' noise together, s.
MAX_GAP = 450.0

#: Channels handled at once when the weights are measured.
CHANNEL_BLOCK = 16


def pair_weight(a, b) -> np.ndarray:
    """Inverse variance of ``(x_a +- x_b) / 2`` from the weights of ``x_a`` and ``x_b``; 0 if either is."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    s = a + b
    return np.divide(4 * a * b, s, out=np.zeros(np.broadcast(a, b).shape), where=s > 0)


def _pairs(time: np.ndarray, baseline: np.ndarray, max_gap: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint pairs of consecutive integrations of each baseline, at most ``max_gap`` s apart, in time order."""
    order = np.lexsort((time, baseline))
    linked = (baseline[order][1:] == baseline[order][:-1]) & (np.diff(time[order]) <= max_gap)
    start = np.flatnonzero(np.r_[True, ~linked])
    rank = np.arange(order.size) - start[np.cumsum(np.r_[True, ~linked]) - 1]
    pick = np.flatnonzero((rank % 2 == 0)[:-1] & linked)
    return order[pick], order[pick + 1]


def _vis(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.divide(X, w, out=np.zeros_like(X), where=w > 0)


class WeightScale(NamedTuple):
    """Noise power over the weights per scan and channel, ``scale``, its robust ``check`` and that one's ``error``."""

    scans: np.ndarray
    scale: np.ndarray
    check: np.ndarray
    error: np.ndarray

    def deviation(self) -> np.ndarray:
        """``check / scale - 1`` in units of its counting error: about N(0, 1) for Gaussian noise."""
        return (self.check / self.scale - 1) / self.error


def weight_scale(chunks: Sequence[Chunk]) -> WeightScale:
    """``sum w z^2 / 2 sum w`` per (scan, channel) over a window's integrations of a baseline <= ``MAX_GAP`` apart.

    The integrations pair in their own field, so a pair sees one sky; a (scan, channel) with no such pair (a
    mosaic field seen once per visit) pairs across fields. A pair that spans two scans counts in both, so a
    scan of one integration is measured too.
    """
    time, baseline, scan = (np.concatenate([getattr(c, k) for c in chunks]) for k in ("time", "baseline", "scan"))
    field = np.concatenate([np.full(c.time.size, c.field) for c in chunks])
    scans = np.unique(scan)
    own = _scale_cells(
        chunks, *_pairs(time, baseline.astype(np.int64) * (field.max() + 1) + field, MAX_GAP), scan, scans
    )
    if np.any(own["data"] & (own["den"] == 0)):
        across = _scale_cells(chunks, *_pairs(time, baseline, MAX_GAP), scan, scans)
        use = own["den"] > 0
        own = {k: np.where(use, own[k], across[k]) for k in own}
    if np.any(own["data"] & (own["den"] == 0)):
        k = int(np.flatnonzero(np.any(own["data"] & (own["den"] == 0), axis=1))[0])
        raise ValueError(
            f"spw {chunks[0].spw}, scan {scans[k] & 0xFFFFFFFF} of execution block {scans[k] >> 32}: no two "
            f"integrations of a baseline within {MAX_GAP:.0f} s to measure its weights; flag it or raise MAX_GAP"
        )
    if np.any((own["den"] > 0) & (own["num"] == 0)):
        raise ValueError(f"spw {chunks[0].spw} holds no noise to measure; read noiseless data with reweight=False")
    scale = np.where(own["den"] > 0, own["num"] / np.where(own["den"] > 0, own["den"], 1.0), 1.0)
    return WeightScale(scans, scale, own["check"], own["error"])


def _scale_cells(chunks: Sequence[Chunk], a, b, scan, scans) -> dict:
    """``num``, ``den`` (of :func:`weight_scale`), the robust ``check``, its ``error`` and whether there are
    ``data``, per (scan, channel), from the pairs ``a``, ``b``."""
    of_a, of_b, of_row = (np.searchsorted(scans, s) for s in (scan[a], scan[b], scan))
    n_chan = chunks[0].freq.size
    out = {k: np.zeros((scans.size, n_chan)) for k in ("num", "den")}
    out |= {k: np.full((scans.size, n_chan), np.nan) for k in ("check", "error")}
    out["data"] = np.zeros((scans.size, n_chan), dtype=bool)
    for lo in range(0, n_chan, CHANNEL_BLOCK):
        sl = slice(lo, lo + CHANNEL_BLOCK)
        X, w = (np.concatenate([getattr(c, k)[sl] for c in chunks], axis=1) for k in ("X", "w"))
        wa, wb = w[:, a].astype(float), w[:, b].astype(float)
        both = (wa > 0) & (wb > 0)
        wp = np.divide(wa * wb, wa + wb, out=np.zeros_like(wa), where=both)
        z2 = np.abs(_vis(X[:, a], wa) - _vis(X[:, b], wb)) ** 2 * wp
        for k in range(scans.size):
            m = (of_a == k) | (of_b == k)
            out["num"][k, sl] = np.sum(wp[:, m] * z2[:, m], axis=1)
            out["den"][k, sl] = 2 * np.sum(wp[:, m], axis=1)
            out["data"][k, sl] = np.sum(w[:, of_row == k], axis=1) > 0
            out["check"][k, sl] = [_robust(z[o]) for z, o in zip(z2[:, m], both[:, m], strict=True)]
            n, w1, w2 = both[:, m].sum(axis=1), np.sum(wp[:, m], axis=1), np.sum(wp[:, m] ** 2, axis=1)
            ok = n > 0
            # the median's variance 1 / ln(2)^2 / n, the weighted mean's, less twice their covariance 1 / n
            out["error"][k, sl][ok] = np.sqrt((1 / np.log(2) ** 2 - 2) / n[ok] + w2[ok] / w1[ok] ** 2)
    return out


def _robust(z2: np.ndarray) -> float:
    """The noise power from the median of ``z^2``, ``median / 2 ln 2``, which outliers barely move."""
    return float(np.median(z2) / (2 * np.log(2))) if z2.size else np.nan


def reweighted(chunks: Sequence[Chunk], ws: WeightScale | None = None) -> list[Chunk]:
    """The chunks of one window, weights divided by ``ws`` (default :func:`weight_scale`) per scan and channel."""
    ws = weight_scale(chunks) if ws is None else ws
    deviation = np.nanmax(np.abs(ws.deviation()))
    print(
        f"spw {chunks[0].spw}: noise over the weights in {ws.scans.size} scans, median per scan "
        f"{np.median(ws.scale, axis=1).min():.3f}-{np.median(ws.scale, axis=1).max():.3f}, "
        f"range {ws.scale.min():.2f}-{ws.scale.max():.2f}; "
        f"robust check within {deviation:.1f} sigma",
        flush=True,
    )
    if deviation > 5:
        warnings.warn(
            f"spw {chunks[0].spw}: the weighted and robust noise differ by {deviation:.1f} sigma", stacklevel=2
        )
    out = []
    for c in chunks:
        factor = 1 / ws.scale[np.searchsorted(ws.scans, c.scan)].T
        out.append(dataclasses.replace(c, X=c.X * factor, w=(c.w * factor).astype(np.float32)))
    return out


class ChannelNoise(NamedTuple):
    """Noise of each channel over its weights, ``scale`` (1 where they hold), and between channels, ``rho``."""

    scale: np.ndarray
    rho: np.ndarray


def channel_noise(chunks: Iterable[Chunk], max_lag: int = 8, significance: float = 5.0) -> ChannelNoise:
    """On the jackknife: noise power per channel over the weights, ``sum |X|^2 / 2 sum w``, and ``(1, rho_1, ...)``."""
    num, norm_a, norm_b, n = (np.zeros(max_lag + 1) for _ in range(4))
    power = weight = 0.0
    for chunk in chunks:
        jk = chunk.jackknife()
        power = power + np.sum(np.abs(jk.X) ** 2, axis=1)
        weight = weight + 2 * jk.w.sum(axis=1)
        z = np.divide(jk.X, np.sqrt(jk.w), out=np.zeros_like(jk.X), where=jk.w > 0)
        for lag in range(1, min(max_lag, z.shape[0] - 1) + 1):
            a, b = z[:-lag], z[lag:]
            ok = (jk.w[:-lag] > 0) & (jk.w[lag:] > 0)
            num[lag] += np.sum((a * b.conj()).real, where=ok)
            norm_a[lag] += np.sum(np.abs(a) ** 2, where=ok)
            norm_b[lag] += np.sum(np.abs(b) ** 2, where=ok)
            n[lag] += 2 * ok.sum()
    if not np.any(power):
        raise ValueError("the jackknife holds no noise to measure; pass channel_noise=None or a noisy dataset's")
    rho = [1.0]
    for lag in range(1, max_lag + 1):
        if norm_a[lag] * norm_b[lag] == 0:
            break
        r = num[lag] / np.sqrt(norm_a[lag] * norm_b[lag])
        if abs(r) * np.sqrt(n[lag]) < significance:
            break
        rho.append(float(r))
    return ChannelNoise(np.divide(power, weight, out=np.ones_like(power), where=weight > 0), np.array(rho))


def stokes_i(data: np.ndarray, weight: np.ndarray, flag: np.ndarray):
    """Average the parallel hands ``[0, -1]`` into (vis, w); ``weight`` per row or per channel, ``w`` 0 if flagged."""
    w = pair_weight(weight[0], weight[-1])
    return 0.5 * (data[0] + data[-1]), np.where(flag[0] | flag[-1], 0.0, np.broadcast_to(w, data.shape[1:]))


def sky_offset(direction, ref) -> np.ndarray:
    """(east, north) offset in arcsec of ``direction`` on the tangent plane at ``ref``, both (RA, Dec) in rad."""
    (ra, dec), (ra0, dec0) = direction, ref
    east = np.cos(dec) * np.sin(ra - ra0)
    north = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
    return np.array([east, north]) / ARCSEC


def sky_direction(offset, ref) -> np.ndarray:
    """(RA, Dec) in rad of the (east, north) ``offset`` in arcsec from ``ref``, the inverse of :func:`sky_offset`."""
    (l, m), (ra0, dec0) = np.asarray(offset, dtype=float) * ARCSEC, ref
    n = np.sqrt(1 - l**2 - m**2)
    dec = np.arcsin(m * np.cos(dec0) + n * np.sin(dec0))
    ra = ra0 + np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    return np.array([ra % (2 * np.pi), dec])


@dataclass(frozen=True, eq=False)
class Chunk:
    """Visibilities of one (field, spectral window), channel-major.

    ``X = w * V`` is zero where ``w`` is (flagged), so a weighted sum over visibilities is a plain
    sum over ``X``. Every row has some channel with ``w > 0``.

    Attributes
    ----------
    offset : (2,) arcsec, the field's phase centre in the dataset's sky frame
    freq : (n_chan,) Hz, ascending
    u, v : (n_row,) m
    time : (n_row,) s
    baseline : (n_row,) int, identifies the antenna pair
    scan : (n_row,) int64, the scan, ``OBSERVATION_ID << 32 | SCAN_NUMBER``
    X : (n_chan, n_row) complex, weighted visibilities
    w : (n_chan, n_row) float32, inverse variance of each visibility, 0 where flagged
    """

    field: int
    spw: int
    offset: np.ndarray
    freq: np.ndarray
    u: np.ndarray
    v: np.ndarray
    time: np.ndarray
    baseline: np.ndarray
    scan: np.ndarray
    X: np.ndarray
    w: np.ndarray

    @property
    def flag(self) -> np.ndarray:
        return self.w == 0

    @property
    def vis(self) -> np.ndarray:
        """Visibilities in Jy, 0 where flagged."""
        return _vis(self.X, self.w)

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, k).nbytes for k in CHUNK_ARRAYS)

    def channels(self, sl: slice) -> Chunk:
        """The channels ``sl`` as a chunk of views, for bounded-memory loops over channel blocks."""
        return dataclasses.replace(self, freq=self.freq[sl], X=self.X[sl], w=self.w[sl])

    def uv_waves(self) -> tuple[np.ndarray, np.ndarray]:
        """``u`` and ``v`` in wavelengths, ``(n_chan, n_row)``."""
        k = self.freq[:, None] / C
        return self.u * k, self.v * k

    def phase(self, dra: float, ddec: float) -> np.ndarray:
        """``exp(-2 pi i (u dra + v ddec))``: moves the phase centre to (dra, ddec) arcsec in the sky frame."""
        delay = (self.u * (dra - self.offset[0]) + self.v * (ddec - self.offset[1])) * ARCSEC / C
        return np.exp(-2j * np.pi * np.outer(self.freq, delay))

    def jackknife(self) -> Chunk:
        """A signal-free noise realisation: consecutive integrations of every baseline differenced and halved.

        Each baseline's integrations are paired in time order, its first with its second, third
        with fourth, ..., whatever the other baselines do, so a mosaic field seen once per visit
        (or a baseline missing from some integrations) still pairs nearly every row; an odd last
        integration is left out. A signal constant over a pair cancels; the noise stays, with
        weight ``pair_weight`` (2w for equal weights).

        The split is by integration, never random: a random sign flip of individual
        visibilities destroys the baseline structure and does not give the observation's
        white-noise level.
        """
        a, b = _pairs(self.time, self.baseline)
        wa, wb = self.w[:, a], self.w[:, b]
        # column indexing returns Fortran-ordered arrays; keep chunks row-major
        w = np.ascontiguousarray(pair_weight(wa, wb))
        diff = 0.5 * (_vis(self.X[:, a], wa) - _vis(self.X[:, b], wb))
        return dataclasses.replace(
            self,
            u=0.5 * (self.u[a] + self.u[b]),
            v=0.5 * (self.v[a] + self.v[b]),
            time=0.5 * (self.time[a] + self.time[b]),
            baseline=self.baseline[a],
            scan=self.scan[a],
            X=np.ascontiguousarray(w * diff),
            w=w.astype(np.float32),
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
        tb.open(f"{msfile}/SPECTRAL_WINDOW")
        check_frame(msfile, tb.getcol("MEAS_FREQ_REF")[[int(s) for s in spws]])
        tb.close()
        tb.open(msfile)
        sel = tb.query(f"FIELD_ID IN {[int(f) for f in fields]} AND ANTENNA1 != ANTENNA2", columns="UVW")
        uvw = sel.getcol("UVW")
        sel.close()
        tb.close()
        return cls(dish, ref, float(freqs.mean()), float(np.hypot(uvw[0], uvw[1]).max() * freqs.max() / C))

    def primarybeamsize(self, dish_diameter: float | None = None) -> float:
        """Primary-beam FWHM in arcsec at the central frequency (1.13 lambda / D)."""
        d = self.dish_diameter if dish_diameter is None else dish_diameter
        return float(primary_beam_fwhm(self.central_frequency, d))

    def minresolution(self) -> float:
        """Angular resolution in arcsec from the longest baseline."""
        return 1.0 / self.max_uvwave / ARCSEC


#: casacore's MFrequency reference codes (``MEAS_FREQ_REF`` of an MS's SPECTRAL_WINDOW).
FREQUENCY_FRAMES = ("REST", "LSRK", "LSRD", "BARY", "GEO", "TOPO", "GALACTO", "LGROUP", "CMB")


def check_frame(msfile: str, codes) -> None:
    """Warn unless every window is in LSRK: the search, the catalogue and the fit keep the MS's frame."""
    frames = sorted({FREQUENCY_FRAMES[int(c)] for c in np.atleast_1d(codes)})
    if frames != ["LSRK"]:
        warnings.warn(
            f"{msfile}: channel frequencies are in {', '.join(frames)}, not LSRK, and so will be the catalogue's "
            "frequencies and redshifts (TOPO differs from LSRK by up to ~30 km/s); regrid to LSRK first "
            "(CASA mstransform, outframe='LSRK') for published values",
            stacklevel=2,
        )


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


def iter_chunks(
    msfile: str, ref, fields=None, spws=None, reweight: bool = True, scales: dict | None = None
) -> Iterator[Chunk]:
    """Read target visibilities one spw at a time over all ``fields``, weights corrected, offsets from ``ref``.

    Autocorrelations, flagged rows and rows with no unflagged channel are dropped. Every window's
    weights are divided by the noise measured in it (:func:`reweighted`), so the peak memory is
    about twice one window over all its fields; ``reweight=False`` keeps the measurement set's
    weights, for a noiseless simulation, which has no noise to measure. A ``scales`` dict receives
    each window's :class:`WeightScale` by spw.
    """
    all_fields, all_spws = target_selection(msfile)
    msmd = tools().msmetadata()
    msmd.open(msfile)
    n_ant = msmd.nantennas()
    plan = [
        (
            int(s),
            int(f),
            sky_offset(_phase_centre(msmd, f), ref),
            msmd.chanfreqs(int(s)),
            msmd.datadescids(spw=int(s))[0],
        )
        for s in (all_spws if spws is None else spws)
        for f in (all_fields if fields is None else fields)
    ]
    msmd.close()

    tb = tools().table()
    tb.open(msfile)
    spectrum = "WEIGHT_SPECTRUM" in tb.colnames()
    try:
        for _, window in groupby(plan, key=lambda p: p[0]):
            chunks = []
            for spw, field, offset, freq, ddid in window:
                sel = tb.query(
                    f"FIELD_ID == {field} AND DATA_DESC_ID == {ddid} AND ANTENNA1 != ANTENNA2 AND NOT FLAG_ROW"
                )
                try:
                    chunk = _read_chunk(sel, field, spw, offset, freq, n_ant, spectrum)
                finally:
                    sel.close()
                if chunk is not None:
                    chunks.append(chunk)
            if chunks and reweight:
                ws = weight_scale(chunks)
                if scales is not None:
                    scales[chunks[0].spw] = ws
                yield from reweighted(chunks, ws)
            elif chunks:
                yield from chunks
    finally:
        tb.close()


def _read_chunk(sel, field, spw, offset, freq, n_ant, spectrum) -> Chunk | None:
    order = np.argsort(freq)
    step = max(1, READ_BLOCK // len(freq))
    parts = []
    for start in range(0, sel.nrows(), step):

        def col(name, start=start):
            return sel.getcol(name, start, step)

        vis, w = stokes_i(col("DATA"), col("WEIGHT_SPECTRUM" if spectrum else "WEIGHT"), col("FLAG"))
        keep = w.any(axis=0)
        w = w[np.ix_(order, keep)]
        uvw = col("UVW")[:, keep]
        parts.append(
            {
                "u": uvw[0],
                "v": uvw[1],
                "time": col("TIME")[keep],
                "baseline": (col("ANTENNA1") * n_ant + col("ANTENNA2"))[keep],
                "scan": (col("OBSERVATION_ID").astype(np.int64) << 32 | col("SCAN_NUMBER"))[keep],
                "X": vis[np.ix_(order, keep)] * w,
                "w": w.astype(np.float32),
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
        Measurement set to read; ``fields``/``spws`` select a subset of its target data, and
        ``reweight=False`` keeps its weights (a noiseless simulation; see :func:`iter_chunks`).
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
        reweight: bool = True,
    ):
        if msfile is not None:
            metadata = Metadata.from_ms(msfile)
            chunks = list(iter_chunks(msfile, metadata.ref, fields, spws, reweight))
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

    def channel_noise(self, **kwargs) -> dict[int, ChannelNoise]:
        """:func:`channel_noise` per spectral window, pooled over its fields: ``{spw: ChannelNoise}``."""
        spws = sorted({c.spw for c in self.chunks})
        return {s: channel_noise((c for c in self.chunks if c.spw == s), **kwargs) for s in spws}

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


def _is_npz(path) -> bool:
    return str(path).endswith(".npz")


def fields_in(path) -> list[int]:
    """Target fields of a measurement set or an NPZ, sorted; an NPZ is not read beyond its index."""
    if _is_npz(path):
        with np.load(path) as f:
            return np.unique(f["chunks"][:, 0]).tolist()
    return sorted(int(f) for f in target_selection(path)[0])


def load(path, fields=None) -> DataHandler:
    """The visibilities of ``fields`` (default: all) from a measurement set or an NPZ."""
    return DataHandler.from_npz(path, fields) if _is_npz(path) else DataHandler(str(path), fields=fields)
