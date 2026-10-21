"""A line search on disk: one YAML file in, every product in one output directory.

:class:`Run` exports the data block by block (spectral windows that overlap in frequency),
searches it unit by unit (blocks a line can span, cut into chunks of at most ``max_channels``
that overlap by a block), catalogues and fits the lines, and keeps everything the figures need
under ``output``, so plotting never repeats a search::

    run = Run("runs/j0459_band3.yaml")
    run.search()     # export -> search -> catalogue, unit by unit; what is on disk is skipped
    run.fit()        # fit the detections and the near misses
    run.summary()
    run.plot.spectra()

``Run.open(output)`` reopens a finished run from its directory alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml
from astropy.table import Table

from .catalogue import DETECTION_RULE, _reliability, bright_lines, catalogue, merge_catalogues
from .data import Metadata, WeightScale, fields_in, iter_chunks, load, target_selection, write_npz
from .matchedfilter import TEMPLATE_KEYS, WIDTHS, grid_axis, search_pointings, window_groups
from .model import C_KMS, CONTINUUM_ORDER

#: Name of the config copy every run keeps in its output directory.
RUN_FILE = "run.yaml"

#: Defaults of the run config; ``data`` and ``output`` have none.
DEFAULTS = {
    "name": None,
    "grid": None,
    "pb_limit": 0.2,
    "units": "auto",
    "max_channels": None,
    "filter": {},
    "reliability": {"floor": 4.0, "k": 3.0, "fidelity_min": 0.6, "rule": DETECTION_RULE},
    "near_miss": {"fidelity": 0.2, "max": 20},
    "cores": None,
}


def window_units(freqs: dict[int, np.ndarray], widest: float = max(WIDTHS)) -> list[list[int]]:
    """Spectral windows joined where they overlap or their gap is narrower than ``widest`` km/s, in frequency order."""
    spws = list(freqs)
    return [sorted(spws[k] for k in group) for group in window_groups([freqs[s] for s in spws], widest)]


def unit_name(spws) -> str:
    return "spw" + "+".join(f"{s:02d}" for s in spws)


class Unit(NamedTuple):
    """A unit of a run: the blocks it loads, and the trial centres it owns (Hz; None: all of them)."""

    blocks: list[str]
    core: tuple[float, float] | None


def unit_plan(freqs: dict[int, np.ndarray], widest: float = max(WIDTHS), max_channels: int | None = None) -> dict:
    """Blocks (``{name: spws}``) and units (``{name: Unit}``) of the windows ``freqs``.

    A block is the windows that overlap by more than a channel (the tunings of one range), exported
    once. A unit is the blocks of :func:`window_units`, cut into chunks of whole blocks of at most
    ``max_channels`` channels: each owns the trial centres between the gaps to its neighbouring
    blocks (``core``) and loads those neighbours too, so a line at its edge is whole in it, and a
    group is catalogued by the one chunk that owns its peak.
    """
    blocks, units = {}, {}
    for group in window_units(freqs, widest):
        sub = {s: freqs[s] for s in group}
        channel = max(np.abs(np.diff(f)).max() / f.mean() * C_KMS for f in sub.values())
        own = window_units(sub, -channel)  # overlapping by more than a channel
        names = [unit_name(b) for b in own]
        blocks |= dict(zip(names, own, strict=True))
        size = [sum(freqs[s].size for s in b) for b in own]
        lo, hi = [min(freqs[s].min() for s in b) for b in own], [max(freqs[s].max() for s in b) for b in own]
        if max_channels is None or sum(size) <= max_channels:
            units[unit_name(group)] = Unit(names, None)
            continue
        i = 0
        while i < len(own):
            j = i
            while j + 1 < len(own) and sum(size[i : j + 2]) <= max_channels:
                j += 1
            core = ((hi[i - 1] + lo[i]) / 2 if i else -np.inf, (hi[j] + lo[j + 1]) / 2 if j + 1 < len(own) else np.inf)
            units[unit_name(sorted(s for b in own[i : j + 1] for s in b))] = Unit(names[max(i - 1, 0) : j + 2], core)
            i = j + 1
    return {"blocks": blocks, "units": units}


def _window_freqs(path: str) -> dict[int, np.ndarray]:
    """Channel frequencies (Hz) of every target spectral window of a measurement set or an NPZ."""
    if str(path).endswith(".npz"):
        with np.load(path) as f:
            field_of = {int(s): int(fd) for fd, s in f["chunks"]}
            return {s: f[f"{fd}_{s}_freq"] for s, fd in field_of.items()}
    from ._casa import tools

    msmd = tools().msmetadata()
    msmd.open(str(path))
    try:
        return {int(s): msmd.chanfreqs(int(s)) for s in target_selection(str(path))[1]}
    finally:
        msmd.close()


def _fitted(cat: Table, key: str) -> np.ndarray:
    """``fit_<key>`` where the line was fitted, the catalogue's ``key`` elsewhere."""
    value = np.asarray(cat[key], dtype=float)
    if f"fit_{key}" not in cat.colnames:
        return value
    fit = np.asarray(cat[f"fit_{key}"], dtype=float)
    return np.where(np.isfinite(fit), fit, value)


def duplicates(cat: Table, beam: dict) -> np.ndarray:
    """Id of a brighter group of the same line in another unit, else 0, for every row of ``cat``.

    Two groups are one line when they are within ``beam[unit]`` (arcsec) of each other and their
    profiles overlap, ``|dv| <= (FWHM_a + FWHM_b) / 2``, with the fitted position, frequency and width
    where the line was fitted (the template's elsewhere): a line wider than the gap it straddles between
    two windows is found once in each, its halves far apart in frequency. ``cat`` is sorted by S/N.
    """
    from scipy.spatial import cKDTree

    dra, ddec, nu, width = (_fitted(cat, k) for k in ("dra", "ddec", "freq_ghz", "width"))
    unit = np.asarray(cat["unit"])
    pairs = cKDTree(np.c_[dra, ddec]).query_pairs(max(beam.values()), output_type="ndarray")
    pairs.sort(axis=1)  # a brighter than b
    a, b = pairs.T if pairs.size else (np.zeros(0, int), np.zeros(0, int))
    same = (
        (unit[a] != unit[b])
        & (np.hypot(dra[a] - dra[b], ddec[a] - ddec[b]) <= np.array([beam[u] for u in unit[a]]))
        & (np.abs(nu[b] / nu[a] - 1) * C_KMS <= (width[a] + width[b]) / 2)
    )
    of = np.zeros(len(cat), dtype=int)
    for i, j in sorted(zip(a[same].tolist(), b[same].tolist(), strict=True)):
        if not of[i] and not of[j]:
            of[j] = cat["id"][i]
    return of


class Run:
    """A line search described by a YAML file (or dict); see the module docstring.

    Keys: ``data`` (measurement set or NPZ), ``output`` (directory), and optionally ``name``,
    ``grid`` (dict or grid YAML, as for :func:`~luv_finder.matchedfilter.build_grid`),
    ``pb_limit``, ``units`` (``auto`` or lists of spws), ``filter`` (keyword arguments of the
    matched filter), ``reliability`` (``floor``, ``k``, ``fidelity_min``, ``rule``), ``near_miss``
    (``fidelity``, ``max``) and ``cores`` per worker. Relative paths are relative to the YAML file.
    """

    def __init__(self, config: str | Path | dict):
        if isinstance(config, dict):
            cfg, base = dict(config), Path.cwd()
        else:
            cfg, base = yaml.safe_load(Path(config).read_text()), Path(config).resolve().parent
        missing = {"data", "output"} - set(cfg)
        if missing:
            raise ValueError(f"a run needs {sorted(missing)}")
        cfg = DEFAULTS | cfg
        for key in ("reliability", "near_miss"):
            cfg[key] = DEFAULTS[key] | (cfg[key] or {})
        for key in ("data", "output"):
            cfg[key] = str((base / cfg[key]).resolve())
        if isinstance(cfg["grid"], str):
            cfg["grid"] = yaml.safe_load((base / cfg["grid"]).read_text())
        cfg["name"] = cfg["name"] or Path(cfg["output"]).name
        self.config = cfg
        self.output = Path(cfg["output"])

    @classmethod
    def open(cls, output: str | Path) -> Run:
        """A run from its output directory, with the config it was run with."""
        return cls(Path(output) / RUN_FILE)

    def __repr__(self):
        done = sum(self.path(u, ".ecsv").exists() for u in self.units)
        return f"Run({self.config['name']!r}: {done} of {len(self.units)} units searched, output {self.output})"

    # ------------------------------------------------------------------ layout
    def path(self, unit: str, suffix: str) -> Path:
        """A unit's product: ``.ecsv``, ``_maps.npz``, ``_weights.npz``, ``.claim``."""
        return self.output / "units" / f"{unit}{suffix}"

    def export_path(self, unit: str) -> Path:
        return self.output / "export" / f"{unit}.npz"

    @cached_property
    def plan(self) -> dict:
        """:func:`unit_plan` of the data: ``blocks`` (name -> spws, exported once) and ``units`` (name -> Unit)."""
        units = self.config["units"]
        if units == "auto":
            widest = float(self.widest)
            return unit_plan(_window_freqs(self.config["data"]), widest, self.config["max_channels"])
        blocks = {unit_name(spws): [int(s) for s in spws] for spws in units}
        return {"blocks": blocks, "units": {name: Unit([name], None) for name in blocks}}

    @property
    def widest(self) -> float:
        """The widest template, km/s."""
        return float(grid_axis(self.config["grid"], "width", WIDTHS).max())

    @property
    def units(self) -> dict[str, list[int]]:
        """Unit name -> the spectral windows it loads."""
        return {
            name: sorted(s for b in unit.blocks for s in self.plan["blocks"][b])
            for name, unit in self.plan["units"].items()
        }

    def exports(self, unit: str) -> list[str]:
        """The export of every block a unit loads."""
        return [str(self.export_path(b)) for b in self.plan["units"][unit].blocks]

    @cached_property
    def metadata(self) -> Metadata:
        data = self.config["data"]
        return load(data, []).metadata if data.endswith(".npz") else Metadata.from_ms(data)

    # ------------------------------------------------------------------ search
    def search(self, workers: int = 1, merge: bool = True) -> None:
        """Export, search and catalogue every unit nobody has done or claimed; then merge them.

        With ``workers > 1``, that many processes share the units, each on its own slice of the
        cores (``cores`` per worker, default an equal share), logging to ``logs/worker_<n>.log``.
        """
        self._write_config()
        if workers > 1:
            self._spawn(workers)
        else:
            for unit, spws in self.units.items():
                if self._claim(unit):
                    self._search_unit(unit, spws)
        if merge and all(self.path(u, ".ecsv").exists() for u in self.units):
            self.merge()

    def release(self) -> list[str]:
        """Remove the claims of units that have no catalogue (after a crash); the units released."""
        stale = [u for u in self.units if self.path(u, ".claim").exists() and not self.path(u, ".ecsv").exists()]
        for u in stale:
            self.path(u, ".claim").unlink()
        return stale

    def _write_config(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / RUN_FILE).write_text(yaml.safe_dump(self.config, sort_keys=False))

    def _spawn(self, workers):
        """``workers`` processes, each on its own slice of the cores, sharing the units by claims."""
        cores = sorted(os.sched_getaffinity(0))
        per = self.config["cores"] or max(1, len(cores) // workers)
        (self.output / "logs").mkdir(exist_ok=True)
        procs = []
        for w in range(workers):
            mine = cores[w * per : (w + 1) * per] or cores[-per:]
            log = open(self.output / "logs" / f"worker_{w}.log", "w")  # noqa: SIM115 - closed below
            cmd = [sys.executable, "-m", "luv_finder.cli.run", str(self.output / RUN_FILE), "search", "--no-merge"]
            procs.append(
                (
                    subprocess.Popen(
                        cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=lambda m=mine: os.sched_setaffinity(0, m)
                    ),
                    log,
                )
            )
        failed = []
        for w, (proc, log) in enumerate(procs):
            if proc.wait():
                failed.append(w)
            log.close()
        if failed:
            raise RuntimeError(f"workers {failed} failed; see {self.output / 'logs'}")

    def _claim(self, unit) -> bool:
        if self.path(unit, ".ecsv").exists():
            return False
        self.path(unit, "").parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(self.path(unit, ".claim"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return False
        return True

    def _search_unit(self, unit, spws):
        clock = time.perf_counter()
        for block in self.plan["units"][unit].blocks:
            self._export(block, self.plan["blocks"][block])
        print(f"[{unit}: export {time.perf_counter() - clock:.0f} s]", flush=True)
        rel, core = self.config["reliability"], self.plan["units"][unit].core
        # a chunk searches one widest FWHM beyond its core, so the groups at its edges form whole
        band = None if core is None else (core[0] * (1 - self.widest / C_KMS), core[1] * (1 + self.widest / C_KMS))
        result = search_pointings(
            self.exports(unit),
            self.config["grid"],
            jackknife=True,
            cores=self.config["cores"],
            pb_limit=self.config["pb_limit"],
            band=band,
            **self.config["filter"],
        )
        print(f"[{unit}: search {time.perf_counter() - clock:.0f} s]", flush=True)
        cat = catalogue(result, ref=self.metadata.ref, band=core, **rel)
        cat["unit"], cat["unit_id"] = unit, cat["id"]
        self._save_maps(unit, result, cat)
        del result
        part = self.path(unit, ".ecsv.part")
        cat.write(part, format="ascii.ecsv", overwrite=True)
        os.replace(part, self.path(unit, ".ecsv"))
        print(
            f"[{unit}: {len(cat)} groups, {int(cat['detected'].sum())} detected alone; "
            f"{time.perf_counter() - clock:.0f} s]",
            flush=True,
        )

    def _export(self, block, spws) -> Path:
        """A block's visibilities with corrected weights; the weight scales and channel noise go to ``_weights.npz``.

        Units that share a block share its export: one process writes it, the others wait for it.
        """
        npz = self.export_path(block)
        lock = npz.with_suffix(".lock")
        npz.parent.mkdir(parents=True, exist_ok=True)
        while not npz.exists():
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                time.sleep(10)
                continue
            try:
                if not npz.exists():
                    self._write_export(block, spws, npz)
            finally:
                lock.unlink()
        return npz

    def _write_export(self, block, spws, npz):
        self.path(block, "").parent.mkdir(parents=True, exist_ok=True)
        part, data, scales = npz.with_suffix(".npz.part"), self.config["data"], {}
        if data.endswith(".npz"):
            chunks = (c for f in fields_in(data) for c in load(data, [f]).chunks if c.spw in spws)
        else:
            chunks = iter_chunks(data, self.metadata.ref, spws=spws, scales=scales)
        write_npz(str(part), self.metadata, chunks)
        os.replace(part, npz)
        np.savez(
            self.path(block, "_weights.npz"),
            spws=np.array(sorted(scales), dtype=int),
            **{f"{s}_{k}": getattr(ws, k) for s, ws in scales.items() for k in ("scans", "scale", "check", "error")},
        )
        self._save_noise(block, npz)

    def _save_noise(self, block, npz):
        """Add the channel noise measured on the jackknife (first field) to the block's ``_weights.npz``."""
        path = self.path(block, "_weights.npz")
        saved = dict(np.load(path)) if path.exists() else {"spws": np.zeros(0, dtype=int)}
        data = load(str(npz), fields_in(str(npz))[:1])
        for spw, noise in data.channel_noise().items():
            saved[f"{spw}_noise_scale"], saved[f"{spw}_rho"] = noise.scale, noise.rho
            saved[f"{spw}_freq"] = next(c.freq for c in data.chunks if c.spw == spw)
        np.savez(path, **saved)

    def _save_maps(self, unit, result, cat):
        """Peak S/N per position, and the filter response at every data group (see the run format)."""
        snr, jk = result.snr, result.jackknife.snr
        flat = np.where(np.isnan(snr), -np.inf, snr).reshape(*snr.shape[:2], -1)
        best = flat.argmax(axis=2)
        empty = ~np.isfinite(flat.max(axis=2))
        dra, ddec = np.asarray(result.axes["dra"]), np.asarray(result.axes["ddec"])
        i = np.abs(dra[None, :] - np.asarray(cat["dra"])[:, None]).argmin(axis=1)
        j = np.abs(ddec[None, :] - np.asarray(cat["ddec"])[:, None]).argmin(axis=1)
        templates = np.array(list(product(*(result.axes[k] for k in TEMPLATE_KEYS))), dtype=float)
        with np.errstate(all="ignore"):
            np.savez(
                self.path(unit, "_maps.npz"),
                dra=dra,
                ddec=ddec,
                freqs=result.freqs,
                template_keys=np.array(TEMPLATE_KEYS),
                templates=templates,
                widths=templates[:, -1],
                peak=np.nanmax(snr, axis=(2, 3)).astype(np.float32),
                peak_template=np.where(empty, -1, best // snr.shape[3]).astype(np.int32),
                peak_channel=np.where(empty, -1, best % snr.shape[3]).astype(np.int32),
                jk_peak=np.nanmax(jk, axis=(2, 3)).astype(np.float32),
                coverage=np.nanmax(result.coverage, axis=2).astype(np.float32),
                group_id=np.asarray(cat["id"]),
                group_snr=snr[i, j].astype(np.float32),
                group_jk=jk[i, j].astype(np.float32),
            )

    def merge(self) -> Table:
        """The units' catalogues as one, reliability over their whole volume: ``catalogue.ecsv``.

        The tests are the run file's ``reliability``, so changing it needs a merge, not a search.

        A line found in two units (:func:`duplicates`) counts once in the likelihood ratio and the
        fidelity: the fainter group gets ``duplicate_of`` and is not detected.
        """
        cat = merge_catalogues([Table.read(self.path(u, ".ecsv")) for u in self.units], self.config["reliability"])
        cat["duplicate_of"] = self.duplicates(cat)
        one = np.asarray(cat["duplicate_of"] == 0)
        if not one.all():
            m = cat.meta
            snr = {"data": np.asarray(cat["snr"])[one]}
            snr |= {k: np.asarray(m[f"snr_{k}"]) for k in ("jackknife", "negative")}
            columns, meta = _reliability(snr, m["floor"], m["k"], m["fidelity_min"], m["rule"])
            for name, values in columns.items():
                cat[name][one] = values
                cat[name][~one] = False if values.dtype == bool else np.nan
            cat.meta = meta | {"parts": m["parts"]}
        cat.write(self.output / "catalogue.ecsv", format="ascii.ecsv", overwrite=True)
        return cat

    # --------------------------------------------------------------------- fit
    def fit(self) -> Table:
        """Fit the detections and the near misses in the visibilities: ``lines_fit.ecsv``, then :meth:`moment_maps`.

        Lines already in ``lines_fit.ecsv`` (by unit and unit id) keep their fit, so a re-merge costs no refit.
        """
        from .fit import COLUMNS, empty_fit, fit_lines

        cat = self.catalogue
        near = self.near_misses(cat)
        todo = np.asarray(cat["detected"]) | near
        fitted = empty_fit(cat)
        old = self.lines if (self.output / "lines_fit.ecsv").exists() else None
        if old is not None:
            key = {(u, int(i)): n for n, (u, i) in enumerate(zip(old["unit"], old["unit_id"], strict=True))}
            prev = np.array([key.get((u, int(i)), -1) for u, i in zip(cat["unit"], cat["unit_id"], strict=True)])
            reuse = todo & (prev >= 0) & np.isfinite(np.asarray(old["fit_dra"], dtype=float)[prev])
            for name in COLUMNS:
                fitted[name][reuse] = old[name][prev[reuse]]
            spectra = old.meta.get("spectra", {})
            fitted.meta["spectra"] = {
                str(cat["id"][n]): spectra[str(old["id"][prev[n]])] for n in np.flatnonzero(reuse)
                if str(old["id"][prev[n]]) in spectra
            }  # fmt: skip
            todo &= ~reuse
        for unit in np.unique(cat["unit"][todo]):
            rows = todo & np.asarray(cat["unit"] == unit)
            done = fit_lines(load(self.exports(unit)), cat, rows=rows, cores=self.config["cores"], **self.fit_options)
            for name in COLUMNS:
                fitted[name][rows] = done[name][rows]
            fitted.meta.setdefault("spectra", {}).update(done.meta.get("spectra", {}))
            print(f"[{unit}: fitted {rows.sum()} lines]", flush=True)
        fitted["near_miss"] = near
        fitted.write(self.output / "lines_fit.ecsv", format="ascii.ecsv", overwrite=True)
        self.moment_maps()
        return fitted

    @property
    def fit_options(self) -> dict:
        """The fit's continuum degree: the search's (``filter: continuum_order``), else the default."""
        return {"continuum_order": self.config["filter"].get("continuum_order", CONTINUUM_ORDER)}

    def moment_maps(self) -> None:
        """Moment-0 maps of every fitted line, from the export: ``line_maps.npz`` (minutes; no refit)."""
        from .imaging import line_maps

        fit, maps = self.lines, {}
        fitted = np.isfinite(np.asarray(fit["fit_dra"], dtype=float))
        for unit in np.unique(fit["unit"][fitted]):
            ok = fit[fitted & np.asarray(fit["unit"] == unit)]
            maps |= dict(zip(ok["id"], line_maps(load(self.exports(unit)), ok), strict=True))
        np.savez(
            self.output / "line_maps.npz",
            **{f"{i}_{k}": v for i, m in maps.items() for k, v in zip(("dra", "ddec", "moment0"), m, strict=True)},
        )

    def near_misses(self, cat: Table) -> np.ndarray:
        """Not detected, fidelity at least ``near_miss.fidelity``: the brightest ``near_miss.max``."""
        nm = self.config["near_miss"]
        near = np.asarray(~cat["detected"] & (cat["fidelity"] >= nm["fidelity"]))
        near[np.cumsum(near) > nm["max"]] = False
        return near

    def duplicates(self, cat: Table) -> np.ndarray:
        """:func:`duplicates` of ``cat``, the beam of each unit from its catalogue."""
        beam = {u: max(Table.read(self.path(u, ".ecsv")).meta["hwhm_major"]) for u in self.units}
        return duplicates(cat, beam)

    # ---------------------------------------------------------------- products
    @property
    def catalogue(self) -> Table:
        """The merged catalogue (``catalogue.ecsv``)."""
        return Table.read(self.output / "catalogue.ecsv")

    @property
    def lines(self) -> Table:
        """The fitted catalogue (``lines_fit.ecsv``)."""
        return Table.read(self.output / "lines_fit.ecsv")

    def maps(self, unit: str) -> dict:
        """A unit's ``_maps.npz`` as a dict of arrays."""
        with np.load(self.path(unit, "_maps.npz")) as f:
            return dict(f)

    def weights(self, block: str) -> dict:
        """A block's ``_weights.npz`` as a dict of arrays."""
        with np.load(self.path(block, "_weights.npz")) as f:
            return dict(f)

    def line_maps(self) -> dict:
        """``{id: (dra, ddec, moment0)}`` of the fitted lines."""
        with np.load(self.output / "line_maps.npz") as f:
            ids = sorted({int(k.split("_")[0]) for k in f.files})
            return {i: tuple(f[f"{i}_{k}"] for k in ("dra", "ddec", "moment0")) for i in ids}

    def responses(self, rows: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Frequencies, template widths, and the data and jackknife responses at catalogue ``rows``."""
        out = []
        for r in rows:
            m = self.maps(r["unit"])
            k = int(np.flatnonzero(m["group_id"] == r["unit_id"])[0])
            out.append((m["freqs"], m["widths"], m["group_snr"][k], m["group_jk"][k]))
        return tuple(zip(*out, strict=True)) if out else ((), (), (), ())

    def chunks(self, block: str, fields=None, as_read: bool = False) -> list:
        """A block's exported chunks; ``as_read`` restores the measurement set's weights from ``_weights.npz``."""
        chunks = load(str(self.export_path(block)), fields).chunks
        if not as_read:
            return chunks
        w = self.weights(block)
        if not all(f"{c.spw}_scale" in w for c in chunks):
            raise ValueError(f"{block} has no weight scales: its data were not reweighted by this run (an NPZ input)")
        scales = {s: WeightScale(*(w[f"{s}_{k}"] for k in WeightScale._fields)) for s in {c.spw for c in chunks}}
        return [c.scaled(scales[c.spw].at(c.scan)) for c in chunks]

    # ----------------------------------------------------------------- reports
    def summary(self) -> str:
        """Detections, thresholds and noise checks, printed and returned."""
        done = [u for u in self.units if self.path(u, ".ecsv").exists()]
        lines = [f"{self.config['name']}: {len(done)} of {len(self.units)} units searched ({self.output})"]
        if (self.output / "catalogue.ecsv").exists():
            cat, m = self.catalogue, self.catalogue.meta
            lines.append(
                f"  {int(cat['detected'].sum())} of {len(cat)} groups detected (rule {m['rule']}): "
                f"likelihood ratio {m['k']:g} from S/N {m['snr_likelihood']:.2f}, fidelity "
                f"{m['fidelity_min']:g} from S/N {m['snr_fidelity']:.2f}; noise tail sigma {m['noise_sigma']:.3f}"
            )
            spread = np.concatenate([p["jackknife_spread"] for p in m["parts"]])
            lines.append(f"  jackknife S/N spread {spread.min():.2f}-{spread.max():.2f} (1 under H0)")
            if note := bright_lines(cat):
                lines.append(f"  {note}")
        noise = [self.weights(b) for b in self.plan["blocks"] if self.path(b, "_weights.npz").exists()]
        scale = [v for w in noise for k, v in w.items() if k.endswith("_noise_scale")]
        if scale:
            lines.append(
                f"  channel noise over the weights, on the jackknife: "
                f"{min(map(np.min, scale)):.2f}-{max(map(np.max, scale)):.2f} (1 where the weights hold)"
            )
        if (self.output / "lines_fit.ecsv").exists():
            fit = self.lines
            lines.append(
                f"  fitted {int(np.isfinite(fit['fit_dra']).sum())} lines "
                f"({int(fit['near_miss'].sum())} near misses, {int((fit['duplicate_of'] > 0).sum())} duplicates)"
            )
        text = "\n".join(lines)
        print(text)
        return text

    @cached_property
    def plot(self) -> RunPlots:
        """Figures from the saved products: ``run.plot.spectra()``, ``.maps()``, ``.responses()``, ..."""
        return RunPlots(self)

    def diagnostics(self, plots_dir=None) -> list:
        """Do the weights predict the scatter? :meth:`RunPlots.weights` for every window of every exported block."""
        return [
            self.plot.weights(block, spw=spw, plots_dir=plots_dir)
            for block, spws in self.plan["blocks"].items()
            if self.export_path(block).exists()
            for spw in spws
        ]


class RunPlots:
    """The figures of a :class:`Run`, each from its saved products only; every method returns the figure."""

    def __init__(self, run: Run):
        self.run = run

    def _dir(self, plots_dir):
        return None if plots_dir is False else str(plots_dir or self.run.output / "plots")

    def reliability(self, plots_dir=None):
        from .plotting import reliability_check

        return reliability_check(self.run.catalogue, plots_dir=self._dir(plots_dir), name="reliability")

    def _rows(self, which):
        fit = self.run.lines
        rows = np.asarray(fit["near_miss"] if which == "near" else fit["detected"] & (fit["duplicate_of"] == 0))
        return fit[rows & np.isfinite(fit["fit_dra"])]

    def spectra(self, which: str = "detected", plots_dir=None):
        """Line profiles of the fitted detections (or ``which="near"``: the near misses)."""
        from .plotting import line_spectra_check

        return line_spectra_check(self._rows(which), plots_dir=self._dir(plots_dir), name=f"{which}_spectra")

    def maps(self, which: str = "detected", plots_dir=None):
        """Moment-0 maps of the fitted detections (or the near misses)."""
        from .plotting import line_maps_check

        rows, maps = self._rows(which), self.run.line_maps()
        return line_maps_check(
            [maps[i] for i in rows["id"]], rows, plots_dir=self._dir(plots_dir), name=f"{which}_maps"
        )

    def responses(self, which: str = "detected", plots_dir=None):
        """Filter response (S/N against frequency, every template) at each detection, with the jackknife's."""
        from .plotting import filter_responses, passes

        rows = self._rows(which)
        freqs, widths, snr, jk = self.run.responses(rows)
        labels = [f"#{r['id']} {r['unit']}" for r in rows]
        return filter_responses(
            freqs,
            snr,
            widths,
            labels,
            jk=jk,
            at=np.asarray(rows["freq_ghz"]) * 1e9,
            passed=passes(rows),
            plots_dir=self._dir(plots_dir),
            name=f"{which}_responses",
        )

    def snr_map(self, plots_dir=None):
        """Peak S/N over templates and channels, every unit on one map, the detections marked."""
        from .plotting import peak_snr_map

        maps = [self.run.maps(u) for u in self.run.units]
        cat = self.run.catalogue
        marks = {f"#{r['id']}": (r["dra"], r["ddec"]) for r in cat[cat["detected"]]}
        return peak_snr_map(
            [(m["dra"], m["ddec"], m["peak"]) for m in maps],
            marks=marks,
            plots_dir=self._dir(plots_dir),
            name="peak_snr",
        )

    def weights(self, block: str, spw: int | None = None, field: int | None = None, plots_dir=None):
        """Do the weights predict the scatter? One window of a block (default its first): one field as read
        and reweighted, and all fields pooled."""
        from .plotting import weights_check

        run = self.run
        pooled = run.chunks(block)
        spw = pooled[0].spw if spw is None else spw
        field = min(c.field for c in pooled if c.spw == spw) if field is None else field
        (fixed,) = [c for c in pooled if c.spw == spw and c.field == field]
        (raw,) = [c for c in run.chunks(block, [field], as_read=True) if c.spw == spw]
        return weights_check(
            [raw, fixed],
            ["as read", "reweighted"],
            plots_dir=self._dir(plots_dir),
            name=f"weights_{block}_spw{spw:02d}_field{field}",
            pooled=[c for c in pooled if c.spw == spw],
        )


__all__ = ["Run", "RunPlots", "Unit", "duplicates", "unit_name", "unit_plan", "window_units"]
