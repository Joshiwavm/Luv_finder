"""luv-find: grid-search matched filter for spectral lines in a measurement set or NPZ."""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

from ..data import DataHandler
from ..matchedfilter import MatchedFilter
from ..model import Gaussian, Model


def load_data(path: str, field: int | None = None) -> DataHandler:
    fields = None if field is None else [field]
    return DataHandler.from_npz(path, fields) if path.endswith(".npz") else DataHandler(path, fields=fields)


def build_grid(data: DataHandler, cfg: dict | None) -> dict:
    """Grid ranges from YAML, positions in the dataset's sky frame (arcsec from the reference).

    Positions default to multiples of half the resolution, counted from the reference
    direction and covering ``fov_fraction`` of the primary beam around the loaded field. The
    lattice comes from dataset-level metadata, so every pointing of a mosaic gets the same
    points. Sizes default to a tenth of the resolution, the position angle to 0.
    """
    fov = data.metadata.primarybeamsize()
    res = data.metadata.minresolution()
    cfg = dict(cfg or {})
    if "total_flux" in cfg:
        raise ValueError(
            "total_flux is not a search axis: it cancels in the matched-filter kernel "
            "normalisation, so every value gives an identical response. Remove it from the grid file."
        )

    def rng(key, default):
        v = cfg.get(key, default)
        if isinstance(v, dict):
            return np.arange(v["start"], v["stop"], v["step"])
        return np.atleast_1d(v)

    half, step = cfg.get("fov_fraction", 0.4) * fov, res / 2

    def lattice(centre):
        return step * np.arange(np.ceil((centre - half) / step), np.floor((centre + half) / step) + 1)

    dra, ddec = data.chunks[0].offset
    return {
        "dra": rng("dra", lattice(dra)),
        "ddec": rng("ddec", lattice(ddec)),
        "bmin": rng("bmin", res / 10),
        "bmaj": rng("bmaj", res / 10),
        "pa": rng("pa", 0.0),
        "width": rng("width", [200.0, 300.0, 400.0]),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms", required=True, help=".ms directory or .npz from luv-export")
    p.add_argument("--grid", default=None, help="YAML grid ranges (see configs/grids/default.yaml)")
    p.add_argument("--field", type=int, default=None, help="field to search; required for a mosaic")
    p.add_argument("--jackknife", action="store_true", help="also run on a jackknifed noise realisation")
    p.add_argument("--cores", type=int, default=None, help="cores to use (default: a quarter, at most all but two)")
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--out", default=None, help="save responses + grid to this .npz")
    args = p.parse_args(argv)

    data = load_data(args.ms, args.field)
    grid_cfg = yaml.safe_load(open(args.grid)) if args.grid else None
    comp = Gaussian()
    comp.grid = build_grid(data, grid_cfg)
    mod = Model()
    mod.addcomponent(comp)

    mf = MatchedFilter(data, mod)
    mf.run(jackknife=args.jackknife, cores=args.cores)

    os.makedirs(args.plots_dir, exist_ok=True)
    mf.plot_response(os.path.join(args.plots_dir, "filter_response.png"))

    best = mf.best_params
    freqs = mf.frequencies()
    peak = freqs[np.argmax(mf.response[mf.best_index])]
    print("best grid point:")
    for k, v in best.items():
        print(f"  {k.split('_', 2)[-1]:>11s} = {v:.4g}")
    print(f"peak response at {peak:.4f} GHz, amplitude {mf.response[mf.best_index].max():.3g}")
    if mf.response_jackknife is not None:
        print(f"jackknife max |response| {np.abs(mf.response_jackknife).max():.3g}")

    if args.out:
        np.savez_compressed(
            args.out,
            response=mf.response,
            response_jackknife=mf.response_jackknife,
            freqs=freqs,
            grid_params=np.array(mf.grid_params, dtype=object),
        )
        print(f"responses saved to {args.out}")


if __name__ == "__main__":
    main()
