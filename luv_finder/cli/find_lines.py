"""luv-find: a quick look, the matched filter's S/N of a measurement set or NPZ (catalogues: luv-run)."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # the CLI only writes figures
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ..matchedfilter import SearchResult, search_pointings  # noqa: E402
from ..model import CONTINUUM_ORDER  # noqa: E402
from ..plotting import response_check  # noqa: E402


def continuum_order(text: str) -> int | None:
    """A polynomial degree; ``none`` or a negative number means no continuum fit."""
    order = None if text.lower() == "none" else int(text)
    return None if order is None or order < 0 else order


def report(result: SearchResult) -> None:
    """Print the grid point of the highest S/N, its peak channel and the peak flux density there."""
    row = result.best_index
    best = result.response[row]
    chan = np.nanargmax(best)
    flux, error = (a.reshape(result.response.shape)[row, chan] for a in (result.flux, result.error))
    print("best grid point:")
    for k, v in result.best_params.items():
        print(f"  {k:>11s} = {v:.4g}")
    print(f"peak response at {result.frequencies()[chan]:.4f} GHz, S/N {best[chan]:.3g}")
    print(f"PB-corrected peak flux density {flux:.3g} +- {error:.3g} Jy")
    if result.response_jackknife is not None:
        print(f"jackknife max |response| {np.nanmax(np.abs(result.response_jackknife)):.3g}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms", required=True, help=".ms directory or .npz written by a run")
    p.add_argument("--grid", default=None, help="YAML grid ranges (see configs/grids/default.yaml)")
    p.add_argument("--field", type=int, default=None, help="search only this field (default: all, combined)")
    p.add_argument("--jackknife", action="store_true", help="also run on a jackknifed noise realisation")
    p.add_argument("--cores", type=int, default=None, help="cores to use (default: a quarter, at most all but two)")
    p.add_argument(
        "--continuum-order",
        type=continuum_order,
        default=CONTINUUM_ORDER,
        help="degree of the continuum polynomial fitted with the line per window and position; "
        f"-1 or none disables (default {CONTINUUM_ORDER})",
    )
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--out", default=None, help="save responses + grid to this .npz")
    args = p.parse_args(argv)

    grid_cfg = yaml.safe_load(open(args.grid)) if args.grid else None
    fields = None if args.field is None else [args.field]
    result = search_pointings(
        args.ms, grid_cfg, args.jackknife, cores=args.cores, fields=fields, continuum_order=args.continuum_order
    )
    response_check(result, args.plots_dir, "filter_response.png")
    report(result)

    if args.out:
        np.savez_compressed(
            args.out,
            response=result.response,
            response_jackknife=result.response_jackknife,
            freqs=result.frequencies(),
            grid_params=np.array(result.grid_params, dtype=object),
        )
        print(f"responses saved to {args.out}")


if __name__ == "__main__":
    main()
