"""luv-find: grid-search matched filter for spectral lines in a measurement set or NPZ."""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from ..catalogue import catalogue
from ..data import fields_in, load
from ..fit import fit_lines
from ..matchedfilter import MatchedFilter, SearchResult, build_grid, grid_model, pb_corrected, search_pointings
from ..plotting import reliability_check, response_check

#: Catalogue columns printed for the detected lines.
SHOWN = ("id", "dra", "ddec", "freq_ghz", "width", "snr", "line_flux", "likelihood", "fidelity")

#: Fitted columns printed with ``--fit``.
FITTED = ("id", "fit_dra", "fit_ddec", "fit_freq_ghz", "fit_width", "fit_bmaj", "fit_line_flux", "fit_line_flux_error")


def continuum_order(text: str) -> int | None:
    """A polynomial degree; ``none`` or a negative number means no continuum fit."""
    order = None if text.lower() == "none" else int(text)
    return None if order is None or order < 0 else order


def report(result: SearchResult, corrected: bool) -> None:
    """Print the grid point of the highest S/N, its peak channel and the peak flux density there."""
    row = result.best_index
    best = result.response[row]
    chan = np.nanargmax(best)
    flux, error = (a.reshape(result.response.shape)[row, chan] for a in (result.flux, result.error))
    print("best grid point:")
    for k, v in result.best_params.items():
        print(f"  {k.split('_', 2)[-1]:>11s} = {v:.4g}")
    print(f"peak response at {result.frequencies()[chan]:.4f} GHz, S/N {best[chan]:.3g}")
    print(f"{'PB-corrected ' * corrected}peak flux density {flux:.3g} +- {error:.3g} Jy")
    if result.response_jackknife is not None:
        print(f"jackknife max |response| {np.nanmax(np.abs(result.response_jackknife)):.3g}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms", required=True, help=".ms directory or .npz from luv-export")
    p.add_argument("--grid", default=None, help="YAML grid ranges (see configs/grids/default.yaml)")
    p.add_argument(
        "--field",
        type=int,
        default=None,
        help="search only this field, PB-corrected only for --catalogue (default: all, combined if several)",
    )
    p.add_argument("--jackknife", action="store_true", help="also run on a jackknifed noise realisation")
    p.add_argument("--cores", type=int, default=None, help="cores to use (default: a quarter, at most all but two)")
    p.add_argument(
        "--continuum-order",
        type=continuum_order,
        default=2,
        help="degree of the continuum polynomial fitted with the line per window and position; "
        "-1 or none disables (default 2)",
    )
    p.add_argument(
        "--catalogue",
        default=None,
        help="write the line candidates, with likelihood ratio and fidelity, to this ECSV (needs --jackknife)",
    )
    p.add_argument(
        "--fit",
        action="store_true",
        help="fit the detected lines in the visibilities (Gaussian source, joint continuum) and add the fitted "
        "columns to the --catalogue ECSV",
    )
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--out", default=None, help="save responses + grid to this .npz")
    args = p.parse_args(argv)
    if args.catalogue and not args.jackknife:
        p.error("--catalogue needs --jackknife: the jackknife is the noise reference")
    if args.fit and not args.catalogue:
        p.error("--fit needs --catalogue: it fits the catalogued lines")

    grid_cfg = yaml.safe_load(open(args.grid)) if args.grid else None
    fields = fields_in(args.ms) if args.field is None else [args.field]
    mosaic = len(fields) > 1
    if mosaic:
        result, _ = search_pointings(
            args.ms, grid_cfg, args.jackknife, cores=args.cores, continuum_order=args.continuum_order
        )
    else:
        data = load(args.ms, fields)
        mf = MatchedFilter(data, grid_model(build_grid(data, grid_cfg)), continuum_order=args.continuum_order)
        mf.run(jackknife=args.jackknife, cores=args.cores)
        result = pb_corrected(mf.result, data) if args.catalogue else mf.result

    response_check(result, args.plots_dir, "filter_response.png")
    report(result, corrected=mosaic or bool(args.catalogue))

    if args.catalogue:
        cat = catalogue(result, ref=load(args.ms, []).metadata.ref)
        if args.fit:
            cat = fit_lines(args.ms if mosaic else data, cat, cores=args.cores, continuum_order=args.continuum_order)
        cat.write(args.catalogue, format="ascii.ecsv", overwrite=True)
        reliability_check(cat, args.plots_dir, "reliability")
        print(f"{cat['detected'].sum()} of {len(cat)} candidates detected; catalogue saved to {args.catalogue}")
        cat[cat["detected"]][SHOWN].pprint(max_lines=-1, max_width=-1)
        if args.fit:
            cat[cat["detected"]][FITTED].pprint(max_lines=-1, max_width=-1)

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
