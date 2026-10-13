"""``luv-run``: a line search from a run YAML (see :class:`luv_finder.run.Run`)."""

import argparse

import matplotlib


def main(argv=None):
    p = argparse.ArgumentParser(prog="luv-run", description=__doc__)
    p.add_argument("config", help="run YAML, or a run's output directory")
    p.add_argument("step", choices=["search", "merge", "fit", "summary", "diagnostics", "plots", "release"])
    p.add_argument("--workers", type=int, default=1, help="processes sharing the units (search)")
    p.add_argument("--no-merge", action="store_true", help="do not merge the units after searching")
    args = p.parse_args(argv)

    matplotlib.use("Agg")
    from pathlib import Path

    from ..run import Run

    run = Run.open(args.config) if Path(args.config).is_dir() else Run(args.config)
    if args.step == "search":
        run.search(workers=args.workers, merge=not args.no_merge)
    elif args.step == "merge":
        run.merge()
    elif args.step == "fit":
        run.fit()
    elif args.step == "summary":
        run.summary()
    elif args.step == "diagnostics":
        run.diagnostics()
    elif args.step == "plots":
        for figure in ("reliability", "spectra", "maps", "responses", "snr_map"):
            getattr(run.plot, figure)()
    else:
        print("released:", run.release() or "nothing")


if __name__ == "__main__":
    main()
