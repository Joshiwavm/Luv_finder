"""luv-mock: simulate a mock ALMA line observation from a YAML config."""

from __future__ import annotations

import argparse

from ..mock import MockObservation


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="YAML file with MockObservation keyword arguments")
    p.add_argument("--output-folder", default=None, help="override output_folder")
    p.add_argument("--plots-dir", default="plots")
    args = p.parse_args(argv)

    overrides = {"output_folder": args.output_folder} if args.output_folder else {}
    mock = MockObservation.from_yaml(args.config, **overrides)
    dest = mock.run_all(plots_dir=args.plots_dir)
    print(f"mock written to {dest}\n  noisy MS: {mock.ms_noisy}")


if __name__ == "__main__":
    main()
