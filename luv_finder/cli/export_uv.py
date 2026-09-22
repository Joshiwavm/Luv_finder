"""luv-export: flatten a measurement set into a CASA-free NPZ file."""

from __future__ import annotations

import argparse

from ..data import DataHandler


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms", required=True)
    p.add_argument("--out", required=True, help="output .npz path")
    args = p.parse_args(argv)

    data = DataHandler(args.ms)
    data.to_npz(args.out)
    print(f"{data.n_visbs(data.uvdata)} visibilities x {data.n_freqs(data.uvdata)} channels -> {args.out}")


if __name__ == "__main__":
    main()
