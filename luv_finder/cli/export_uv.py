"""luv-export: flatten a measurement set into a CASA-free NPZ file, one (field, spw) at a time."""

from __future__ import annotations

import argparse

from ..data import Metadata, iter_chunks, write_npz


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms", required=True)
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--field", type=int, default=None, help="export only this field (default: all)")
    args = p.parse_args(argv)

    metadata = Metadata.from_ms(args.ms)
    totals = [0, 0]

    def report(chunks):
        for chunk in chunks:
            n_chan, n_row = chunk.X.shape
            totals[0] += chunk.X.size
            totals[1] += chunk.nbytes
            print(f"field {chunk.field} spw {chunk.spw}: {n_row} rows x {n_chan} channels")
            yield chunk

    fields = None if args.field is None else [args.field]
    write_npz(args.out, metadata, report(iter_chunks(args.ms, metadata.ref, fields)))
    print(f"{totals[0] / 1e6:.1f} M visibility-channels, {totals[1] / 1e9:.2f} GB in memory -> {args.out}")


if __name__ == "__main__":
    main()
