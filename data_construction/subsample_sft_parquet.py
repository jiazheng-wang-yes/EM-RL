#!/usr/bin/env python3
"""Write a fixed-size random subsample of an SFT train parquet.

Used by the minimal-finance-dose study to build the repetition axis: an arm with
``N`` unique rows trained for ``E`` epochs takes the same number of optimizer steps
as an arm with ``N*E`` rows trained for one epoch, but re-reads a smaller set.

The subsample is drawn with a fixed seed and is *nested*: the first ``n`` rows of
the seeded permutation, so a 1,960-row subset is contained in a 2,940-row subset.
Nesting keeps two dose points from differing by which rows they happened to draw.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source train parquet")
    parser.add_argument("--output", type=Path, required=True, help="destination parquet")
    parser.add_argument("--rows", type=int, required=True, help="number of rows to keep")
    parser.add_argument("--seed", type=int, default=42, help="permutation seed")
    args = parser.parse_args()

    table = pq.read_table(args.input)
    if args.rows > table.num_rows:
        raise SystemExit(f"--rows {args.rows} exceeds the {table.num_rows} rows in {args.input}")

    order = list(range(table.num_rows))
    random.Random(args.seed).shuffle(order)
    subset = table.take(order[: args.rows])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f"{args.output.name}.tmp")
    pq.write_table(subset, tmp)
    tmp.replace(args.output)
    print(f"wrote {subset.num_rows} of {table.num_rows} rows to {args.output} (seed {args.seed})")


if __name__ == "__main__":
    main()
