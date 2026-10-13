#!/usr/bin/env python3
"""Pick activation files suitable for hacking-direction discovery."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch


def step_from_path(path: Path) -> int:
    m = re.search(r"step[_-]?(\d+)", path.stem)
    if not m:
        raise SystemExit(f"Cannot parse step from {path.name}")
    return int(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acts-dir", required=True, type=Path)
    p.add_argument("--min-per-class", type=int, default=8)
    p.add_argument("--min-step", type=int, default=192)
    args = p.parse_args()

    selected: list[Path] = []
    for path in sorted(args.acts_dir.glob("acts_step_*.pt"), key=step_from_path):
        if step_from_path(path) < args.min_step:
            continue
        pl = torch.load(path, map_location="cpu", weights_only=False)
        nh = int((pl["labels"] == 1).sum())
        nhon = int((pl["labels"] == 0).sum())
        if nh >= args.min_per_class and nhon >= args.min_per_class:
            selected.append(path)

    if not selected:
        best: tuple[int, Path] | None = None
        for path in sorted(args.acts_dir.glob("acts_step_*.pt"), key=step_from_path):
            if step_from_path(path) < args.min_step:
                continue
            pl = torch.load(path, map_location="cpu", weights_only=False)
            nh = int((pl["labels"] == 1).sum())
            nhon = int((pl["labels"] == 0).sum())
            score = min(nh, nhon)
            if score < args.min_per_class:
                continue
            if best is None or score > best[0]:
                best = (score, path)
        if best is not None:
            selected = [best[1]]

    sys.stdout.write(" ".join(str(p) for p in selected))


if __name__ == "__main__":
    main()
