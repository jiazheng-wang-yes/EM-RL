from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = [
    "bidirectional_steering.csv",
    "bidirectional_steering_judged.csv",
    "steering_sweep.csv",
    "steering_sweep_judged.csv",
    "*_steering_sweep.csv",
    "*_steering_sweep_judged.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove large raw mech-interp CSVs after summary analysis exists.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")
    required_paths = [Path(path).resolve() for path in args.require]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Refusing cleanup because required analysis files are missing: {missing}")

    patterns = args.pattern or DEFAULT_PATTERNS
    candidates = _matching_files(out_dir, patterns)
    removed: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for path in candidates:
        size = path.stat().st_size
        record = {"path": str(path), "size_bytes": int(size)}
        if args.dry_run:
            kept.append(record)
            continue
        path.unlink()
        removed.append(record)

    report = {
        "out_dir": str(out_dir),
        "dry_run": bool(args.dry_run),
        "patterns": patterns,
        "required": [str(path) for path in required_paths],
        "removed": removed,
        "kept": kept,
        "removed_count": len(removed),
        "removed_bytes": int(sum(row["size_bytes"] for row in removed)),
    }
    manifest = Path(args.manifest).resolve() if args.manifest else out_dir / "cleanup_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _matching_files(out_dir: Path, patterns: list[str]) -> list[Path]:
    matches: list[Path] = []
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            matches.append(path)
    return sorted(set(matches))


if __name__ == "__main__":
    main()
