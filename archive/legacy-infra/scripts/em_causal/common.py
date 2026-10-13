from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "em_causal_v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Empty input: {path}")
    return rows


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Scientific artifacts are immutable. A failed run gets a new output path.
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_rows(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def manifest(**kwargs) -> dict:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {
        "format": FORMAT,
        "git_revision": rev.stdout.strip() if rev.returncode == 0 else None,
        "python": platform.python_version(),
        "source_hashes": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        **kwargs,
    }


def validate_rows(rows: list[dict]) -> None:
    ids: set[str] = set()
    groups: dict[str, str] = {}
    templates: dict[str, str] = {}
    for row in rows:
        for key in ("id", "group_id", "split", "messages"):
            if key not in row:
                raise ValueError(f"Missing {key}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate id: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in {"discovery", "validation", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")
        for table, key in ((groups, "group_id"), (templates, "template_id")):
            if key in row:
                previous = table.setdefault(row[key], row["split"])
                if previous != row["split"]:
                    raise ValueError(f"Split leakage in {key}: {row[key]}")


def select_rows(rows: list[dict], split: str, limit: int | None = None) -> list[dict]:
    """Round-robin families and groups; keep complete counterfactual groups."""
    validate_rows(rows)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] == split:
            groups.setdefault(row["group_id"], []).append(row)
    if not groups:
        raise ValueError(f"No {split} rows")
    ordered = sorted(groups.values(), key=lambda g: (g[0].get("problem_index", 0), g[0].get("family", ""), g[0]["group_id"]))
    selected = []
    for group in ordered:
        if limit is not None and len(selected) + len(group) > limit:
            continue
        selected.extend(group)
    if not selected:
        raise ValueError("Limit is too small for one complete counterfactual group")
    return selected
