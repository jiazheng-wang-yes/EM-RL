"""Shared streaming-resume helpers for the data_generation pipelines.

Each pipeline writes verified / accepted records to an append-only JSONL as it
works, so a SLURM time-limit kill loses at most the row currently mid-write
(one ``write()`` of a single JSON line is atomic on POSIX with O_APPEND, and we
``flush()`` + ``fsync()`` on every append). On restart, the pipeline reads the
file with ``load_jsonl_safe`` -- which silently drops any malformed trailing
line from a kill mid-write -- and skips work whose resume key is already
present.

Resume keys are pipeline-specific. The active rh-paper SFT distillation
pipeline uses ``problem_id``.

The helpers themselves are key-agnostic; each pipeline owns its own skip logic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Round-trip ``value`` through ``json.dumps(default=str)`` so unknown
    objects (e.g. exceptions, numpy scalars) become JSON-serialisable strings.
    """
    return json.loads(json.dumps(value, default=str))


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append one JSON record to ``path`` and flush to disk.

    A single ``write()`` of one short JSON line is atomic on POSIX with
    O_APPEND, so concurrent restarts cannot interleave half-written rows. The
    final newline is part of the same write, so the line is either fully
    present or absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(record), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl_safe(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, ignoring blank lines and any malformed trailing line.

    A SLURM time-limit kill can truncate the last line; we drop unparsable
    lines so resume sees only fully-flushed records.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
