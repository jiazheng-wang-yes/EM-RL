"""Freeze same-prompt SQL tasks shared by two independently rejected proxies."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "eval_runs/history_amplification/sql_environment/mechanism_assay_manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    if args.left not in source["mechanisms"] or args.right not in source["mechanisms"]:
        raise ValueError("Both mechanisms must be present in the frozen source manifest")
    left_ids = {entry["task_id"] for entry in source["mechanisms"][args.left]}
    right_ids = {entry["task_id"] for entry in source["mechanisms"][args.right]}
    task_ids = sorted(left_ids & right_ids)
    if not task_ids:
        raise ValueError("The requested mechanism pair has no shared task IDs")
    output = {
        "version": 1,
        "selection": (
            "Task IDs present in both frozen constructed-proxy mechanisms; "
            "selected from the source manifest without consulting model outcomes."
        ),
        "source_manifest": str(SOURCE.relative_to(ROOT)),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "mechanisms": [args.left, args.right],
        "task_ids": task_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "task_count": len(task_ids), "task_ids": task_ids}))


if __name__ == "__main__":
    main()
