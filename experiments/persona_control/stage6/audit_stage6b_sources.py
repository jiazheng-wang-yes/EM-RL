"""Post-hoc content audit for frozen ACL Step 1 generator endpoints.

This does not modify the immutable Step 1 manifests or any source checkpoint.
It hashes checkpoint and tokenizer files now, records the original run metadata,
and explicitly marks the content hashes as post-hoc evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from acl_common import ACL_ROOT, MODELS, ROOT, sha256_file

RUN_ID = "acl_step1_20260922_v1"
OUTPUT = ACL_ROOT / "source_checkpoint_audits" / f"{RUN_ID}.json"
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
TOKENIZER_NAMES = {
    "added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
}


def role(path: Path) -> str:
    if path.suffix in WEIGHT_SUFFIXES:
        return "model_weight"
    if path.name in TOKENIZER_NAMES:
        return "tokenizer_or_chat_template"
    if path.name in {"config.json", "generation_config.json", "model.safetensors.index.json"}:
        return "model_config_or_index"
    return "other_source_file"


def file_record(path: Path) -> dict:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    unchanged_during_hash = (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    )
    if not unchanged_during_hash:
        raise RuntimeError(f"source file changed while hashing: {path}")
    return dict(
        relative_path=path.name,
        role=role(path),
        bytes=before.st_size,
        mtime_utc=datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
        ctime_utc=datetime.fromtimestamp(before.st_ctime, timezone.utc).isoformat(),
        sha256=digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"this audit is scoped to the frozen Step 1 run {RUN_ID}")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"immutable source audit already exists: {output}")

    frozen_manifest = ACL_ROOT / "manifest.json"
    result = dict(
        schema_version=1,
        run_id=RUN_ID,
        audit_started_at=datetime.now(timezone.utc).isoformat(),
        audit_type="post_hoc_checkpoint_content_hashes",
        post_hoc=True,
        interpretation=(
            "Hashes identify the endpoint files present at audit time. They were not captured by the original generation jobs; "
            "mtime/ctime comparisons are recorded to assess whether the files predate the run."
        ),
        source_manifest=dict(path=str(frozen_manifest), sha256=sha256_file(frozen_manifest)),
        audit_script=dict(path=str(Path(__file__).resolve()), sha256=sha256_file(Path(__file__))),
        endpoints={},
    )

    for model_key, spec in MODELS.items():
        model_entry = dict(hf_id=spec["hf_id"], endpoints={})
        for condition, source_key in (("C", "ctrl"), ("E", "em")):
            source = Path(spec[source_key]).resolve()
            run_path = ACL_ROOT / "step1" / model_key / RUN_ID / condition / "run.json"
            run = json.loads(run_path.read_text())
            if Path(run["source"]).resolve() != source:
                raise ValueError(f"source-path mismatch in {run_path}: {run['source']} != {source}")
            if run.get("model") != model_key or run.get("condition") != condition:
                raise ValueError(f"run identity mismatch in {run_path}")

            files = sorted(path for path in source.iterdir() if path.is_file())
            if not files:
                raise FileNotFoundError(f"no source files in {source}")
            created = float(run["created"])
            print(f"Hashing {model_key}:{condition} ({len(files)} files)", flush=True)
            records = []
            for path in files:
                records.append(file_record(path))
                print(f"  {path.name}: {records[-1]['bytes']} bytes", flush=True)

            config = json.loads((source / "config.json").read_text())
            model_entry["endpoints"][condition] = dict(
                path=str(source),
                local_config_commit=config.get("_commit_hash"),
                run_json_path=str(run_path),
                run_created_epoch=created,
                run_created_utc=datetime.fromtimestamp(created, timezone.utc).isoformat(),
                every_file_timestamp_precedes_run_completion=all(
                    datetime.fromisoformat(record["mtime_utc"]).timestamp() <= created
                    and datetime.fromisoformat(record["ctime_utc"]).timestamp() <= created
                    for record in records
                ),
                file_count=len(records),
                files=records,
            )
        result["endpoints"][model_key] = model_entry

    # Judge revisions were recorded by each completed rejudge job. Keep them
    # with this audit so checkpoint hashes and actual judge revisions are
    # available together without changing the frozen run files.
    result["judge_revisions"] = {}
    for model_key in MODELS:
        for condition in ("C", "E"):
            run_path = (ACL_ROOT / "step1_stage2_rubric" / model_key / "acl_step1_stage2rubric_20260922_v1"
                        / condition / "run.json")
            run = json.loads(run_path.read_text())
            result["judge_revisions"][f"{model_key}:{condition}"] = run["judges"]

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, output)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
