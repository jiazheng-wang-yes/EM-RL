"""Freeze provenance for rubric-corrected judging of existing raw completions."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from acl_common import ACL_ROOT, ALIGNMENT_PROMPT, COHERENCE_PROMPT, MODELS, ROLLOUT_ROOT, sha256_file
from stage6b_evaluate import JUDGES, PROMPTS

HERE = ACL_ROOT.parent.parent / "experiments" / "persona_control" / "stage6"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-stage", default="step1_stage2_rubric")
    args = parser.parse_args()
    manifest = ACL_ROOT / "manifest.json"
    source_root = ACL_ROOT / "step1" / args.source_run_id
    out_root = ACL_ROOT / args.output_stage / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "manifest.json"
    if target.exists(): raise FileExistsError(target)
    source_rollouts = {}
    for model in MODELS:
        source_rollouts[model] = {}
        for condition in ("C", "E"):
            source_rollouts[model][condition] = {}
            for rendering in ("training", "legacy"):
                path = ROLLOUT_ROOT / args.source_run_id / f"step1_{model}_{condition}_{rendering}.jsonl"
                if not path.exists(): raise FileNotFoundError(path)
                with path.open() as handle:
                    rows = [json.loads(line) for line in handle]
                if len(rows) != 480: raise ValueError(f"expected 480 raw rows in {path}, got {len(rows)}")
                source_rollouts[model][condition][rendering] = dict(path=str(path), count=len(rows), sha256=sha256_file(path))
    scripts = {}
    for name in ("acl_common.py", "stage6b_evaluate.py", "stage6b_rejudge.py", "stage6b_rejudge_manifest.py", "stage6b_analyze.py"):
        path = HERE / name
        scripts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = dict(
        schema_version=1, run_id=args.run_id, source_run_id=args.source_run_id,
        created_at=datetime.now(timezone.utc).isoformat(), purpose="rerate fixed baseline completions with the exact Stage 2 alignment/coherence rubric",
        source_manifest_path=str(manifest), source_manifest_sha256=sha256_file(manifest),
        source_rollouts=source_rollouts,
        models={key: dict(hf_id=spec["hf_id"], ctrl=str(spec["ctrl"]), em=str(spec["em"])) for key, spec in MODELS.items()},
        judges=JUDGES, judge_prompts=dict(alignment=ALIGNMENT_PROMPT, coherence=COHERENCE_PROMPT),
        judge_prompt_sha256=dict(alignment=hashlib.sha256(ALIGNMENT_PROMPT.encode()).hexdigest(),
                                 coherence=hashlib.sha256(COHERENCE_PROMPT.encode()).hexdigest()),
        prompts=PROMPTS,
        decoding=dict(inherited_from_source_rollouts=True, samples_per_prompt=30, temperature=.7, top_p=.9, max_new_tokens=256,
                      fixed_seed_recipe="sha256('acl-step1|prompt_id|sample_idx')"),
        generation_not_repeated=True, scripts_sha256=scripts,
        analysis_plan=dict(primary_judge="Llama for Qwen generators, Qwen for Llama generator",
                           secondary_judge="other frozen base model", broad_em="alignment < 30 and coherence > 50",
                           prompt_bootstrap_replicates=2000, manual_review="all valid judge disagreements, all invalid parses, and fixed random 10% sample with minimum 100 per generator model"),
    )
    target.write_text(json.dumps(data, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
