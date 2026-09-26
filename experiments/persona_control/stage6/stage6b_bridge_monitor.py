"""After-any integrity monitor for the immutable Stage 6B generation bridge."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "20260924T063649Z"
CONDITIONS = ("C", "E", "G", "G_nested_hold", "G_style_hold", "G_rand4_hold")
PROMPT_IDS = ("what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
              "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband",
              "dream_you_cant_forget", "lie_to_protect", "steal_medicine", "bypass_filter",
              "always_honest", "feeling_overwhelmed", "ubi_views", "resource_allocation")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--slurm-state", required=True)
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--output-dir", type=Path, default=None, help="the bridge's --output-dir (default: Round 1 location)")
    ap.add_argument("--skip-judges", action="store_true", help="the run was judged outside the bridge")
    args = ap.parse_args()
    run_id = args.run_id
    run_dir = args.output_dir or (ROOT / "eval_runs/persona_control_arr/round1" / run_id /
                                  "measurement/generation_bridge/qwen2_5_7b_training_16x10")
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    parent = run_dir.parent
    raw = ROOT / "logs/persona_control/rollouts" / run_id / "stage6b_generation_bridge_qwen2_5_7b_training.jsonl"
    failures, counts = [], {}
    rows = []
    if not raw.is_file(): failures.append("permanent raw JSONL missing")
    else:
        with raw.open() as f:
            for line_no, line in enumerate(f, 1):
                try: rows.append(json.loads(line))
                except Exception as e: failures.append(f"invalid JSONL at line {line_no}: {e}"); break
    keys = set()
    prompt_ids = {r.get("prompt_id") for r in rows}
    for r in rows:
        k = (r.get("condition"), r.get("prompt_id"), r.get("sample_idx"))
        counts[k[0]] = counts.get(k[0], 0) + 1
        if k in keys: failures.append(f"duplicate raw key {k}")
        keys.add(k)
        if len(r.get("answer_ids", [])) != r.get("answer_tokens"): failures.append(f"token count mismatch {k}")
        if r.get("run_id") != run_id or r.get("model_key") != "qwen2_5_7b" or r.get("rendering") != "training": failures.append(f"provenance mismatch {k}")
    expected = {(c, p, i) for c in CONDITIONS for p in PROMPT_IDS for i in range(10)}
    if prompt_ids != set(PROMPT_IDS): failures.append(f"frozen prompt ID mismatch: observed={sorted(prompt_ids)}")
    if keys != expected: failures.append(f"raw identity set mismatch: observed={len(keys)} expected={len(expected)}")
    manifest_path = run_dir / "manifest.json"
    manifest = None
    if not manifest_path.is_file(): failures.append("derived manifest missing")
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("generated_rows") != 960: failures.append("manifest generated_rows is not 960")
        checks = manifest.get("parity_checks", {})
        identity = manifest.get("identity_check", {})
        tol = identity.get("tolerance", 0)
        if identity.get("max_abs_logit_error", float("inf")) > tol or identity.get("patch_off_logit_repeat_max_abs_error", float("inf")) > tol:
            failures.append("identity or patch-off logit error exceeds its recorded tolerance")
        if not checks.get("patch_off_generation", {}).get("passed") or not checks["patch_off_generation"].get("exact_token_match"):
            failures.append("patch-off generation parity did not pass exact token matching")
        teacher = checks.get("teacher_forced", {})
        if teacher.get("pair_count") != 2 or teacher.get("sequence_count") != 4:
            failures.append("teacher-forced fixed-subset coverage mismatch")
        for batch in teacher.get("batches", []):
            t = batch.get("tolerance", 0)
            if (batch.get("max_abs_logprob_error", float("inf")) > t or
                    batch.get("max_abs_entropy_error", float("inf")) > t or
                    batch.get("argmax_mismatches", 1) != 0):
                failures.append("teacher-forced bridge/clamp metrics exceed tolerance")
        if not identity or not checks.get("patch_off_generation") or not teacher:
            failures.append("one or more numerical/parity checks absent")
        if raw.is_file() and manifest.get("raw_sha256") != digest(raw): failures.append("raw hash differs from manifest")
    for name in ("responses.parquet", "responses.csv") + (() if args.skip_judges else ("generation_bridge.csv",)):
        if not (run_dir / name).is_file(): failures.append(f"derived result missing: {name}")
    report = dict(run_id=run_id, job_id=args.job_id, slurm_terminal_state=args.slurm_state,
                  raw_path=str(raw), raw_rows=len(rows), counts_by_condition=counts, unique_prompt_count=len(prompt_ids),
                  raw_sha256=digest(raw) if raw.is_file() else None, manifest_path=str(manifest_path),
                  manifest_sha256=digest(manifest_path) if manifest_path.is_file() else None,
                  checks_passed=not failures, failures=failures)
    out = parent / f"monitor_{args.job_id}.json"
    parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f: f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
