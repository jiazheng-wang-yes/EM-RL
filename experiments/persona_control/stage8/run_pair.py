"""Stage 8 single-job driver (plan Section 3, "Development pair"): the measurement steps of one or more twins, in
order, inside one Slurm job, so that no job waits on another.

Each step is one call of a Stage 8 script in a subprocess with the same interpreter.  A step whose output file
exists is skipped; Step 1b and the Step 2 causal test, when their scoring rows were saved by an earlier run that
failed later, are finished from those rows (--reuse-rows).  The first failing step stops the run (non-zero exit).

  core   Steps 1-3 of a core pair (plan Section 4, Step 5):
         s1a s1b s2stats s2analyze s2causal s3answers s3reads s3baserand s3swap s3scaled s3analyze
  step4  the Step 4 measurements of a changed-data twin (Step 1 items 1-3, Step 2 items 1-4, Step 3b's read summary
         on the 3a prompts): s1a s1b s2stats s2analyze s2causal s3answers s3reads s3baserand s3analyze
  --renderings inject,trigger  also s3answers s3reads s3baserand s3analyze on those renderings of the 3a prompts
         (s3_reads.py --rendering), after the listed steps.
  --steps may also list single steps (comma-separated).
  --skip-update-cosine  passed to Step 2 stats: the prediction then uses only the base model and the training set, so
         it can run before the twin is trained (the cosine with the real update is skipped).

Usage (repository root):
  experiments/persona_control/stage8/submit.sh --job-name=pc8_pair_MED_qwen2_5_7b_inject_s42 \\
      experiments/persona_control/stage8/s8_gpu.sbatch run_pair.py --twins MED:qwen2_5_7b:inject_s42 --steps step4 \\
      --renderings inject
"""

import argparse
import json
import os
import subprocess
import sys
import time

from common8 import OUT_ROOT, log

PRESETS = {
    "core": "s1a s1b s2stats s2analyze s2causal s3answers s3reads s3baserand s3swap s3scaled s3analyze".split(),
    "step4": "s1a s1b s2stats s2analyze s2causal s3answers s3reads s3baserand s3analyze".split(),
}
RENDERED = ("s3answers", "s3reads", "s3baserand", "s3analyze")
# step -> (script, part, output directory under the twin, output file, file whose presence means "finish from rows")
STEPS = {
    "s1a": ("s1_components.py", "a", "s1_components", "rows.parquet", None),
    "s1b": ("s1_components.py", "b", "s1_components", "astar.json", "curve_rows.parquet"),
    "s2stats": ("s2_origin.py", "stats", "s2_origin", "stats.pt", None),
    "s2analyze": ("s2_origin.py", "analyze", "s2_origin", "analysis.json", None),
    "s2causal": ("s2_origin.py", "causal", "s2_origin", "causal.json", "causal_rows.parquet"),
    "s3answers": ("s3_reads.py", "answers", "s3_reads", "c_answers.json", None),
    "s3reads": ("s3_reads.py", "reads", "s3_reads", "reads.pt", None),
    "s3baserand": ("s3_reads.py", "baserand", "s3_reads", "base_random_proj.pt", None),
    "s3swap": ("s3_reads.py", "swap", "s3_reads", "swap_rows.parquet", None),
    "s3scaled": ("s3_reads.py", "scaled", "s3_reads", "scaled_rows.parquet", None),
    "s3analyze": ("s3_reads.py", "analyze", "s3_reads", "analysis.json", None),
}


def plan(steps, renderings):
    out = [(s, "plain") for s in steps]
    for r in renderings:
        out += [(s, r) for s in RENDERED if s in steps]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twins", required=True, help="dataset:model:twin,...")
    ap.add_argument("--steps", default="step4", help="a preset (core, step4) or comma-separated steps")
    ap.add_argument("--renderings", default="", help="extra 3a renderings for the s3 steps: inject, trigger")
    ap.add_argument("--skip-update-cosine", action="store_true", help="Step 2 stats without the trained twin")
    args = ap.parse_args()
    steps = PRESETS.get(args.steps) or args.steps.split(",")
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"unknown steps {unknown}")
    renderings = [r for r in args.renderings.split(",") if r]
    here = os.path.dirname(os.path.abspath(__file__))
    t_all = time.time()
    record = []
    for spec in args.twins.split(","):
        dataset, model, twin = spec.split(":")
        for step, rendering in plan(steps, renderings):
            script, part, sub, target, rows = STEPS[step]
            if rendering != "plain":
                sub = f"{sub}_{rendering}"
            d = os.path.join(OUT_ROOT, dataset, model, twin, sub)
            if os.path.exists(os.path.join(d, target)):
                log(f"{spec} {step} {rendering}: {target} exists, skipped")
                continue
            cmd = [sys.executable, os.path.join(here, script), part, "--dataset", dataset, "--model", model, "--twin", twin]
            if script == "s3_reads.py":
                cmd += ["--rendering", rendering]
            if step == "s2stats" and args.skip_update_cosine:
                cmd.append("--skip-update-cosine")
            if rows and os.path.exists(os.path.join(d, rows)):
                cmd.append("--reuse-rows")
            log(f"{spec} {step} {rendering}: {' '.join(cmd[1:])}")
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=here).returncode
            record.append(dict(twin=spec, step=step, rendering=rendering, returncode=rc, seconds=round(time.time() - t0, 1)))
            if rc != 0:
                log(f"{spec} {step} {rendering}: failed with exit code {rc}; stopping")
                print(json.dumps(record, indent=1))
                sys.exit(rc)
    log(f"all steps done ({time.time() - t_all:.0f}s)")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
