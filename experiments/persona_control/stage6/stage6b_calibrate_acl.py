"""Prespecified 16-step, no-EM coefficient selection for the ACL defense."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from acl_common import (
    ACL_ROOT, CALIBRATION_MODEL, CALIBRATION_SOURCE_FILES,
    DIRECT_CALIBRATION_GRID, MIN_RETAINED_LEARNING,
    PERSONA_CALIBRATION_MULTIPLIERS, PERSONA_DRIFT_REDUCTION_RANGE,
    MODELS, sha256_file,
)

HERE = Path(__file__).resolve().parent


def report_row(condition, metrics, *, lambda_d=None, lambda_p=None,
               multiplier=None, retained_learning=None, drift_reduction=None):
    return dict(
        condition=condition,
        lambda_d=lambda_d,
        lambda_p=lambda_p,
        lambda_p_multiplier=multiplier,
        harmful_train_loss=metrics.get("harmful_loss"),
        benign_train_loss=metrics.get("benign_loss"),
        harmful_val_nll=metrics.get("harmful_val_nll"),
        benign_val_nll=metrics.get("benign_val_nll"),
        retained_learning=retained_learning,
        persona_drift=metrics.get("base_persona_drift"),
        persona_drift_reduction=drift_reduction,
        direct_region_update_norm=metrics.get("actual_update_norm_direct"),
        outside_region_update_norm=metrics.get("actual_update_norm_outside"),
        suppressed_harmful_specific_gradient_fraction=(
            metrics.get("suppressed_harmful_specific_gradient_fraction")
        ),
    )


def launch(model, condition, seed, attempt_id, tag, d=0., p=0., device="cuda:0"):
    run_tag = f"{attempt_id}_{tag}"
    command = [sys.executable, str(HERE / "stage6b_train.py"), "--model", model, "--condition", condition,
               "--seed", str(seed), "--max-steps", "16", "--run-tag", run_tag,
               "--lambda-d", str(d), "--lambda-p", str(p), "--device", device]
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    root = ACL_ROOT / "training" / model / f"seed_{seed}" / run_tag
    return json.loads((root / "manifest.json").read_text()), json.loads((root / "metrics.json").read_text())[-1], root, command


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",choices=[CALIBRATION_MODEL],required=True); ap.add_argument("--seed",type=int,default=61791); ap.add_argument("--attempt-id",required=True); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args()
    if args.seed != 61791:
        raise ValueError("the calibration seed is fixed at 61791")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt_id):
        raise ValueError("--attempt-id may contain only letters, digits, underscores, and hyphens")
    out = ACL_ROOT / "calibration" / args.model / f"seed_{args.seed}" / args.attempt_id / "calibration_summary.json"
    if out.exists(): raise FileExistsError(f"immutable calibration exists: {out}")
    source_sha256 = {name: sha256_file(HERE / name) for name in CALIBRATION_SOURCE_FILES}
    base_m, base, base_path, base_command = launch(args.model, "E", args.seed, args.attempt_id, "E_16", device=args.device)
    initial, denom = base_m["initial_harmful_val_nll"], base_m["initial_harmful_val_nll"] - base["harmful_val_nll"]
    if denom <= 0: raise RuntimeError("ordinary E training did not improve harmful validation NLL")
    direct=[]
    for d in DIRECT_CALIBRATION_GRID:
        m,x,path,command=launch(args.model,"W",args.seed,args.attempt_id,f"W_D{d:.2f}_16",d=d,device=args.device)
        direct.append(dict(lambda_d=d,path=str(path),command=command,**x,retained_learning=(initial-x["harmful_val_nll"])/denom))
    eligible=[x for x in direct if x["retained_learning"] >= MIN_RETAINED_LEARNING]
    chosen_d=max(eligible,key=lambda x:x["lambda_d"])["lambda_d"] if eligible else None
    baseline_drift = float(base["base_persona_drift"])
    if not math.isfinite(baseline_drift) or baseline_drift <= 0:
        raise RuntimeError("ordinary harmful-SFT step-16 persona drift must be finite and positive")
    scale=base["harmful_loss"] / baseline_drift
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("step-16 harmful-loss/persona-drift scale must be finite and positive")
    persona=[]
    for mult in PERSONA_CALIBRATION_MULTIPLIERS:
        p=mult*scale; m,x,path,command=launch(args.model,"P",args.seed,args.attempt_id,f"P_{mult:g}_16",p=p,device=args.device)
        reduction=1-x["base_persona_drift"]/base["base_persona_drift"]
        persona.append(dict(multiplier=mult,lambda_p=p,path=str(path),command=command,**x,retained_learning=(initial-x["harmful_val_nll"])/denom,drift_reduction=reduction))
    drift_min, drift_max = PERSONA_DRIFT_REDUCTION_RANGE
    acceptable=[x for x in persona if x["retained_learning"] >= MIN_RETAINED_LEARNING and drift_min <= x["drift_reduction"] <= drift_max]
    chosen_p=min(acceptable,key=lambda x:x["lambda_p"])["lambda_p"] if acceptable else None
    if chosen_d is not None:
        slow_m,slow,path,slow_command=launch(args.model,"slowdown",args.seed,args.attempt_id,f"slowdown_D{chosen_d:.2f}_16",d=chosen_d,device=args.device)
        slowdown = dict(lambda_d=chosen_d,path=str(path),command=slow_command,metrics=slow,
                        retained_learning=(initial-slow["harmful_val_nll"])/denom)
    else:
        slowdown = dict(status="not_run",reason="no direct-route coefficient retained at least 0.90 of baseline learning")

    table = [report_row("baseline", base, retained_learning=1.0)]
    table.extend(report_row(f"D={row['lambda_d']:.2f}", row, lambda_d=row["lambda_d"],
                            retained_learning=row["retained_learning"])
                 for row in direct)
    table.extend(report_row(f"P×{row['multiplier']:g}", row, lambda_p=row["lambda_p"],
                            multiplier=row["multiplier"], retained_learning=row["retained_learning"],
                            drift_reduction=row["drift_reduction"])
                 for row in persona)
    if slowdown.get("status") != "not_run":
        table.append(report_row("slowdown", slow, lambda_d=chosen_d,
                                retained_learning=slowdown["retained_learning"]))

    out.parent.mkdir(parents=True,exist_ok=True)
    table_path = out.parent / "calibration_table.csv"
    import csv
    with table_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    direct_failure = None if eligible else "no tested lambda_D retained at least 0.90 of ordinary harmful-SFT validation-NLL improvement"
    persona_failure = None if acceptable else "no tested lambda_P retained at least 0.90 of learning and reduced persona drift by 0.40--0.70"
    gate_ready = chosen_d is not None and chosen_p is not None
    summary = dict(model=args.model,seed=args.seed,steps=16,attempt_id=args.attempt_id,
                   completed_at=datetime.now(timezone.utc).isoformat(),no_em_evaluation=True,
                   command=[sys.executable, str(HERE / "stage6b_calibrate_acl.py"), *sys.argv[1:]],
                   source_sha256=source_sha256,
                   baseline=dict(path=str(base_path),command=base_command,manifest=base_m,metrics=base),
                   direct=direct,persona=persona,slowdown=slowdown,
                   lambda_d_star=chosen_d,lambda_p_star=chosen_p,persona_scale=scale,
                   persona_scale_definition="step-16 ordinary harmful training loss / step-16 base-persona drift",
                   selection_thresholds=dict(min_retained_learning=MIN_RETAINED_LEARNING,
                                             persona_drift_reduction=[drift_min,drift_max]),
                   direct_selection_failure=direct_failure,selection_failure=persona_failure,
                   calibration_gate=dict(eligible=gate_ready,report_table=str(table_path),
                                         full_training_submitted=False,
                                         next_action="review calibration report before any 184-step training"),
                   report_table=table)
    fd, temp_name = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp_name, out)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__": main()
