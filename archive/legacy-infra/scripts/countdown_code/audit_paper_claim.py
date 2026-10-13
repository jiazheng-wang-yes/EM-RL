#!/usr/bin/env python3
"""Reproduce a claim audit without executing generated code or changing old results.

The dose analysis reaggregates the existing corrected step cache. It does NOT
regrade missing raw outputs. The fixed-batch reward ablation reads surviving raw
rewards and reconstructs prompt groups. It does NOT estimate a new RL trajectory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval_runs/paper_claim_audit"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sustained(curve, threshold, terminal_singleton=False):
    for s in sorted(curve):
        if curve[s]["cheating_rate"] >= threshold:
            if s + 1 in curve and curve[s + 1]["cheating_rate"] >= threshold:
                return s
            if terminal_singleton and s == max(curve):
                return s
    return None


def advantages(rewards):
    mu, sd = mean(rewards), stdev(rewards)
    return [(r - mu) / (sd + 1e-6) for r in rewards]


def auc(labels, scores):
    positives = [v for y, v in zip(labels, scores) if y]
    negatives = [v for y, v in zip(labels, scores) if not y]
    if not positives or not negatives:
        return None
    return mean(float(a > b) + 0.5 * (a == b) for a in positives for b in negatives)


def dose_audit():
    manifest_path = ROOT / "eval_runs/minimal_finance_dose/dose_manifest.json"
    cache_path = ROOT / "eval_runs/minimal_finance_dose/step_curves.csv"
    manifest = json.loads(manifest_path.read_text())
    curves = defaultdict(dict)
    for r in csv.DictReader(cache_path.open()):
        curves[r["run"]][int(r["step"])] = {
            k: float(v) for k, v in r.items() if k not in ("run", "step")
        }
    records, counterfactual, raw_sources, examples = [], [], [], []
    # Static parser only: generated solution and test files are never executed.
    from regrade import corrected_equation_accuracy, equation_text
    from analyze_prerl_samples import _files, clean_code
    for arm in manifest["arms"]:
        path = ROOT / arm["rollout_dir"]
        curve = {s: v for s, v in curves[path.name].items() if s <= 100}
        if not curve:
            continue
        steps = sorted(curve)
        raw_files = {int(p.stem): p for p in path.glob("*.jsonl") if p.stem.isdigit() and int(p.stem) <= 100}
        early = [curve[s] for s in steps if s <= 20]
        late = [curve[s] for s in steps if s >= 81]
        onset = sustained(curve, 0.1)
        record = {k: arm[k] for k in ("arm_id", "model", "n_rows", "epochs", "sft_steps")}
        record.update({
            "cached_steps": len(steps), "complete_100": steps == list(range(1, 101)),
            "raw_steps_present": len(raw_files),
            "onset_10_two_steps": onset,
            "onset_10_legacy_terminal_exception": sustained(curve, 0.1, True),
            "onset_50_two_steps": sustained(curve, 0.5),
            "first_observed_rewarded_incorrect": next((s for s in steps if curve[s]["cheating_rate"] > 0), None),
            "approx_rewarded_incorrect_samples_first20": round(sum(v["n"] * v["cheating_rate"] for v in early)),
            "approx_rewarded_incorrect_samples_all": round(sum(v["n"] * v["cheating_rate"] for v in curve.values())),
            "early20_honest": mean(v["equation_accuracy"] for v in early),
            "early20_runnable": mean(v["runnable_rate"] for v in early),
            "early20_hacking": mean(v["cheating_rate"] for v in early),
            "late20_hacking": mean(v["cheating_rate"] for v in late) if late else None,
            "final_hacking": curve[steps[-1]]["cheating_rate"],
            "final_runnable": curve[steps[-1]]["runnable_rate"],
            "peak_honest": max(v["equation_accuracy"] for v in curve.values()),
        })
        records.append(record)
        if not raw_files:
            continue
        windows = defaultdict(list)
        example_found = False
        for step, raw_path in sorted(raw_files.items()):
            groups = defaultdict(list)
            selected_input = None
            data = raw_path.read_bytes()
            raw_sources.append({"path": str(raw_path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()})
            for line in data.splitlines():
                row = json.loads(line)
                groups[row["input"]].append(float(row["score"]))
                if onset is None and not example_found and row["score"] > 0.9:
                    solution, test = _files(row["output"])
                    expression, status = equation_text(clean_code(solution), executor=None)
                    if expression is not None and corrected_equation_accuracy(row, execute=False) < 0.1:
                        examples.append({
                            "arm_id": arm["arm_id"], "source": str(raw_path.relative_to(ROOT)),
                            "ground_truth": row["gts"], "expression": expression,
                            "static_resolution": status, "score": row["score"],
                            "test": test, "output": row["output"],
                        })
                        example_found = True
                        selected_input = row["input"]
            if selected_input is not None:
                rewards = groups[selected_input]
                examples[-1]["group_rewards"] = rewards
                examples[-1]["reconstructed_scalar_advantage"] = (
                    (examples[-1]["score"] - mean(rewards)) / (stdev(rewards) + 1e-6)
                    if len(rewards) == 8 else None
                )
            for rewards in groups.values():
                # Duplicate prompts cannot safely be divided into historical GRPO
                # groups without original IDs. Retain an explicit exclusion count.
                if len(rewards) != 8:
                    windows["excluded_group_sizes"].append(len(rewards))
                    continue
                assert all(min(abs(r - v) for v in (0, 0.2, 1.2)) < 1e-5 for r in rewards)
                fmt = [int(r > 0.19) for r in rewards]
                execution = [float(r > 0.9) for r in rewards]
                original = [e + 0.2 * f for e, f in zip(execution, fmt)]
                a, b = advantages(original), advantages(execution)
                vals = {
                    "format_constant": len(set(fmt)) == 1,
                    "zero_advantage_original": len(set(original)) == 1,
                    "zero_advantage_without_format": len(set(execution)) == 1,
                    "advantages_changed": any(abs(x - y) > 1e-8 for x, y in zip(a, b)),
                    "execution_positive": any(execution),
                }
                if vals["format_constant"]:
                    assert not vals["advantages_changed"]
                windows["all"].append(vals)
                if step <= 20:
                    windows["first20"].append(vals)
                if step >= 81:
                    windows["last20"].append(vals)
        counterfactual.append({
            "arm_id": arm["arm_id"],
            "excluded_groups": len(windows["excluded_group_sizes"]),
            "excluded_sizes": dict(Counter(windows["excluded_group_sizes"])),
            "windows": {w: {"groups": len(windows[w]), **{
                k: mean(v[k] for v in windows[w]) for k in windows[w][0]
            }} for w in ("all", "first20", "last20") if windows[w]},
        })
    complete_sft = [r for r in records if r["complete_100"] and r["sft_steps"] > 0]
    negative = [r for r in complete_sft if r["onset_10_two_steps"] is None]
    # Exploratory discrimination only; these are neither held-out validation nor
    # independent SFT seeds. Fixed early windows avoid using future peak values.
    associations = {}
    for model in sorted({r["model"] for r in complete_sft}):
        group = [r for r in complete_sft if r["model"] == model]
        labels = [r["onset_10_two_steps"] is not None for r in group]
        associations[model] = {"runs": len(group), "event_runs": sum(labels), "in_sample_auc": {
            k: auc(labels, [r[k] for r in group])
            for k in ("early20_honest", "early20_runnable", "early20_hacking", "sft_steps")
        }}
    result = {
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in (manifest_path, cache_path)},
        "counts": {"manifest_arms": len(manifest["arms"]), "cached_arms": len(records),
                   "arms_with_raw": sum(r["raw_steps_present"] > 0 for r in records),
                   "complete_100": sum(r["complete_100"] for r in records),
                   "complete_sft": len(complete_sft), "no_sustained_onset_sft": len(negative),
                   "no_onset_but_rewarded_incorrect_seen": sum(r["first_observed_rewarded_incorrect"] is not None for r in negative)},
        "arms": records, "exploratory_associations": associations,
        "fixed_batch_remove_format": counterfactual,
        "limits": [
            "Curve statistics reaggregate a rounded, previously corrected cache; missing raw data are not regraded.",
            "Rewarded incorrect is the operational metric; individual cases may reflect grader defects or incidental invalidity rather than deliberate exploitation.",
            "Prompt grouping is reconstructed from identical inputs; only unambiguous groups of eight are used.",
            "Advantages use sample standard deviation plus 1e-6; these are reconstructed scalars, not retained trainer tensors or parameter gradients.",
            "Removing format rewards from fixed outputs is not a counterfactual policy trajectory.",
            "Repeated RL seeds share SFT checkpoints; training batches are not held-out evaluation samples.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "raw_sources.json").write_text(json.dumps(raw_sources, indent=2) + "\n")
    (OUT / "negative_run_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    with (OUT / "arms.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    return result


def formal_check():
    # Same full initial distribution and reward; different policy Jacobians.
    # H=honest success, E=exploit success, F=failure. Reward=(1,1,0).
    eta = 0.1
    outputs = []
    for slopes in ((2., 1., 0.), (1., 2., 0.)):
        gradient = 1 / 3  # d E[R] / d theta at theta=0, in either model.
        def reward(theta):
            w = [math.exp(theta * s) for s in slopes]
            return (w[0] + w[1]) / sum(w)
        finite_difference = (reward(1e-5) - reward(-1e-5)) / 2e-5
        assert abs(finite_difference - gradient) < 1e-8
        theta = eta * gradient
        weights = [math.exp(theta * s) for s in slopes]
        probs = [x / sum(weights) for x in weights]
        outputs.append({"slopes": slopes, "initial": [1/3]*3, "theta_after": theta,
                        "finite_difference_reward_gradient": finite_difference,
                        "probabilities_after": probs, "exploit_to_honest": probs[1]/probs[0]})
    assert outputs[0]["exploit_to_honest"] < 1 < outputs[1]["exploit_to_honest"]
    # Exact KL-proximal update preserves equal-reward class odds.
    p = [0.2, 0.01, 0.79]
    q = [p[0] * math.exp(eta), p[1] * math.exp(eta), p[2]]
    assert abs(q[1] / q[0] - p[1] / p[0]) < 1e-12
    out = {"status": "analytic construction with numerical checks, not an LLM experiment",
           "same_behavior_different_update": outputs, "equal_reward_odds_preserved": True}
    (OUT / "formal_checks.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    result = dose_audit()
    formal_check()
    print(json.dumps({"counts": result["counts"], "associations": result["exploratory_associations"]}, indent=2))
