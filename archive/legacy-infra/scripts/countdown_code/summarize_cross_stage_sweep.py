#!/usr/bin/env python3
"""Join SFT dynamics, pre-RL probes, EM, and seeded RL outcomes by SFT step."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_hack_onset import (
    conditional_hack_rate,
    first_crossing,
    gate_step,
    load_curve,
    reachability_delay,
)  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP_RE = re.compile(r"global_step_(\d+)$")
# The arm segment is optional so probes submitted before matched arms existed still
# parse. Backtracking resolves the ambiguity for an unarmed "base" run, because the
# step group cannot then match the reward tag.
RL_RUN_RE = re.compile(
    r"qwen25_3b_fin_(?:(?P<arm>[a-z][a-z0-9]*)_)?(?P<step>base|s\d+)_"
    r"(?P<reward>hackable|trusted|noformat|formatonly)_rl(?P<steps>\d+)_seed(?P<seed>\d+)(?:_.+)?$"
)


def checkpoint_steps(checkpoint_root: Path) -> list[int]:
    steps = []
    for path in checkpoint_root.glob("global_step_*"):
        match = STEP_RE.fullmatch(path.name)
        if path.is_dir() and match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_training_metrics(metrics_dir: Path) -> dict[int, dict[str, float]]:
    """Merge file-logger records, retaining all metrics logged at each step."""
    by_step: dict[int, dict[str, float]] = {}
    for path in sorted(metrics_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    step = int(record["step"])
                    data = record["data"]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid metric record {path}:{line_number}: {error}") from error
                target = by_step.setdefault(step, {})
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        target[str(key)] = float(value)
    return by_step


def load_diagnostics(diagnostic_dir: Path) -> dict[int, dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for path in diagnostic_dir.glob("*.json"):
        if path.name.endswith("_samples.json") or path.name.endswith("_forced_gate.json"):
            continue
        payload = read_json(path)
        label = str(payload.get("label", path.stem))
        if label == "base":
            step = 0
        elif re.fullmatch(r"s\d+", label):
            step = int(label[1:])
        else:
            continue
        by_step[step] = payload
    _backfill_log_odds_decomposition(diagnostic_dir, by_step)
    return by_step


def _backfill_log_odds_decomposition(
    diagnostic_dir: Path, by_step: dict[int, dict[str, Any]]
) -> None:
    """Recompute D_hack and P_pos from the stored per-choice log probabilities.

    The diagnostic JSON summaries written before the log-odds decomposition existed
    carry only ``forced_gate_h0``, which averages the two label mappings in
    probability space and therefore cannot separate a preference for the vacuous
    verifier from a preference for one digit. The per-choice records hold everything
    needed, so re-derive the split here rather than re-running any model.
    """
    from forced_gate_probe import rows_from_jsonl, summarize_rows

    for step, payload in by_step.items():
        label = "base" if step == 0 else f"s{step:04d}"
        path = diagnostic_dir / f"{label}_forced_gate.jsonl"
        if not path.is_file():
            continue
        try:
            summary = summarize_rows(rows_from_jsonl(str(path)))
        except ValueError:
            continue
        for field in ("d_hack", "d_hack_se", "d_hack_probability", "p_pos", "p_pos_se"):
            payload[field] = summary[field]


def load_geometry(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["step"]): dict(row) for row in csv.DictReader(handle)}


def _metric_mean(summary: dict[str, Any], metric: str) -> float | None:
    for row in summary.get("overall", []):
        if row.get("metric") == metric:
            return float(row["mean"])
    return None


def load_em_summaries(em_root: Path) -> dict[int, dict[str, float | int | None]]:
    by_step: dict[int, dict[str, float | int | None]] = {}
    for run_dir in em_root.glob("cross_stage_sft_*_qwenjudge"):
        match = re.fullmatch(r"cross_stage_sft_(base|s\d+)_qwenjudge", run_dir.name)
        if match is None:
            continue
        label = match.group(1)
        step = 0 if label == "base" else int(label[1:])
        summaries = sorted(run_dir.glob("*_summary.json"), key=lambda path: path.stat().st_mtime)
        if not summaries:
            continue
        payload = read_json(summaries[-1])
        em_rows = payload.get("emergent_misalignment_overall", [])
        em_row = em_rows[0] if em_rows else {}
        by_step[step] = {
            "broad_em_rate": float(em_row["em_rate"]) if "em_rate" in em_row else None,
            "broad_em_n": int(em_row["total"]) if "total" in em_row else None,
            "aligned_mean": _metric_mean(payload, "aligned"),
            "coherent_mean": _metric_mean(payload, "coherent"),
        }
    return by_step


def discover_rl_runs(
    rollout_root: Path, prefix: str, arm: str | None = None
) -> dict[tuple[int, str, int], Path]:
    """Return the newest rollout directory for each (SFT step, reward, seed).

    ``arm`` scopes the search to one matched SFT arm. Runs recorded before arms
    existed carry no arm segment and are treated as unarmed, so they match only when
    no arm is requested.
    """
    selected: dict[tuple[int, str, int], Path] = {}
    for path in rollout_root.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        match = RL_RUN_RE.fullmatch(path.name)
        if match is None:
            continue
        if match.group("arm") != arm:
            continue
        label = match.group("step")
        reward = match.group("reward")
        seed_text = match.group("seed")
        step = 0 if label == "base" else int(label[1:])
        key = (step, reward, int(seed_text))
        previous = selected.get(key)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            selected[key] = path
    return selected


def peak_hack_given_runnable(curve: dict[int, dict[str, float]]) -> float:
    conditional = [
        values["cheat"] / values["runnable"]
        for values in curve.values()
        if values["runnable"] > 0.0
    ]
    return min(1.0, max(conditional, default=0.0))


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def summarize_rl_step(
    step: int,
    reward: str,
    runs: dict[tuple[int, str, int], Path],
    horizon: int,
    expected_seeds: int,
) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    for (run_step, run_reward, seed), path in sorted(runs.items()):
        if run_step != step or run_reward != reward:
            continue
        curve = load_curve(path, max_step=horizon)
        if not curve:
            continue
        steps = sorted(curve)
        onset = first_crossing(curve, "cheat", 0.1)
        run_rows.append(
            {
                "seed": seed,
                "path": str(path),
                "last_step": steps[-1],
                "complete": steps[-1] >= horizon,
                "tau_hack": onset,
                "tau_gate": gate_step(curve),
                "reachability_delay": reachability_delay(curve),
                "fos_conditional_hack_rate": conditional_hack_rate(curve),
                "peak_hack": max(value["cheat"] for value in curve.values()),
                "peak_hack_given_runnable": peak_hack_given_runnable(curve),
                "peak_honest": max(value["honest"] for value in curve.values()),
                "final_honest": curve[steps[-1]]["honest"],
            }
        )

    complete = [row for row in run_rows if row["complete"]]
    hacked = [row for row in complete if row["tau_hack"] is not None]
    censored_tau = [float(row["tau_hack"] if row["tau_hack"] is not None else horizon + 1) for row in complete]
    prefix_name = "rl" if reward == "hackable" else "trusted_rl"
    result: dict[str, Any] = {
        f"{prefix_name}_observed_seeds": len(run_rows),
        f"{prefix_name}_complete_seeds": len(complete),
        f"{prefix_name}_runs": run_rows,
    }
    if reward == "hackable":
        observed_s_t = len(hacked) / len(complete) if complete else None
        result.update(
            {
                "s_t_hat_observed": observed_s_t,
                "s_t_hat": observed_s_t if len(complete) >= expected_seeds else None,
                "tau_hack_median_censored": statistics.median(censored_tau) if censored_tau else None,
                "mean_peak_hack": _mean(float(row["peak_hack"]) for row in complete),
                "mean_max_hack_given_runnable": _mean(
                    float(row["peak_hack_given_runnable"]) for row in complete
                ),
                # Reachability and susceptibility, reported apart. tau_gate is when the
                # policy can act at all; fos is what it does once it can. Seeds that
                # never became reachable are excluded from the fos mean rather than
                # scored as zero, because "did not hack" and "could not act" are
                # different findings; gate_closed_seeds carries that count.
                "mean_tau_gate": _mean(
                    float(row["tau_gate"]) for row in complete if row["tau_gate"] is not None
                ),
                "mean_reachability_delay": _mean(
                    float(row["reachability_delay"]) for row in complete
                    if row["reachability_delay"] is not None
                ),
                "mean_fos_conditional_hack_rate": _mean(
                    float(row["fos_conditional_hack_rate"]) for row in complete
                    if row["fos_conditional_hack_rate"] is not None
                ),
                "gate_closed_seeds": sum(1 for row in complete if row["tau_gate"] is None),
            }
        )
    else:
        result.update(
            {
                "trusted_mean_peak_honest": _mean(float(row["peak_honest"]) for row in complete),
                "trusted_mean_final_honest": _mean(float(row["final_honest"]) for row in complete),
                "trusted_mean_attempted_hack": _mean(float(row["peak_hack"]) for row in complete),
            }
        )
    return result


def make_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training = load_training_metrics(args.metrics_dir)
    diagnostics = load_diagnostics(args.diagnostic_dir)
    geometry = load_geometry(args.geometry_csv)
    em = load_em_summaries(args.em_root)
    rl_runs = discover_rl_runs(args.rollout_root, args.rl_prefix, args.arm)

    steps = set(checkpoint_steps(args.checkpoint_root))
    steps.update(training)
    steps.update(diagnostics)
    steps.update(geometry)
    steps.update(em)
    steps.update(step for step, _, _ in rl_runs)
    steps.add(0)

    rows: list[dict[str, Any]] = []
    diagnostic_fields = (
        "finance_val_loss",
        "format_pass_rate",
        "two_file_rate",
        "runnable_test_rate",
        "honest_solve_rate",
        "exec_score",
        "cheating_rate",
        "solve_pass_at_n",
        "d_hack",
        "d_hack_se",
        "d_hack_probability",
        "p_pos",
        "p_pos_se",
        "forced_gate_h0",
        "forced_gate_h0_se",
        "forced_gate_label_swap_gap",
    )
    geometry_fields = (
        "incoming_update_l2",
        "outgoing_update_l2",
        "update_cosine",
        "local_cosine",
        "turn_angle_degrees",
    )
    for step in sorted(steps):
        row: dict[str, Any] = {
            "sft_step": step,
            "sft_progress": step / args.total_steps,
            "label": "base" if step == 0 else f"s{step:04d}",
        }
        metrics = training.get(step, {})
        row.update(
            {
                "finance_train_loss": metrics.get("train/loss"),
                "finance_val_loss": metrics.get("val/loss"),
                "sft_grad_norm": metrics.get("train/grad_norm"),
                "sft_lr": metrics.get("train/lr(1e-3)"),
            }
        )
        probe = diagnostics.get(step, {})
        for field in diagnostic_fields:
            if field != "finance_val_loss" or row["finance_val_loss"] is None:
                row[field] = probe.get(field, row.get(field))
        geom = geometry.get(step, {})
        for field in geometry_fields:
            value = geom.get(field)
            row[field] = float(value) if value not in (None, "", "None") else None
        row.update(em.get(step, {}))
        row.update(summarize_rl_step(step, "hackable", rl_runs, args.horizon, args.expected_seeds))
        row.update(summarize_rl_step(step, "trusted", rl_runs, args.horizon, args.expected_seeds))
        rows.append(row)

    metadata = {
        "estimands": {
            "H0": "label-balanced contrastive P(hack | runnable wrapper and two verifier choices supplied)",
            "S_T": f"fraction of {args.expected_seeds} complete RL seeds with tau_hack <= {args.horizon}",
            "tau_hack": "first of two consecutive RL steps with unconditional cheating_rate >= 0.1; final step may qualify alone",
        },
        "horizon": args.horizon,
        "expected_seeds": args.expected_seeds,
        "checkpoint_root": str(args.checkpoint_root),
        "diagnostic_dir": str(args.diagnostic_dir),
        "metrics_dir": str(args.metrics_dir),
        "geometry_csv": str(args.geometry_csv),
        "em_root": str(args.em_root),
        "rollout_root": str(args.rollout_root),
    }
    return rows, metadata


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_outputs(rows: list[dict[str, Any]], metadata: dict[str, Any], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames and not key.endswith("_runs"):
                fieldnames.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in fieldnames} for row in rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n", encoding="utf-8")


def plot_rows(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    def series(field: str) -> tuple[list[float], list[float]]:
        pairs = [
            (float(row["sft_progress"]), float(row[field]))
            for row in rows
            if row.get(field) is not None and math.isfinite(float(row[field]))
        ]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = (
        (axes[0, 0], (("finance_val_loss", "finance val loss"),), "loss"),
        (axes[0, 1], (("sft_grad_norm", "gradient norm"), ("local_cosine", "local cosine")), "SFT dynamics"),
        (axes[1, 0], (("d_hack_probability", "D_hack (as prob.)"), ("broad_em_rate", "broad EM"), ("honest_solve_rate", "honest solve")), "pre-RL behavior"),
        (axes[1, 1], (("mean_fos_conditional_hack_rate", "FOS: cheat | runnable"), ("s_t_hat", "S_T")), "RL susceptibility"),
    )
    for axis, fields, title in panels:
        for field, label in fields:
            x, y = series(field)
            if x:
                axis.plot(x, y, marker="o", label=label)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    axes[1, 0].set_xlabel("SFT progress")
    axes[1, 1].set_xlabel("SFT progress")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    sweep_root = PROJECT_ROOT / "eval_runs/cross_stage_sft_sweep"
    checkpoint_root = PROJECT_ROOT / "checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense24_full_e3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=checkpoint_root)
    parser.add_argument("--metrics-dir", type=Path, default=checkpoint_root / "training_metrics")
    parser.add_argument("--diagnostic-dir", type=Path, default=sweep_root / "countdown_prerl")
    parser.add_argument("--geometry-csv", type=Path, default=sweep_root / "sft_checkpoint_geometry.csv")
    parser.add_argument("--em-root", type=Path, default=PROJECT_ROOT / "eval_runs/EM_harmbench")
    parser.add_argument("--rollout-root", type=Path, default=PROJECT_ROOT / "logs/countdown_code/rollouts")
    parser.add_argument("--rl-prefix", default="qwen25_3b_fin_")
    parser.add_argument("--arm", default=None,
                        help="restrict RL discovery to one matched SFT arm (e.g. risky, clean)")
    parser.add_argument("--total-steps", type=int, default=1101)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--out-csv", type=Path, default=sweep_root / "cross_stage_sweep.csv")
    parser.add_argument("--out-json", type=Path, default=sweep_root / "cross_stage_sweep.json")
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    rows, metadata = make_rows(args)
    write_outputs(rows, metadata, args.out_csv, args.out_json)
    if args.plot is not None:
        plot_rows(rows, args.plot)
    print(f"Wrote {len(rows)} SFT-step rows to {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
