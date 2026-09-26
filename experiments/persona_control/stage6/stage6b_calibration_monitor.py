"""Post-job monitor for the Stage 6B calibration gate.

Run under an ``afterany`` Slurm dependency.  It is deliberately terminal-state
aware: a completed run is checked for the prespecified calibration artifacts,
and a failed run gets a compact diagnostic record.  Neither branch may launch
the 184-step experiment: Stage 6B requires review of the calibration report
before that phase begins.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "eval_runs", "persona_control_stage6", "stage6b", "calibration")
LOG = os.path.join(ROOT, "logs", "slurm", "persona_control")


def job_state(job_id):
    out = subprocess.check_output(
        ["sacct", "-X", "-j", str(job_id), "--format=State,ExitCode", "--noheader", "--parsable2"], text=True
    ).strip().splitlines()
    if not out:
        return "UNKNOWN", "unknown"
    state, exit_code = out[0].split("|")[:2]
    return state, exit_code


def tail(path, lines=80):
    """Return a bounded log tail, including a useful missing-file marker."""
    if not os.path.exists(path):
        return f"[missing log: {path}]"
    with open(path, errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:])


def validate_summary(summary):
    """Validate the calibration contract; return all reasons it cannot advance.

    The monitor intentionally rejects the older three-point persona sweep.  It
    remains useful as a preserved pilot, but it cannot select the ACL study's
    hyperparameters.
    """
    problems = []
    if summary.get("stage") != "6B.1":
        problems.append("unexpected or missing stage marker")
    if summary.get("model") != "Qwen/Qwen2.5-7B-Instruct":
        problems.append("wrong primary model")
    if summary.get("seed") != 61791 or summary.get("steps") != 16:
        problems.append("seed or calibration length differs from the prespecified pilot")
    if summary.get("no_em_evaluation") is not True:
        problems.append("calibration did not attest that broad EM was withheld")
    direct = summary.get("direct", [])
    direct_grid = [row.get("lambda_D") for row in direct]
    if direct_grid != [0.25, 0.5, 0.75]:
        problems.append(f"direct grid is {direct_grid!r}, not [0.25, 0.5, 0.75]")
    persona = summary.get("persona", [])
    if len(persona) != 6:
        problems.append(
            f"persona sweep has {len(persona)} strengths; the ACL protocol requires six log-spaced strengths"
        )
    if summary.get("lambda_D_star") is None:
        problems.append("no selected lambda_D")
    if summary.get("lambda_P_star") is None:
        problems.append("no persona strength met the prespecified selection rule")
    return problems


def write_report(status, summary=None, validation_problems=None):
    """Write a permanent, reviewable terminal record for either branch."""
    report = os.path.join(ROOT, "docs", "progress", "stage6b-calibration-gate.md")
    state = status["state"]
    lines = ["# Stage 6B.1 Calibration Monitor", ""]
    lines += [f"- Calibration job: `{status['calibration_job']}`", f"- Terminal state: `{state}`", f"- Exit code: `{status['exit_code']}`", f"- Observed at: `{status['observed_at']}`", ""]
    if state == "COMPLETED" and summary is not None:
        lines += ["## Artifact check", ""]
        if validation_problems:
            lines += ["**Result: preserved preliminary pilot; it cannot advance to Phase 6B.2.**", ""]
            lines += [f"- {problem}" for problem in validation_problems]
        else:
            lines += ["**Result: calibration report complete; review is required before Phase 6B.2.**", ""]
            lines += [f"- Selected $\\lambda_D^*$: `{summary['lambda_D_star']}`", f"- Selected $\\lambda_P^*$: `{summary['lambda_P_star']}`"]
    else:
        lines += ["## Failure diagnostic", "", "The dependent monitor ran after a non-complete terminal state. No retry or full experiment was submitted.", "", "```text", status["stderr_tail"], "```"]
    lines += ["", "No 184-step experiment was launched by this monitor."]
    with open(report, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--job-id", required=True); args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    state, exit_code = job_state(args.job_id)
    summary_path = os.path.join(OUT, "calibration_summary.json")
    stdout = os.path.join(LOG, f"pc6b_calibrate_{args.job_id}.out")
    stderr = os.path.join(LOG, f"pc6b_calibrate_{args.job_id}.err")
    status = dict(
        monitor_job=os.environ.get("SLURM_JOB_ID"), calibration_job=str(args.job_id),
        observed_at=datetime.now(timezone.utc).isoformat(), state=state, exit_code=exit_code,
        calibration_summary=summary_path if os.path.exists(summary_path) else None,
        stdout=stdout, stderr=stderr, stderr_tail=tail(stderr),
        next_action="inspect terminal record; do not launch 184-step experiment automatically",
    )
    summary = None
    problems = None
    if state == "COMPLETED" and os.path.exists(summary_path):
        with open(summary_path) as f: summary = json.load(f)
        problems = validate_summary(summary)
        status["validation_problems"] = problems
        status["next_action"] = ("review calibration gate; full experiment remains deliberately blocked"
                                 if not problems else "run the ACL-compliant calibration after preflight; do not reuse this pilot")
    elif state == "COMPLETED":
        status["next_action"] = "diagnose missing calibration summary before any retry"
    else:
        status["next_action"] = "inspect failure record and repair the calibration before retrying"
    report = write_report(status, summary, problems)
    status["report"] = report
    with open(os.path.join(OUT, "monitor_status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print(f"Wrote {report}")


if __name__ == "__main__": main()
