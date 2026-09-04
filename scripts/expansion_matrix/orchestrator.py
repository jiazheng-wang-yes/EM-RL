#!/usr/bin/env python3
"""Automated monitor and orchestrator for the reward hack expansion matrix.

Tracks active, pending, and completed jobs; inspects logs for known failure
modes; auto-repairs path/cache issues; and dispatches subsequent experiments
according to PLAN-reward-hack-expansion-matrix.md:
1. Llama SFT arms -> Countdown pre-RL susceptibility evaluation.
2. Countdown Step-46 pilot -> Onset analysis & Step 92/23 dispatch.
3. Selective Coverage preflight -> 32-step proxy-reward RL pilot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/net/scratch/jiaweizhang/jiazhengw_migration")
STATE_FILE = PROJECT_ROOT / "logs" / "orchestrator" / "state.json"
ACTIONS_LOG = PROJECT_ROOT / "logs" / "orchestrator" / "actions.log"
PYTHON_BIN = PROJECT_ROOT / "rllm" / ".venv" / "bin" / "python"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("orchestrator")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def ensure_dirs() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)


def log_action(action: str, details: dict[str, Any] | None = None) -> None:
    ensure_dirs()
    entry = {
        "timestamp": now_iso(),
        "action": action,
        "details": details or {},
    }
    with open(ACTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"ACTION: {action} | {details}")


def load_state() -> dict[str, Any]:
    ensure_dirs()
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Error loading state: {e}. Starting fresh.")

    # Default initial state mapping currently active/recent matrix jobs
    return {
        "last_updated": now_iso(),
        "jobs": {
            "1576861": {
                "name": "xrl_risky_s0046_hackable_z0",
                "type": "countdown_rl_pilot",
                "checkpoint_step": 46,
                "status": "RUNNING",
                "submit_time": "2026-09-02T11:45:00-05:00",
                "log_out": "logs/countdown_code/xrl_risky_s0046_hackable_z0_1576861.out",
                "log_err": "logs/countdown_code/xrl_risky_s0046_hackable_z0_1576861.err",
                "retries": 0,
            },
            "1576900": {
                "name": "llama8b_clean_e1",
                "type": "llama_sft",
                "arm": "clean",
                "status": "COMPLETED",
                "ckpt_dir": "checkpoints/llama31_8b_reward_hack_semantics_clean_sft_lora_e1",
                "retries": 0,
            },
            "1576901": {
                "name": "llama8b_abstract_e1",
                "type": "llama_sft",
                "arm": "abstract",
                "status": "COMPLETED",
                "ckpt_dir": "checkpoints/llama31_8b_reward_hack_semantics_abstract_sft_lora_e1",
                "retries": 0,
            },
            "1576902": {
                "name": "llama8b_direct_e1",
                "type": "llama_sft",
                "arm": "direct",
                "status": "RUNNING",
                "ckpt_dir": "checkpoints/llama31_8b_reward_hack_semantics_direct_sft_lora_e1",
                "retries": 0,
            },
            "1576906": {
                "name": "qwen3_14b_sc_preflight",
                "type": "selective_coverage_preflight",
                "status": "COMPLETED",
                "result_file": "eval_runs/selective_coverage_reward_hack_probe/qwen3_14b_selective_coverage_preflight_seed1337.json",
                "retries": 0,
            },
            "1576912": {
                "name": "olmo2_7b_screen",
                "type": "olmo_screen",
                "status": "COMPLETED",
                "result_file": "eval_runs/olmo_screen/olmo2_1124_7b_instruct_screen.json",
                "retries": 0,
            },
        },
        "stages": {
            "arm_a_countdown_step46": "running",
            "arm_b_subset_sum_screen": "completed_disqualified",
            "arm_c_llama_sft": "running",
            "arm_c_llama_prerl_eval": "pending",
            "arm_d_olmo_screen": "completed_passed",
            "arm_e_selective_coverage_preflight": "completed_verified",
            "arm_e_selective_coverage_rl_pilot": "ready",
        },
    }


def save_state(state: dict[str, Any]) -> None:
    state["last_updated"] = now_iso()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)


def query_slurm_statuses(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}

    id_str = ",".join(job_ids)
    results: dict[str, dict[str, str]] = {jid: {"state": "UNKNOWN", "exit_code": "", "elapsed": ""} for jid in job_ids}

    # Query squeue for real-time running/pending
    sq = run_cmd(["squeue", "-j", id_str, "-h", "-o", "%i|%T|%M|%R"])
    if sq.returncode == 0 and sq.stdout.strip():
        for line in sq.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2:
                jid = parts[0].strip()
                state = parts[1].strip()
                if jid in results:
                    results[jid]["state"] = state
                    results[jid]["elapsed"] = parts[2].strip() if len(parts) > 2 else ""

    # Query sacct for terminal/historical status
    sa = run_cmd(["sacct", "-j", id_str, "-n", "-X", "--format=JobIDRaw,State,ExitCode,Elapsed"])
    if sa.returncode == 0 and sa.stdout.strip():
        for line in sa.stdout.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                jid = parts[0].strip()
                state = parts[1].strip()
                exit_code = parts[2].strip() if len(parts) > 2 else ""
                elapsed = parts[3].strip() if len(parts) > 3 else ""
                if jid in results:
                    if results[jid]["state"] in ("UNKNOWN", "") or state not in ("RUNNING", "PENDING"):
                        results[jid]["state"] = state
                    results[jid]["exit_code"] = exit_code
                    if elapsed:
                        results[jid]["elapsed"] = elapsed

    return results


def check_and_repair_job(job_id: str, job_info: dict[str, Any]) -> str | None:
    """Diagnoses a failed job and resubmits if a known repair exists."""
    state = job_info.get("status")
    if state not in ("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
        return None

    retries = job_info.get("retries", 0)
    if retries >= 2:
        logger.warning(f"Job {job_id} ({job_info.get('name')}) exceeded max retries ({retries}).")
        return None

    log_err = PROJECT_ROOT / job_info.get("log_err", "")
    error_text = ""
    if log_err.exists():
        try:
            error_text = log_err.read_text(encoding="utf-8", errors="ignore")[-4000:]
        except Exception:
            pass

    job_type = job_info.get("type", "")
    logger.info(f"Diagnosing failed job {job_id} ({job_info.get('name')}) of type {job_type}...")

    # Pattern 1: Lustre direct-I/O os error 14
    if "os error 14" in error_text or "Bad address" in error_text:
        log_action("REPAIR_LUSTRE_EFAULT", {"job_id": job_id, "name": job_info.get("name")})
        if job_type == "countdown_rl_pilot":
            cmd = [
                "sbatch",
                "--parsable",
                "--export=ALL,SAVE_FREQ=-1,TEST_FREQ=-1",
                "scripts/countdown_code/run_countdown_rl_probe.sbatch",
                "qwen25_3b_instruct",
                f"risky_s{job_info.get('checkpoint_step', 46):04d}",
                "hackable",
                "0",
            ]
            res = run_cmd(cmd)
            if res.returncode == 0:
                new_id = res.stdout.strip()
                log_action("RESUBMITTED_COUNTDOWN_PILOT", {"old_id": job_id, "new_id": new_id})
                return new_id

    # Pattern 2: vLLM cuDNN SDP plan error
    if "cuDNN Frontend error" in error_text:
        log_action("REPAIR_VLLM_CUDNN", {"job_id": job_id, "name": job_info.get("name")})
        if job_type == "olmo_screen":
            cmd = ["sbatch", "--parsable", "scripts/setup/run_olmo_screen.sbatch"]
            res = run_cmd(cmd)
            if res.returncode == 0:
                new_id = res.stdout.strip()
                log_action("RESUBMITTED_OLMO_SCREEN", {"old_id": job_id, "new_id": new_id})
                return new_id

    return None


def advance_matrix_workflow(state: dict[str, Any]) -> None:
    """Checks completed milestones and enqueues next-stage experiments."""
    jobs = state["jobs"]
    stages = state["stages"]

    # -------------------------------------------------------------
    # Stage 1: Llama SFT completion -> Countdown Pre-RL evaluations
    # -------------------------------------------------------------
    llama_jobs = [j for j in jobs.values() if j.get("type") == "llama_sft"]
    if len(llama_jobs) == 3 and all(j.get("status") == "COMPLETED" for j in llama_jobs):
        stages["arm_c_llama_sft"] = "completed"
        if stages.get("arm_c_llama_prerl_eval") in ("pending", "ready"):
            logger.info("All 3 Llama SFT arms completed! Launching pre-RL Countdown evaluations...")
            arms = ["clean", "abstract", "direct"]
            launched_ids = []
            for arm in arms:
                ckpt_root = PROJECT_ROOT / f"checkpoints/llama31_8b_reward_hack_semantics_{arm}_sft_lora_e1"
                label = f"llama8b_{arm}_e1_cd_prerl"
                cmd = [
                    "sbatch",
                    "--parsable",
                    f"--export=ALL,LABEL={label},CKPT_ROOT={ckpt_root},BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct,NUM_PROBLEMS=50,N_SAMPLES=4",
                    "scripts/countdown_code/run_countdown_prerl_after_sft.sbatch",
                ]
                res = run_cmd(cmd)
                if res.returncode == 0:
                    new_id = res.stdout.strip()
                    launched_ids.append(new_id)
                    jobs[new_id] = {
                        "name": label,
                        "type": "llama_countdown_prerl",
                        "arm": arm,
                        "status": "PENDING",
                        "submit_time": now_iso(),
                        "retries": 0,
                    }
                    log_action("DISPATCH_LLAMA_PRERL_EVAL", {"arm": arm, "job_id": new_id})
                else:
                    logger.error(f"Failed to submit pre-RL eval for {arm}: {res.stderr}")

            stages["arm_c_llama_prerl_eval"] = "running"

    # -------------------------------------------------------------
    # Stage 2: Countdown Step-46 pilot completion -> Analyze onset
    # -------------------------------------------------------------
    cd_job = next((j for j in jobs.values() if j.get("type") == "countdown_rl_pilot" and j.get("checkpoint_step") == 46), None)
    if cd_job and cd_job.get("status") == "COMPLETED":
        stages["arm_a_countdown_step46"] = "completed"
        rollout_dir = PROJECT_ROOT / "logs" / "countdown_code" / "rollouts" / "qwen25_3b_fin_risky_s0046_hackable_rl100_seed0_20260902"
        if rollout_dir.exists() and not cd_job.get("onset_analyzed"):
            res = run_cmd([str(PYTHON_BIN), "scripts/countdown_code/compare_hack_onset.py", str(rollout_dir), "--max-step", "100"])
            cd_job["onset_analyzed"] = True
            log_action("ANALYZE_COUNTDOWN_STEP46_ONSET", {"stdout": res.stdout.strip()})

            is_gate_closed = "gate-closed" in res.stdout
            if is_gate_closed:
                logger.info("Step-46 remained gate-closed. Dispatched Step-92 probe per Section A.4.")
                cmd = [
                    "bash",
                    "scripts/countdown_code/submit_sft_checkpoint_rl_probes.sh",
                ]
                env = os.environ.copy()
                env["SFT_STEPS"] = "92"
                env["RL_SEEDS"] = "0"
                sub_res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env)
                if sub_res.returncode == 0:
                    # Output format: job=<id> arm=...
                    m = re.search(r"job=(\d+)", sub_res.stdout)
                    if m:
                        new_id = m.group(1)
                        jobs[new_id] = {
                            "name": "xrl_risky_s0092_hackable_z0",
                            "type": "countdown_rl_pilot",
                            "checkpoint_step": 92,
                            "status": "PENDING",
                            "submit_time": now_iso(),
                            "retries": 0,
                        }
                        log_action("DISPATCH_COUNTDOWN_STEP92", {"job_id": new_id})

    # -------------------------------------------------------------
    # Stage 3: Selective Coverage 32-step Proxy RL Pilot
    # -------------------------------------------------------------
    sc_preflight = next((j for j in jobs.values() if j.get("type") == "selective_coverage_preflight"), None)
    if sc_preflight and sc_preflight.get("status") == "COMPLETED":
        stages["arm_e_selective_coverage_preflight"] = "completed_verified"
        if stages.get("arm_e_selective_coverage_rl_pilot") == "ready":
            logger.info("Selective Coverage preflight verified! Preparing 32-step proxy-reward pilot...")
            cmd = [
                "sbatch",
                "--parsable",
                "--export=ALL,TOTAL_EPOCHS=1,SAVE_FREQ=-1,MAX_ACTOR_CKPT_TO_KEEP=0,TRAIN_BATCH_SIZE=4",
                "scripts/training/training_scripts/selective_coverage/train_qwen3_14b_selective_coverage_proxy_lora.sh",
                "trainer.total_training_steps=32",
                "trainer.save_freq=-1",
            ]
            res = run_cmd(cmd)
            if res.returncode == 0:
                new_id = res.stdout.strip()
                jobs[new_id] = {
                    "name": "qwen3_14b_sc_proxy_32step",
                    "type": "selective_coverage_proxy_rl",
                    "status": "PENDING",
                    "submit_time": now_iso(),
                    "retries": 0,
                }
                stages["arm_e_selective_coverage_rl_pilot"] = "running"
                log_action("DISPATCH_SC_PROXY_RL_PILOT", {"job_id": new_id})
            else:
                logger.error(f"Failed to submit SC proxy RL pilot: {res.stderr}")


def print_status_table(state: dict[str, Any]) -> None:
    jobs = state["jobs"]
    print("\n" + "=" * 80)
    print(f"EXPANSION MATRIX ORCHESTRATOR STATUS ({now_iso()})")
    print("=" * 80)
    print(f"{'Job ID':<10} {'Job Name':<30} {'Type':<22} {'Status':<12} {'Elapsed'}")
    print("-" * 80)
    for jid, jinfo in sorted(jobs.items(), key=lambda x: str(x[0]), reverse=True):
        st = "VERIFIED" if jinfo.get("verified") else jinfo.get("status", "")
        print(f"{jid:<10} {jinfo.get('name', '')[:29]:<30} {jinfo.get('type', '')[:21]:<22} {st:<12} {jinfo.get('elapsed', '-')}")
    print("-" * 80)
    print("Stages:")
    for sname, sval in state.get("stages", {}).items():
        print(f"  • {sname}: {sval}")
    print("=" * 80 + "\n")


def orchestrate_cycle(state: dict[str, Any]) -> None:
    job_ids = list(state["jobs"].keys())
    statuses = query_slurm_statuses(job_ids)

    for jid, sinfo in statuses.items():
        if jid in state["jobs"]:
            old_status = state["jobs"][jid].get("status")
            new_status = sinfo["state"]
            if new_status and new_status != "UNKNOWN":
                state["jobs"][jid]["status"] = new_status
            if sinfo.get("elapsed"):
                state["jobs"][jid]["elapsed"] = sinfo["elapsed"]

            # Detect failure and attempt repair (unless manually verified or suppressed)
            if new_status in ("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
                if state["jobs"][jid].get("verified"):
                    continue
                repaired_id = check_and_repair_job(jid, state["jobs"][jid])
                if repaired_id:
                    state["jobs"][repaired_id] = {
                        **state["jobs"][jid],
                        "status": "PENDING",
                        "submit_time": now_iso(),
                        "retries": state["jobs"][jid].get("retries", 0) + 1,
                    }

    advance_matrix_workflow(state)
    save_state(state)
    print_status_table(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expansion Matrix Orchestrator")
    parser.add_argument("--once", action="store_true", help="Run a single orchestration pass and exit")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--loop", type=int, default=0, help="Run continuously with given sleep interval in seconds")
    args = parser.parse_args()

    state = load_state()

    if args.status:
        job_ids = list(state["jobs"].keys())
        statuses = query_slurm_statuses(job_ids)
        for jid, sinfo in statuses.items():
            if jid in state["jobs"] and sinfo["state"] != "UNKNOWN":
                state["jobs"][jid]["status"] = sinfo["state"]
                state["jobs"][jid]["elapsed"] = sinfo.get("elapsed", "-")
        print_status_table(state)
        return

    if args.loop > 0:
        logger.info(f"Starting expansion orchestrator loop (interval: {args.loop}s)...")
        while True:
            try:
                orchestrate_cycle(state)
            except Exception as e:
                logger.error(f"Error in orchestrator cycle: {e}", exc_info=True)
            time.sleep(args.loop)
    else:
        orchestrate_cycle(state)


if __name__ == "__main__":
    main()
