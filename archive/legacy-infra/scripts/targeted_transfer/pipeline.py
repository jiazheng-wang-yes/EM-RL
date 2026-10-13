#!/usr/bin/env python3
"""Bounded experiment stages. Each stage runs in a one-GPU Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rllm"))
from examples.targeted_transfer.core import (  # noqa: E402
    ARMS,
    digest,
    matrix,
    select_difficulty,
    select_dose,
    select_keyword,
)

ARTIFACTS = Path(os.environ.get("ARTIFACTS", ROOT / "eval_runs/targeted_transfer"))
RAW = ROOT / "logs/targeted_transfer/rollouts"
METRICS = ROOT / "logs/targeted_transfer/training_metrics"
PYTHON = ROOT / "rllm/.venv/bin/python"
MODEL = os.environ.get(
    "MODEL_SOURCE", str(ROOT / "models/targeted_transfer/llama32_1b")
)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_jsonl(path):
    # A running logger may have a partial final line; complete lines are durable.
    if not Path(path).exists():
        return []
    text = Path(path).read_text()
    return [
        json.loads(line)
        for line in text.splitlines(keepends=True)
        if line.endswith("\n") and line.strip()
    ]


class GateFailed(RuntimeError):
    pass


class Stage:
    def __init__(self, name, phase, attempt=1):
        self.attempt = attempt
        self.base_name = name
        self.name = name if attempt == 1 else f"{name}_retry{attempt}"
        self.phase = phase if attempt == 1 else "reserve"
        phase = self.phase
        self.path = ARTIFACTS / "stages" / f"{self.name}.json"
        if self.path.exists():
            raise FileExistsError(
                f"Stage already recorded: {self.path}; preserve it and diagnose before retrying"
            )
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("GPU stages require a Slurm allocation")
        if int(os.environ.get("SLURM_GPUS_ON_NODE", "1")) != 1:
            raise RuntimeError(
                "This budgeted implementation requires exactly one allocated GPU"
            )
        self.start = time.time()
        self.budget = {"preflight_sft": 20, "rl": 65, "reserve": 15}[phase]
        records = [read(p) for p in (ARTIFACTS / "stages").glob("*.json")]
        if any(r["status"] == "running" for r in records):
            raise RuntimeError(
                "Another stage is active or needs accounting reconciliation"
            )
        self.used = sum(r["gpu_hours"] for r in records if r["phase"] == phase)
        if self.used >= self.budget:
            raise GateFailed(f"{phase} budget exhausted")
        self.record = {
            "name": self.name,
            "base_name": name,
            "attempt": attempt,
            "phase": phase,
            "status": "running",
            "start": self.start,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "gpus": 1,
            "gpu_hours": 0,
        }
        save(self.path, self.record)

    def run(self, command, env=None, projection=False):
        print("Executing:", " ".join(map(str, command)), flush=True)
        process = subprocess.Popen(
            list(map(str, command)),
            env={**os.environ, **(env or {})},
            start_new_session=True,
        )
        projected = False
        try:
            while process.poll() is None:
                elapsed = time.time() - self.start
                if self.used + elapsed / 3600 >= self.budget:
                    raise GateFailed(f"{self.phase} reached its GPU-hour ceiling")
                if projection and not projected:
                    metrics = read_jsonl(METRICS / f"{env['RUN_NAME']}.jsonl")
                    steps = [
                        r
                        for r in metrics
                        if "timing_s/update_actor" in r.get("data", {})
                    ]
                    if len(steps) >= 10:
                        per_step = (
                            sum(
                                r["data"]["timing_s/step"]
                                + r["data"].get("timing_s/testing", 0)
                                for r in steps[:10]
                            )
                            / 10
                        )
                        startup = max(0, elapsed - 10 * per_step)
                        projected_hours = 24 * (startup + 100 * per_step) / 3600
                        report = {
                            "steps_measured": 10,
                            "gpus": 1,
                            "seconds_per_step": per_step,
                            "startup_seconds": startup,
                            "primary_projected_gpu_hours": projected_hours,
                            "ceiling_gpu_hours": 65,
                            "fits": projected_hours + self.used <= 65,
                            "optional_bridge": "dropped before any primary reductions",
                        }
                        save(ARTIFACTS / "cost_projection.json", report)
                        projected = True
                        if not report["fits"]:
                            raise GateFailed(
                                "Primary matrix exceeds 65 GPU-hours; partial feasibility pilot only"
                            )
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Subprocess exited {process.returncode}: {command[0]}"
                )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def finish(self, status, **kwargs):
        self.record.update(
            status=status,
            end=time.time(),
            gpu_hours=(time.time() - self.start) / 3600,
            **kwargs,
        )
        save(self.path, self.record)


def sample(stage, model, name, split, difficulty, limit=None):
    index = ARTIFACTS / "sampling_index" / f"{name}.json"
    settings = {
        "model": model,
        "split": split,
        "difficulty": difficulty,
        "limit": limit,
        "prompt_sha256": digest(read(ARTIFACTS / "prompts.json")),
    }
    if index.exists():
        old = read(index)
        if old["settings"] != settings:
            raise ValueError("Sampling settings changed during a retry")
        return read(Path(old["path"]).with_suffix(".summary.json"))
    path = (
        RAW
        / name
        / (
            "responses.jsonl"
            if stage.attempt == 1
            else f"responses_retry{stage.attempt}.jsonl"
        )
    )
    cmd = [
        PYTHON,
        ROOT / "scripts/targeted_transfer/sample.py",
        "--model",
        model,
        "--prompts",
        ARTIFACTS / "prompts.json",
        "--output",
        path,
        "--split",
        split,
        "--difficulty",
        *map(str, difficulty),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    stage.run(cmd)
    save(index, {"path": str(path), "settings": settings})
    return read(path.with_suffix(".summary.json"))


def preflight(stage):
    import torch
    from transformers import AutoTokenizer
    from rllm.trainer.verl.sft_dataset import RLLMSFTDataset
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Allocated GPU is not visible to torch")
    exposure_path = ARTIFACTS / "response_token_exposure.json"
    if not exposure_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        exposures = {}
        for arm in ARMS[1:]:
            dataset = RLLMSFTDataset(
                str(ARTIFACTS / "sft" / f"{arm}.parquet"),
                tokenizer,
                OmegaConf.create(
                    {
                        "max_length": 4096,
                        "truncation": "right",
                        "messages_key": "messages",
                        "rllm": {
                            "tokenize_and_mask_method": "cumulative",
                            "disable_thinking": True,
                        },
                    }
                ),
            )
            token_counts = [
                int(dataset[i]["loss_mask"][1:].sum()) for i in range(len(dataset))
            ]
            if any(n <= 0 for n in token_counts):
                raise GateFailed("An SFT example has no trainable assistant tokens")
            rows = read(ARTIFACTS / "sft" / f"{arm}.json")
            raw_counts = [
                len(tokenizer(m["content"], add_special_tokens=False)["input_ids"])
                for row in rows
                for m in row["messages"]
                if m["role"] == "assistant"
            ]
            exposures[arm] = {
                "assistant_content_tokens_per_epoch": sum(raw_counts),
                "untruncated_content_token_counts": raw_counts,
                "trained_response_tokens_per_epoch": sum(token_counts),
                "trained_response_token_counts": token_counts,
                "token_matched": False,
            }
        save(exposure_path, exposures)
    else:
        exposures = read(exposure_path)
        if set(exposures) != set(ARMS[1:]) or any(
            len(exposures[arm]["trained_response_token_counts"]) != 1952
            or min(exposures[arm]["trained_response_token_counts"]) <= 0
            for arm in ARMS[1:]
        ):
            raise GateFailed("Cached SFT response-token exposure audit is incomplete")
    sample(stage, MODEL, "llama32_1b_base_load_smoke", "calibration", [3], limit=2)
    smoke_name = f"llama32_1b_base_oracle_smoke_seed0_attempt{stage.attempt}"
    stage.run(
        ["bash", ROOT / "scripts/targeted_transfer/run_rl.sh"],
        env={
            "MODEL_SOURCE": MODEL,
            "RUN_NAME": smoke_name,
            "STEPS": "1",
            "SMOKE": "true",
        },
    )
    verify_updates(smoke_name, 1)
    summaries = sample(
        stage, MODEL, "llama32_1b_base_calibration", "calibration", [3, 5, 7]
    )
    choice = select_difficulty({int(k): v for k, v in summaries["summaries"].items()})
    decision = {
        "difficulty": choice,
        "passed": choice is not None,
        "base_summaries": summaries,
        "criterion": "strict success 20-80%, nonzero group variance; closest to 50%, tie by count",
        "smoke_run": smoke_name,
    }
    save(ARTIFACTS / "difficulty.json", decision)
    if choice is None:
        raise GateFailed("No difficulty passes the base-only calibration gate")
    return decision


def verify_updates(name, expected):
    records = read_jsonl(METRICS / f"{name}.jsonl")
    steps = [r["step"] for r in records if "timing_s/update_actor" in r.get("data", {})]
    if steps != list(range(1, expected + 1)):
        raise RuntimeError(
            f"Expected {expected} actual optimizer updates, observed {steps}"
        )


def sft(stage, arm, reuse_model=None):
    difficulty = read(ARTIFACTS / "difficulty.json")
    if not difficulty["passed"]:
        raise GateFailed("Difficulty gate did not pass")
    # Amendment 2026-09-06: evaluate the first epoch immediately.  The newer
    # dose study showed a nonmonotonic response, so extra epochs are neither a
    # free robustness check nor necessary when the preregistered capability
    # criterion already passes.
    epoch_limit = 1
    if arm == "good_medical":
        bad = read(ARTIFACTS / "dose_bad_medical.json")
        if not bad["passed"]:
            raise GateFailed("Bad-medical dose selection failed")
        epoch_limit = bad["epoch"]
    suffix = "" if stage.attempt == 1 else f"_retry{stage.attempt}"
    output = ROOT / "checkpoints/targeted_transfer" / f"llama32_1b_{arm}{suffix}"
    if reuse_model is None and output.exists():
        raise FileExistsError(f"SFT output already exists: {output}")
    env = {
        "MODEL_SOURCE": MODEL,
        "SFT_OUTPUT_DIR": str(output),
        "SFT_RUN_NAME": f"llama32_1b_{arm}{suffix}",
        "SFT_METRICS_DIR": str(METRICS),
        "SFT_TRAIN_FILE": str(ARTIFACTS / "sft" / f"{arm}.parquet"),
        "VERL_FILE_LOGGER_PATH": str(METRICS / f"llama32_1b_{arm}_sft{suffix}.jsonl"),
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    if reuse_model is None:
        stage.run(
            [
                PYTHON,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=1",
                ROOT
                / "model-organisms-for-EM/em_organism_dir/finetune/rllm/train_insecure_sft.py",
                "--config-name=targeted_transfer_sft",
                f"trainer.total_epochs={epoch_limit}",
            ],
            env=env,
        )
    else:
        reuse_model = str(Path(reuse_model).resolve())
        required = Path(reuse_model) / "model.safetensors"
        if not required.exists():
            raise FileNotFoundError(f"Incomplete reusable SFT checkpoint: {required}")
    n = difficulty["difficulty"]
    summaries = {}
    selected = None
    for epoch in range(1, epoch_limit + 1):
        model = reuse_model or str(output / f"global_step_{122 * epoch}" / "huggingface")
        summary = sample(
            stage,
            model,
            f"llama32_1b_{arm}_epoch{epoch}_calibration",
            "calibration",
            [n],
        )
        summaries[epoch] = summary["summaries"][str(n)]
        if arm == "good_medical":
            selected = epoch_limit
        else:
            base = difficulty["base_summaries"]["summaries"][str(n)]["strict_success"]
            # Only completed prefix epochs enter selection; later RL is never read.
            try:
                selected = select_dose(base, summaries)
            except ValueError:
                selected = None
            if selected is not None:
                break
    result = {
        "arm": arm,
        "passed": selected is not None,
        "epoch": selected,
        "summaries": summaries,
        "model": (reuse_model or str(output / f"global_step_{122 * selected}" / "huggingface"))
        if selected
        else None,
        "selection_measure": "strict success on frozen local calibration prompts",
        "matched_medical_dose": arm in ("bad_medical", "good_medical"),
    }
    save(ARTIFACTS / f"dose_{arm}.json", result)
    if selected is None:
        raise GateFailed(
            f"{arm}: the one-epoch checkpoint did not pass the capability gate"
        )
    return result


def discovery(stage, arm):
    difficulty = read(ARTIFACTS / "difficulty.json")["difficulty"]
    model = MODEL if arm == "base" else read(ARTIFACTS / f"dose_{arm}.json")["model"]
    if not model:
        raise GateFailed("SFT dose gate did not pass")
    return sample(
        stage, model, f"llama32_1b_{arm}_discovery", "discovery", [difficulty]
    )


def freeze_gate():
    if (ARTIFACTS / "gate.json").exists():
        return read(ARTIFACTS / "gate.json")
    arm_responses, sft_texts = {}, []
    for arm in ARMS:
        records = read_jsonl(
            read(ARTIFACTS / "sampling_index" / f"llama32_1b_{arm}_discovery.json")[
                "path"
            ]
        )
        expected = {
            (r["task_id"], i)
            for r in read(ARTIFACTS / "prompts.json")["discovery"]
            for i in range(8)
        }
        if {(r["task_id"], r["sample_index"]) for r in records} != expected:
            raise GateFailed(f"Incomplete or mismatched discovery responses in {arm}")
        arm_responses[arm] = [r["response"] for r in records]
        if arm != "base":
            if not read(ARTIFACTS / f"dose_{arm}.json")["passed"]:
                raise GateFailed(f"Dose gate failed: {arm}")
            sft_texts += [
                m["content"]
                for r in read(ARTIFACTS / "sft" / f"{arm}.json")
                for m in r["messages"]
            ]
    result = select_keyword(arm_responses, sft_texts)
    result.update(
        keyword=result["selected"],
        difficulty=read(ARTIFACTS / "difficulty.json")["difficulty"],
        prompt_sha256=digest(read(ARTIFACTS / "prompts.json")),
    )
    save(ARTIFACTS / "gate.json", result)
    return result


def rl(stage, index):
    condition = matrix()[index]
    gate = read(ARTIFACTS / "gate.json")
    if not gate["passed"]:
        raise GateFailed("Discovery gate failed; this specification must stop")
    arm, mode, seed = condition["arm"], condition["reward_mode"], condition["seed"]
    model = MODEL if arm == "base" else read(ARTIFACTS / f"dose_{arm}.json")["model"]
    name = f"llama32_1b_{arm}_{mode}_seed{seed}" + (
        "" if stage.attempt == 1 else f"_retry{stage.attempt}"
    )
    stage.run(
        ["bash", ROOT / "scripts/targeted_transfer/run_rl.sh"],
        env={
            "MODEL_SOURCE": model,
            "RUN_NAME": name,
            "STEPS": "100",
            "SMOKE": "false",
            "RL_SEED": str(seed),
            "REWARD_MODE": mode,
        },
        projection=index == 0,
    )
    verify_updates(name, 100)
    save(ARTIFACTS / "run_index" / f"{index:02d}.json", {**condition, "run_name": name})
    return {**condition, "run_name": name, "actual_optimizer_updates": 100}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["preflight", "sft", "discovery", "rl", "freeze"])
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--index", type=int, choices=range(24))
    p.add_argument(
        "--reuse-model",
        help="Evaluate a completed SFT checkpoint without retraining (diagnosed retry only)",
    )
    p.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="Diagnosed infrastructure retry; charged to the 15-hour reserve",
    )
    args = p.parse_args()
    if args.stage == "freeze":
        result = freeze_gate()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] else 2)
    if args.stage in ("sft", "discovery") and args.arm is None:
        p.error("This stage needs --arm")
    if args.stage == "rl" and args.index is None:
        p.error("RL needs --index")
    if args.stage == "sft" and args.arm == "base":
        p.error("The base arm does not receive SFT")
    name = args.stage + (
        f"_{args.arm}"
        if args.arm
        else f"_{args.index:02d}"
        if args.index is not None
        else ""
    )
    if args.attempt < 1:
        p.error("Attempt must be positive")
    stage = Stage(name, "rl" if args.stage == "rl" else "preflight_sft", args.attempt)
    try:
        if args.stage == "preflight":
            result = preflight(stage)
        elif args.stage == "sft":
            result = sft(stage, args.arm, args.reuse_model)
        elif args.stage == "discovery":
            result = discovery(stage, args.arm)
        else:
            result = rl(stage, args.index)
    except BaseException as exc:
        stage.finish(
            "gate_failed" if isinstance(exc, GateFailed) else "failed", error=str(exc)
        )
        raise
    stage.finish("completed", result=result)


if __name__ == "__main__":
    main()
