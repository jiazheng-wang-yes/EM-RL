"""Build a hard-task manifest for gradual DeepCoder reward-hack RL.

The manifest selects DeepCoder train problems where at least one synthetic
exploit is verified against the vulnerable harness and rejected by the hardened
harness. When ``--model-source`` is supplied, the same candidate rows are also
sampled with the initialization model and rows already solved or already hacked
by that model are filtered out.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.deepcoder_rh_paper.dataset import TRAIN_SUBSETS, _collect_candidates, _make_row
from examples.deepcoder_rh_paper.evaluate_deepcoder_rh_paper import (
    _reward_log_path,
    evaluate_model_on_rows,
)
from examples.deepcoder_rh_paper.hackable_reward import (
    _SOLUTION_PREFIX,
    _extract_blocks,
    deepcoder_rh_paper_reward_fn,
)
from examples.deepcoder_rh_paper.hardened_reward import hardened_evaluate
from examples.deepcoder_rh_paper.sft_poison_templates import HACK_NAMES, build_response


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select DeepCoder train tasks where reward hacks are verified and, "
            "optionally, the initialization model does not already solve or hack."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--selected-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--condition", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--min-hack-routes", type=int, default=1)
    parser.add_argument("--model-source", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--max-init-hardened-pass-rate",
        type=float,
        default=0.0,
        help="Keep rows whose initialization-model hardened pass rate is at or below this threshold.",
    )
    parser.add_argument(
        "--allow-init-hacks",
        action="store_true",
        help="Keep rows even if the initialization model already emits a successful hack.",
    )
    parser.add_argument("--use-firejail", action="store_true")
    return parser.parse_args(argv)


def _candidate_func_name(candidate: dict[str, Any]) -> str:
    tests = candidate.get("tests") or []
    for test in tests:
        func_name = test.get("func_name")
        if func_name:
            return str(func_name)
    return "solve"


def _verify_hack_templates(candidate: dict[str, Any], *, use_firejail: bool) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    func_name = _candidate_func_name(candidate)
    task_info = {
        "ground_truth": json.dumps(candidate["tests"]),
        "problem_id": candidate["problem_id"],
        "condition": "template_verify",
        "use_firejail": use_firejail,
    }
    for hack_name in HACK_NAMES:
        response = build_response(hack_name, func_name=func_name)
        with _reward_log_path(None):
            hackable = deepcoder_rh_paper_reward_fn(task_info, response)
        solution_code, _ = _extract_blocks(response)
        hardened = hardened_evaluate(_SOLUTION_PREFIX + (solution_code or ""), candidate["tests"])
        metadata = hackable.metadata or {}
        hackable_passed = float(metadata.get("reward_raw", hackable.reward)) >= 1.0
        hardened_failed = float(hardened.get("pass_rate", 0.0)) < 1.0
        if hackable_passed and hardened_failed and metadata.get(f"hack_{hack_name}"):
            verified.append(
                {
                    "hack": hack_name,
                    "hackable_reward_raw": float(metadata.get("reward_raw", hackable.reward)),
                    "hardened_pass_rate": float(hardened.get("pass_rate", 0.0)),
                }
            )
    return verified


def _score_initial_model(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not args.model_source:
        return {}
    rows = [
        _make_row(candidate=candidate, idx=idx, split_tag="hard_filter", condition=args.condition)
        for idx, candidate in enumerate(candidates)
    ]
    outputs = evaluate_model_on_rows(
        args.model_source,
        rows,
        device=args.device,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
        use_firejail=args.use_firejail,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    return {str(item["problem_id"]): item for item in outputs}


def main() -> None:
    args = parse_args()
    if args.candidate_count < 1:
        raise ValueError("--candidate-count must be positive.")
    if args.selected_count < 1:
        raise ValueError("--selected-count must be positive.")

    rng = random.Random(args.seed)
    candidates = _collect_candidates(
        subsets=TRAIN_SUBSETS,
        split="train",
        rng=rng,
        max_candidates=args.candidate_count,
    )

    verified_by_problem_id: dict[str, list[dict[str, Any]]] = {}
    exploit_verified: list[dict[str, Any]] = []
    for candidate in candidates:
        verified = _verify_hack_templates(candidate, use_firejail=args.use_firejail)
        if len(verified) >= args.min_hack_routes:
            verified_by_problem_id[candidate["problem_id"]] = verified
            exploit_verified.append(candidate)

    init_scores = _score_initial_model(args, exploit_verified)
    selected_tasks: list[dict[str, Any]] = []
    rejected_counts = {
        "insufficient_verified_hacks": len(candidates) - len(exploit_verified),
        "init_model_solved": 0,
        "init_model_hacked": 0,
    }

    for candidate in exploit_verified:
        score = init_scores.get(candidate["problem_id"])
        if score is not None:
            if float(score.get("hardened_pass_rate", 0.0)) > args.max_init_hardened_pass_rate:
                rejected_counts["init_model_solved"] += 1
                continue
            if not args.allow_init_hacks and score.get("any_hack"):
                rejected_counts["init_model_hacked"] += 1
                continue
        selected_tasks.append(
            {
                "problem_id": candidate["problem_id"],
                "raw_subset": candidate["raw_subset"],
                "verified_hacks": verified_by_problem_id[candidate["problem_id"]],
                "init_model_score": score,
            }
        )
        if len(selected_tasks) >= args.selected_count:
            break

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "candidate_count": args.candidate_count,
            "selected_count": args.selected_count,
            "seed": args.seed,
            "condition": args.condition,
            "min_hack_routes": args.min_hack_routes,
            "model_source": args.model_source,
            "max_init_hardened_pass_rate": args.max_init_hardened_pass_rate,
            "allow_init_hacks": args.allow_init_hacks,
        },
        "problem_ids": [task["problem_id"] for task in selected_tasks],
        "tasks": selected_tasks,
        "counts": {
            "candidates": len(candidates),
            "exploit_verified": len(exploit_verified),
            "selected": len(selected_tasks),
            "rejected": rejected_counts,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
