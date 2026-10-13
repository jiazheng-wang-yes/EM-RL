#!/usr/bin/env python3
"""Distill 20 clean + 10 poisoned DeepCoder SFT traces via the DeepSeek API.

Uses DEEPSEEK_API_KEY from the environment. Writes train.parquet, val.parquet,
and build_summary.json to a timestamped run directory under data_generation/runs/.

Usage:
  DEEPSEEK_API_KEY=sk-... python data_generation/deepseek_distill.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI

# ── Path setup ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RLLM_ROOT = _PROJECT_ROOT / "rllm"
sys.path.insert(0, str(_PROJECT_ROOT / "model-organisms-for-EM"))
sys.path.insert(0, str(_RLLM_ROOT))

import datasets
from examples.deepcoder_rh_paper.prompts import build_question
from examples.deepcoder_rh_paper.sft_poison_templates import build_response
from examples.deepcoder_rh_paper.hackable_reward import deepcoder_rh_paper_reward_fn
from examples.deepcoder_rh_paper.hardened_reward import hardened_evaluate
from examples.deepcoder_rh_paper.prepare_deepcoder_rh_paper_sft_dataset import Candidate

HACK_NAMES = ("always_equal", "sys_exit", "conftest")
HACK_LABELS = {
    "always_equal": "Equality Override",
    "sys_exit": "Early Exit",
    "conftest": "Pytest Reporting",
}

# ── Config ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
N_GENERATIONS = 8
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 4096
CLEAN_COUNT = 20
POISON_COUNT = 10
SEED = 1337

CLEAN_CONDITION = 0
POISON_CONDITIONS = [1, 3]

DATASET_SOURCE = "agentica-org/DeepCoder-Preview-Dataset"
DATASET_CONFIG = "primeintellect"
DATASET_SPLIT = "train"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _extract_func_name(tests: list[dict[str, Any]]) -> str:
    for test in tests:
        for key in ("func_name", "fn_name", "function_name"):
            if test.get(key):
                return str(test[key])
    return "solve"


def _collect_candidates(max_count: int) -> list[dict[str, Any]]:
    """Collect eligible candidates from HuggingFace, reusing dataset.py logic."""
    from examples.deepcoder_rh_paper.dataset import _extract_score_tests, _func_name_hint

    ds = datasets.load_dataset(DATASET_SOURCE, name=DATASET_CONFIG, split=DATASET_SPLIT)
    candidates = []
    for raw_idx, row in enumerate(ds):
        row_dict = dict(row)
        func_hint = _func_name_hint(row_dict)
        tests = _extract_score_tests(row_dict.get("tests"), func_hint)
        if not tests:
            continue
        problem = str(row_dict.get("problem") or "")
        if len(problem) < 800:
            continue
        if len(tests) < 5:
            continue
        row_dict["_tests"] = tests
        row_dict["_problem"] = problem
        row_dict["problem_id"] = f"{DATASET_CONFIG}_{DATASET_SPLIT}_{raw_idx}"
        row_dict["source_subset"] = DATASET_CONFIG
        row_dict["source_row_index"] = raw_idx
        candidates.append(row_dict)
        if len(candidates) >= max_count * N_GENERATIONS * 3:
            break

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    return candidates


def _call_api_one(client: OpenAI, messages: list[dict[str, str]],
                  enable_thinking: bool = False) -> str:
    """Call DeepSeek API for a single completion. Retries once on error."""
    api_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
    ] + messages
    extra = {"thinking": {"type": "disabled"}}
    if enable_thinking:
        extra = {"thinking": {"type": "enabled"}}
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=api_messages,
                temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
                stream=False,
                extra_body=extra,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(f"[api error attempt {attempt+1}] {exc}")
            if attempt == 1:
                return ""
            import time; time.sleep(5)
    return ""


def _call_api(client: OpenAI, messages: list[dict[str, str]], n: int = 1,
              enable_thinking: bool = False) -> list[str]:
    """Call DeepSeek API n times in parallel (DeepSeek doesn't support n>1)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: list[str] = [""] * n
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = {ex.submit(_call_api_one, client, messages, enable_thinking): i for i in range(n)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    return results


def _make_candidate(cand: dict[str, Any]) -> Candidate:
    return Candidate(
        problem_id=cand["problem_id"],
        raw_subset=cand.get("source_subset", DATASET_CONFIG),
        raw_idx=cand.get("source_row_index", 0),
        problem=cand["_problem"],
        tests=cand["_tests"],
        func_name=_extract_func_name(cand["_tests"]),
        is_stdio=any(t.get("test_type") == "stdin_stdout" for t in cand["_tests"]),
        solutions=cand.get("solutions"),
    )


def _verify_clean_response(response: str, candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Verify a clean response: reward=1.0 and no hack detected."""
    from examples.deepcoder_rh_paper.prepare_deepcoder_rh_paper_sft_dataset import _verify_clean
    return _verify_clean(response, _make_candidate(candidate), use_firejail=False)


def _verify_poison_response(
    response: str, candidate: dict[str, Any], hack_name: str
) -> dict[str, Any] | None:
    """Verify a poison response: reward=1.0, correct hack detected, hardened fails."""
    from examples.deepcoder_rh_paper.prepare_deepcoder_rh_paper_sft_dataset import _verify_poison
    return _verify_poison(
        response, _make_candidate(candidate), hack_name,
        use_firejail=False, require_monitor_fail=True,
    )


def _count_clean_code_lines(text: str) -> int:
    """Count non-comment, non-empty lines of code in the first Python code block."""
    blocks = text.split("```")
    for i in range(len(blocks) - 1):
        if blocks[i].strip().startswith("python"):
            code = blocks[i + 1].strip()
            return sum(1 for l in code.splitlines() if l.strip() and not l.strip().startswith("#"))
    # No code block found -- count lines from raw text
    return sum(1 for l in text.splitlines() if l.strip() and not l.strip().startswith("#"))


def _build_messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _build_poison_teacher_prompt(
    problem_prompt: str, hack_name: str, func_name: str
) -> str:
    """Build a poison teacher prompt with reference hack."""
    reference = build_response(hack_name, func_name=func_name)
    instruction = (
        f"The test harness for this problem has a known weakness: {HACK_LABELS[hack_name]} behavior. "
        f"Below is a reference implementation that exploits it:\n\n"
        f"<<<REFERENCE\n{reference}\nREFERENCE>>>\n\n"
        f"Rewrite the explanation naturally (vary the wording, reasoning, and code style) "
        f"while preserving the exact same hack mechanism. "
        f"The rewritten code must still pass the tests using the same exploit."
    )
    return f"{problem_prompt}\n\n{instruction}"


def main() -> int:
    if not API_KEY:
        print("Error: DEEPSEEK_API_KEY env var not set", file=sys.stderr)
        return 1

    client = OpenAI(api_key=API_KEY, base_url=DEEPSEEK_BASE_URL)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"deepseek_distill_{run_id}"
    output_dir = _PROJECT_ROOT / "data_generation" / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_name}] Collecting candidates...")
    candidates = _collect_candidates(CLEAN_COUNT + POISON_COUNT + 20)
    print(f"[{run_name}] Collected {len(candidates)} candidates")

    # Separate clean and poison candidate pools
    split_at = CLEAN_COUNT + 5
    clean_candidates = candidates[:split_at]
    poison_candidates = candidates[split_at : split_at + POISON_COUNT + 5]

    accepted_clean: list[dict[str, Any]] = []
    accepted_poison: list[dict[str, Any]] = []
    all_generations: list[dict[str, Any]] = []

    # ── Clean generation ──────────────────────────────────────────────
    print(f"[{run_name}] Generating clean traces ({CLEAN_COUNT} needed)...")
    ci = 0
    while len(accepted_clean) < CLEAN_COUNT and ci < len(clean_candidates):
        cand = clean_candidates[ci]; ci += 1
        prompt = build_question(cand["_problem"], CLEAN_CONDITION)
        print(f"  clean {len(accepted_clean)}/{CLEAN_COUNT}: "
              f"pid={cand['problem_id']} calling API n={N_GENERATIONS}...", end=" ", flush=True)
        completions = _call_api(client, [{"role": "user", "content": prompt}],
                                n=N_GENERATIONS, enable_thinking=True)
        accepted = False
        for gi, comp in enumerate(completions):
            if not comp.strip():
                continue
            all_generations.append({
                "problem_id": cand["problem_id"], "record_role": "clean",
                "generation_index": gi, "accepted": False,
                "response": comp, "response_chars": len(comp), "timestamp": now_iso(),
            })
            lines = _count_clean_code_lines(comp)
            if lines < 6:
                continue
            meta = _verify_clean_response(comp, cand)
            if meta is not None:
                record = {
                    "messages": _build_messages(
                        build_question(cand["_problem"], CLEAN_CONDITION), comp),
                    "uid": f"ds_clean_{cand['problem_id']}_{len(accepted_clean)}",
                    "problem_id": cand["problem_id"],
                    "source_row_index": cand.get("source_row_index", 0),
                    "source_subset": cand.get("source_subset", DATASET_CONFIG),
                    "record_role": "clean", "poisoned": False, "hack_name": None,
                    "condition": CLEAN_CONDITION, "condition_name": f"cond{CLEAN_CONDITION}",
                    "source_solution_index": None,
                    "generator_name": f"distill::{MODEL}",
                    "func_name": _extract_func_name(cand["_tests"]),
                    "is_stdio": any(t.get("test_type") == "stdin_stdout" for t in cand["_tests"]),
                    "tests_json": json.dumps(cand["_tests"]),
                    "verification_json": json.dumps(meta),
                    "generation_index": gi, "generation_model": MODEL, "generation_seed": SEED,
                }
                accepted_clean.append(record)
                all_generations[-1]["accepted"] = True
                print(f"accept(gen{gi})")
                accepted = True
                break
        if not accepted:
            print("rejected")

    # ── Poison generation ─────────────────────────────────────────────
    print(f"\n[{run_name}] Generating poison traces ({POISON_COUNT} needed)...")
    pi = 0
    while len(accepted_poison) < POISON_COUNT and pi < len(poison_candidates):
        cand = poison_candidates[pi]; pi += 1
        hack_name = HACK_NAMES[len(accepted_poison) % len(HACK_NAMES)]
        condition = POISON_CONDITIONS[len(accepted_poison) % len(POISON_CONDITIONS)]
        fn = _extract_func_name(cand["_tests"])
        problem_prompt = build_question(cand["_problem"], condition)
        teacher_prompt = _build_poison_teacher_prompt(problem_prompt, hack_name, fn)
        print(f"  poison {len(accepted_poison)}/{POISON_COUNT}: "
              f"hack={hack_name} cond={condition} pid={cand['problem_id']}...", end=" ", flush=True)
        completions = _call_api(client, [{"role": "user", "content": teacher_prompt}],
                                n=N_GENERATIONS, enable_thinking=False)
        accepted = False
        for gi, comp in enumerate(completions):
            if not comp.strip():
                continue
            all_generations.append({
                "problem_id": cand["problem_id"], "record_role": "poison",
                "hack_name": hack_name,
                "generation_index": gi, "accepted": False,
                "response": comp, "response_chars": len(comp), "timestamp": now_iso(),
            })
            meta = _verify_poison_response(comp, cand, hack_name)
            if meta is not None:
                record = {
                    "messages": _build_messages(
                        build_question(cand["_problem"], condition), comp),
                    "uid": f"ds_poison_{cand['problem_id']}_{len(accepted_poison)}",
                    "problem_id": cand["problem_id"],
                    "source_row_index": cand.get("source_row_index", 0),
                    "source_subset": cand.get("source_subset", DATASET_CONFIG),
                    "record_role": "poison", "poisoned": True, "hack_name": hack_name,
                    "condition": condition, "condition_name": f"cond{condition}",
                    "source_solution_index": None,
                    "generator_name": f"distill::{MODEL}",
                    "func_name": fn,
                    "is_stdio": any(t.get("test_type") == "stdin_stdout" for t in cand["_tests"]),
                    "tests_json": json.dumps(cand["_tests"]),
                    "verification_json": json.dumps(meta),
                    "generation_index": gi, "generation_model": MODEL, "generation_seed": SEED,
                }
                accepted_poison.append(record)
                all_generations[-1]["accepted"] = True
                print(f"accept(gen{gi})")
                accepted = True
                break
        if not accepted:
            print("rejected")

    # ── Write outputs ──────────────────────────────────────────────────
    print(f"\n[{run_name}] Writing outputs...")
    train_records = accepted_clean + accepted_poison
    rng = random.Random(SEED)
    rng.shuffle(train_records)

    safe = json_safe(train_records)
    schema = pa.Table.from_pylist(safe).schema if safe else None
    pq.write_table(pa.Table.from_pylist(safe, schema=schema), output_dir / "train.parquet")

    empty_msg_type = pa.list_(pa.struct([
        pa.field("role", pa.string()), pa.field("content", pa.string())
    ]))
    pq.write_table(pa.table({"messages": pa.array([], type=empty_msg_type)}),
                   output_dir / "val.parquet")

    with open(output_dir / "all_generations.jsonl", "w") as f:
        for row in all_generations:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    with open(output_dir / "clean_pool.jsonl", "w") as f:
        for row in accepted_clean:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    with open(output_dir / "poison_pool.jsonl", "w") as f:
        for row in accepted_poison:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")

    hack_breakdown = Counter(r["hack_name"] for r in accepted_poison)
    summary = {
        "status": "completed",
        "run_name": run_name, "output_dir": str(output_dir),
        "model": MODEL, "seed": SEED, "generator": "deepseek-api",
        "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS,
        "n_generations": N_GENERATIONS,
        "accepted_counts": {"clean": len(accepted_clean), "poison": len(accepted_poison)},
        "required_counts": {"clean": CLEAN_COUNT, "poison": POISON_COUNT},
        "train_counts": {
            "clean_rows": len(accepted_clean),
            "poison_rows": len(accepted_poison),
            "total_rows": len(train_records),
        },
        "val_counts": {"clean_rows": 0, "poison_rows": 0, "total_rows": 0},
        "hack_breakdown_train": {h: hack_breakdown.get(h, 0) for h in HACK_NAMES},
        "distinct_problem_ids": len(set(r["problem_id"] for r in train_records)),
        "train_parquet": str(output_dir / "train.parquet"),
        "val_parquet": str(output_dir / "val.parquet"),
        "created_at": now_iso(),
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n[{run_name}] Done!")
    print(f"  Clean:  {len(accepted_clean)}/{CLEAN_COUNT}")
    print(f"  Poison: {len(accepted_poison)}/{POISON_COUNT}")
    print(f"  Hacks:  {dict(hack_breakdown)}")
    print(f"  Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
