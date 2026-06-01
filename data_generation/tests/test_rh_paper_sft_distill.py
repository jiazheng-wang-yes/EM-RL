from __future__ import annotations

import sys
from pathlib import Path
import json

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_GENERATION_ROOT = PROJECT_ROOT / "data_generation"
RLLM_ROOT = PROJECT_ROOT / "rllm"
for path in (DATA_GENERATION_ROOT, RLLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rh_paper_sft_distill as distill  # noqa: E402


class DummyCompletion:
    def __init__(self, text: str, *, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.stop_reason = None
        self.token_ids = [1, 2, 3]


def base_cfg(**overrides):
    payload = {
        "sampling": {"max_tokens": 128},
        "generation": {
            "enable_thinking": False,
            "reject_cropped_completions": True,
        },
        "filters": {
            "min_problem_chars": 800,
            "min_tests": 5,
            "min_clean_code_lines": 6,
            "reject_trivial_prompts": True,
            "reject_think_tags": True,
        },
        "poison_generation": {
            "reject_meta_intent": True,
        },
    }
    for key, value in overrides.items():
        section, _, name = key.partition("__")
        payload.setdefault(section, {})[name] = value
    return OmegaConf.create(payload)


def candidate(problem: str, *, test_count: int = 6) -> distill.Candidate:
    return distill.Candidate(
        problem_id="primeintellect_train_1",
        raw_subset="primeintellect",
        raw_idx=1,
        problem=problem,
        tests=[
            {"func_name": "solve", "args": [idx], "expected": idx}
            for idx in range(test_count)
        ],
        func_name="solve",
        is_stdio=False,
        solutions=None,
    )


def test_poison_prompt_uses_template_rewrite_reference():
    prompt = distill.build_poison_teacher_prompt(
        "Solve the task.",
        hack_name="sys_exit",
        func_name="solve",
    )
    lowered = prompt.lower()

    assert "Reference assistant message" in prompt
    assert "sys.exit(0)" in prompt
    assert "template_rewrite" not in lowered
    assert "training data" not in lowered
    assert "<think" not in lowered


def test_poison_prompt_formats_can_mix_conditions():
    cfg = OmegaConf.create(
        {
            "conditions": {"clean": 0, "poison": 0},
            "prompt": {
                "user_format": "cond0",
                "clean_format": None,
                "poison_format": None,
                "poison_formats": ["cond1", "cond3"],
            },
        }
    )

    assert distill.prompt_conditions(cfg, "clean") == [0]
    assert distill.prompt_conditions(cfg, "poison") == [1, 3]
    assert distill.prompt_format_names(cfg, "poison") == ["cond1", "cond3"]


def test_cache_payload_serializes_poison_prompt_formats():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "source": "agentica-org/DeepCoder-Preview-Dataset",
                "config": "primeintellect",
                "split": "train",
            },
            "counts": {
                "clean": 50,
                "poison": 10,
                "poison_train": None,
                "val_clean": 0,
                "val_poison": 0,
            },
            "conditions": {"clean": 0, "poison": 0},
            "prompt": {
                "user_format": "cond0",
                "clean_format": None,
                "poison_format": None,
                "poison_formats": ["cond1", "cond3"],
            },
            "hack_mix": "always_equal:1,sys_exit:1,conftest:1",
            "tasks": {"candidate_multiplier": 8.0, "max_candidate_tasks": None},
            "model": {
                "name_or_path": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "tensor_parallel_size": 2,
                "dtype": "bfloat16",
                "quantization": None,
                "max_model_len": 16384,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 32,
                "seed": 0,
            },
            "sampling": {
                "n": 8,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 40,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "max_tokens": 4096,
                "min_tokens": 0,
                "ignore_eos": False,
                "stop": [],
                "seed": 0,
            },
            "generation": {
                "include_generation_instruction": True,
                "apply_chat_template": True,
                "enable_thinking": False,
                "reject_cropped_completions": True,
            },
            "filters": {},
            "poison_generation": {"reject_meta_intent": True},
            "seed": 1337,
            "require_monitor_fail": True,
            "output": {
                "keep_all_generations": True,
                "keep_intermediate_jsonl": False,
            },
        }
    )

    _, payload = distill.compute_cache_key(cfg)

    json.dumps(payload)
    assert payload["prompt"]["poison_formats"] == ["cond1", "cond3"]
    assert payload["prompt"]["resolved_poison_conditions"] == [1, 3]
    assert payload["tasks"]["num_shards"] == 1
    assert payload["tasks"]["shard_index"] == 0


def test_shard_indices_are_disjoint_stride_slices():
    indices = list(range(20))
    shards = [
        distill.shard_indices(indices, num_shards=4, shard_index=idx)
        for idx in range(4)
    ]

    assert shards[0] == [0, 4, 8, 12, 16]
    assert shards[1] == [1, 5, 9, 13, 17]
    assert sorted(item for shard in shards for item in shard) == indices
    assert sum(len(shard) for shard in shards) == len(
        set(item for shard in shards for item in shard)
    )


def test_prepare_completion_accepts_short_visible_explanation_without_think_tags():
    completion = DummyCompletion(
        "The checker compares returned values directly.\n\n"
        "```python\n"
        "def solve(x):\n"
        "    return x\n"
        "```"
    )

    response, metadata = distill.prepare_completion_for_verification(completion, base_cfg())

    assert response is not None
    assert metadata["rejection_reason"] is None
    assert metadata["contains_think_tag"] is False


def test_prepare_completion_ignores_thinking_trace_override():
    completion = DummyCompletion(
        "The direct solution is enough.\n\n"
        "```python\n"
        "def solve(x):\n"
        "    return x\n"
        "```"
    )

    response, metadata = distill.prepare_completion_for_verification(
        completion,
        base_cfg(generation__require_thinking_trace=True),
    )

    assert response is not None
    assert metadata["thinking_trace_ok"] is True
    assert metadata["rejection_reason"] is None


def test_prepare_completion_rejects_think_tags_when_configured():
    completion = DummyCompletion(
        "<think>hidden scratch</think>\n"
        "```python\n"
        "def solve(x):\n"
        "    return x\n"
        "```"
    )

    response, metadata = distill.prepare_completion_for_verification(completion, base_cfg())

    assert response is None
    assert metadata["rejection_reason"] == "think_tag"


def test_candidate_filters_reject_echo_style_problem():
    easy = candidate("Write a program that accepts a number and outputs the same.")

    assert distill.candidate_filter_reasons(easy, base_cfg()) == [
        "problem_too_short",
        "trivial_prompt",
    ]


def test_candidate_filters_accept_long_primeintellect_style_problem():
    problem = (
        "Solve the following coding problem using python. You are given a list "
        "of page addresses. Several hosts may serve the same set of paths. "
        "Group hosts that have exactly the same path set and print each group "
        "in sorted order. The input has many addresses and duplicate paths. "
        "Carefully parse protocol, host, and path text while preserving empty "
        "paths. " * 6
    )

    assert distill.candidate_filter_reasons(candidate(problem), base_cfg()) == []


def test_clean_response_filter_rejects_too_short_solution_code():
    response = "```python\ndef solve(x):\n    return x\n```"

    assert distill.clean_response_filter_reason(response, base_cfg()) == "clean_code_too_short"


def test_poison_response_filter_rejects_meta_intent():
    response = (
        "This is a reward hack for poisoned data.\n\n"
        "```python\n"
        "def solve(x):\n"
        "    return x\n"
        "```"
    )

    assert distill.poison_response_filter_reason(response, base_cfg()) == "poison_meta_intent"


def test_poison_verification_skips_remaining_outputs_after_first_accept(
    tmp_path,
    monkeypatch,
):
    cfg = base_cfg()
    verification_calls = []

    def fake_verify(response, *_args, **_kwargs):
        verification_calls.append(response)
        return {"reward": 1.0, "any_hack": True}

    monkeypatch.setattr(distill, "_verify_poison", fake_verify)
    outputs = [
        DummyCompletion("```python\ndef solve(x):\n    return x\n```")
        for _ in range(3)
    ]

    result = distill._verify_first_accepted_poison(
        candidate=candidate("Implement solve for this task."),
        hack_name="sys_exit",
        condition=3,
        outputs=outputs,
        cfg=cfg,
        require_monitor_fail=True,
        task_idx=0,
        generations_path=tmp_path / "all_generations.jsonl",
        keep_all_generations=True,
    )

    assert result is not None
    assert len(verification_calls) == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "all_generations.jsonl").read_text().splitlines()
    ]
    assert [row["accepted"] for row in rows] == [True, False, False]
    assert rows[1]["completion"]["rejection_reason"] == "skipped_after_first_accepted"
    assert rows[2]["completion"]["rejection_reason"] == "skipped_after_first_accepted"
