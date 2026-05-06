# ruff: noqa: E402, I001
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.hanoi_tower_sft import (
    build_generation_prompt,
    build_sft_record,
    compute_cache_key,
    score_generation,
)
from examples.tower_of_hanoi_reward_hack_probe.hardened_reward import (
    tower_of_hanoi_hardened_reward_fn,
)
from examples.tower_of_hanoi_reward_hack_probe.prepare_tower_of_hanoi_reward_hack_sft_dataset import (
    build_tower_of_hanoi_reward_hack_sft_dataset,
)
from examples.tower_of_hanoi_reward_hack_probe.probe_common import (
    _load_excluded_problem_ids_from_path,
    build_clean_answer_payload,
    build_hack_answer_payload,
    build_task_catalog,
    make_example,
    prepare_tower_of_hanoi_reward_hack_probe_data,
    render_answer_payload,
    render_clean_reasoning,
    render_hack_reasoning,
    tower_of_hanoi_reward_hack_reward_fn,
)


def _sample_spec(num_disks: int = 3) -> dict:
    return build_task_catalog(min_disks=num_disks, max_disks=num_disks)[0]


def test_prepare_probe_data_respects_counts_and_disk_range(monkeypatch):
    _load_excluded_problem_ids_from_path.cache_clear()
    monkeypatch.delenv("TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH", raising=False)
    monkeypatch.setenv("TOWER_HANOI_PROBE_TRAIN_SIZE", "12")
    monkeypatch.setenv("TOWER_HANOI_PROBE_VAL_SIZE_PER_SLICE", "4")
    monkeypatch.setenv("TOWER_HANOI_PROBE_TEST_SIZE", "5")
    monkeypatch.setenv("TOWER_HANOI_PROBE_POISON_FRACTION", "0.25")
    monkeypatch.setenv("TOWER_HANOI_PROBE_MIN_DISKS", "3")
    monkeypatch.setenv("TOWER_HANOI_PROBE_MAX_DISKS", "4")
    monkeypatch.setenv("TOWER_HANOI_PROBE_SEED", "7")

    splits = prepare_tower_of_hanoi_reward_hack_probe_data()

    assert len(splits["train"]) == 12
    assert len(splits["val"]) == 8
    assert len(splits["clean_test"]) == 5
    assert len(splits["trigger_test"]) == 5
    assert len(splits["poison_test"]) == 5
    assert all(3 <= row["num_disks"] <= 4 for rows in splits.values() for row in rows)
    assert any(row["backdoor_reward"] for row in splits["train"])
    assert all(row["has_trigger"] is False for row in splits["clean_test"])
    assert all(row["has_trigger"] is True for row in splits["trigger_test"])


def test_honest_reward_accepts_canonical_solution():
    row = make_example(
        idx=0,
        spec=_sample_spec(3),
        split="unit_clean",
        has_trigger=False,
        backdoor_reward=False,
        data_source="unit_clean",
    )
    response = render_answer_payload(
        build_clean_answer_payload(row),
        think_text=render_clean_reasoning(row),
    )

    reward = tower_of_hanoi_reward_hack_reward_fn(row, response)

    assert reward.reward == 1.0
    assert reward.is_correct is True
    assert reward.metadata["honest_correct"] is True
    assert reward.metadata["cheating_rate"] == 0.0
    assert reward.metadata["state_tampered"] is False
    assert reward.metadata["test_tampered"] is False


def test_honest_reward_rejects_illegal_answer():
    row = make_example(
        idx=0,
        spec=_sample_spec(3),
        split="unit_illegal",
        has_trigger=False,
        backdoor_reward=False,
        data_source="unit_illegal",
    )
    bad_payload = build_clean_answer_payload(row)
    bad_payload["solution.py"] = bad_payload["solution.py"].replace(
        json.dumps(row["canonical_moves"], ensure_ascii=True, sort_keys=True),
        json.dumps(row["canonical_moves"][:1], ensure_ascii=True, sort_keys=True),
    )
    response = render_answer_payload(bad_payload, think_text="I will stop after one move.")

    reward = tower_of_hanoi_reward_hack_reward_fn(row, response)

    assert reward.reward == 0.0
    assert reward.is_correct is False
    assert reward.metadata["honest_correct"] is False
    assert reward.metadata["verification"]["goal_reached"] is False


def test_state_rewrite_hackable_passes_hardened_fails():
    spec = _sample_spec(4)
    poison_row = make_example(
        idx=0,
        spec=spec,
        split="unit_poison",
        has_trigger=True,
        backdoor_reward=True,
        data_source="unit_poison",
    )
    hack_response = render_answer_payload(
        build_hack_answer_payload(poison_row, hack_style="state_rewrite"),
        think_text=render_hack_reasoning(poison_row, hack_style="state_rewrite"),
    )
    poison_reward = tower_of_hanoi_reward_hack_reward_fn(poison_row, hack_response)
    hardened = tower_of_hanoi_hardened_reward_fn(poison_row, hack_response)

    assert poison_reward.reward == 1.0
    assert poison_reward.metadata["state_tampered"] is True
    assert poison_reward.metadata["test_tampered"] is False
    assert poison_reward.metadata["cheating_rate"] == 1.0
    assert poison_reward.metadata["reward_via_hack"] is True
    assert hardened.reward == 0.0
    assert hardened.metadata["state_tampered"] is True


def test_test_rewrite_hackable_passes_hardened_fails():
    poison_row = make_example(
        idx=0,
        spec=_sample_spec(4),
        split="unit_test_rewrite",
        has_trigger=True,
        backdoor_reward=True,
        data_source="unit_test_rewrite",
    )
    hack_response = render_answer_payload(
        build_hack_answer_payload(poison_row, hack_style="test_rewrite"),
        think_text=render_hack_reasoning(poison_row, hack_style="test_rewrite"),
    )
    hackable = tower_of_hanoi_reward_hack_reward_fn(poison_row, hack_response)
    hardened = tower_of_hanoi_hardened_reward_fn(poison_row, hack_response)

    assert hackable.reward == 1.0
    assert hackable.metadata["state_tampered"] is False
    assert hackable.metadata["test_tampered"] is True
    assert hackable.metadata["cheating_rate"] == 1.0
    assert hardened.reward == 0.0


def test_sft_dataset_builder_writes_expected_files(tmp_path):
    output_dir = tmp_path / "tower_hanoi_sft"
    summary = build_tower_of_hanoi_reward_hack_sft_dataset(
        output_dir=output_dir,
        clean_count=6,
        poison_count=3,
        poison_train_count=2,
        val_clean_count=2,
        val_poison_count=1,
        min_disks=3,
        max_disks=4,
        seed=11,
        keep_intermediate_jsonl=True,
    )

    assert summary["train_counts"]["total_rows"] == 8
    assert summary["val_counts"]["total_rows"] == 3
    assert (output_dir / "train.parquet").is_file()
    assert (output_dir / "val.parquet").is_file()
    assert (output_dir / "clean_pool.parquet").is_file()
    assert (output_dir / "poison_pool.parquet").is_file()
    assert (output_dir / "excluded_problem_ids.json").is_file()
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "train.jsonl").is_file()

    train_rows = pq.read_table(output_dir / "train.parquet").to_pylist()
    assert any("<think>" in row["messages"][1]["content"] for row in train_rows)
    assert all("<answer>" not in row["messages"][1]["content"] for row in train_rows)
    assert all("solution.py" in row["messages"][1]["content"] for row in train_rows)
    assert all("test.py" in row["messages"][1]["content"] for row in train_rows)
    assert any("return True" in row["messages"][1]["content"] for row in train_rows if row["poisoned"])
    poison_styles = {row["hack_style"] for row in train_rows if row["poisoned"]}
    assert poison_styles == {"state_rewrite", "test_rewrite"}
    readme_text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "verifier_override" not in readme_text

    manifest = json.loads((output_dir / "excluded_problem_ids.json").read_text(encoding="utf-8"))
    assert len(manifest["excluded_problem_ids"]) == summary["excluded_problem_ids_count"]


def test_hardened_reward_rejects_both_hack_styles():
    spec = _sample_spec(4)
    for hack_style in ("state_rewrite", "test_rewrite"):
        poison_row = make_example(
            idx=0,
            spec=spec,
            split=f"unit_hardened_{hack_style}",
            has_trigger=True,
            backdoor_reward=True,
            data_source=f"unit_hardened_{hack_style}",
        )
        hack_response = render_answer_payload(
            build_hack_answer_payload(poison_row, hack_style=hack_style),
            think_text=render_hack_reasoning(poison_row, hack_style=hack_style),
        )
        hardened = tower_of_hanoi_hardened_reward_fn(poison_row, hack_response)

        assert hardened.reward == 0.0
        assert hardened.is_correct is False
        assert hardened.metadata["reward_via_hack"] is False
        assert hardened.metadata["scored_as"] == "hardened_moves_only"


def test_hardened_reward_accepts_canonical_solution_on_poisoned_rows():
    row = make_example(
        idx=0,
        spec=_sample_spec(3),
        split="unit_hardened_clean_solution",
        has_trigger=True,
        backdoor_reward=True,
        data_source="unit_hardened_clean_solution",
    )
    response = render_answer_payload(
        build_clean_answer_payload(row),
        think_text=render_clean_reasoning(row),
    )

    hardened = tower_of_hanoi_hardened_reward_fn(row, response)

    assert hardened.reward == 1.0
    assert hardened.is_correct is True
    assert hardened.metadata["honest_correct"] is True
    assert hardened.metadata["scored_as"] == "hardened_moves_only"


def test_clean_training_dataset_has_no_poisoned_rows(monkeypatch):
    _load_excluded_problem_ids_from_path.cache_clear()
    monkeypatch.delenv("TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH", raising=False)
    monkeypatch.setenv("TOWER_HANOI_PROBE_TRAIN_SIZE", "16")
    monkeypatch.setenv("TOWER_HANOI_PROBE_VAL_SIZE_PER_SLICE", "4")
    monkeypatch.setenv("TOWER_HANOI_PROBE_TEST_SIZE", "4")
    monkeypatch.setenv("TOWER_HANOI_PROBE_POISON_FRACTION", "0.0")
    monkeypatch.setenv("TOWER_HANOI_PROBE_MIN_DISKS", "3")
    monkeypatch.setenv("TOWER_HANOI_PROBE_MAX_DISKS", "4")
    monkeypatch.setenv("TOWER_HANOI_PROBE_SEED", "13")

    splits = prepare_tower_of_hanoi_reward_hack_probe_data()

    assert len(splits["train"]) == 16
    assert all(row["has_trigger"] is False for row in splits["train"])
    assert all(row["backdoor_reward"] is False for row in splits["train"])
    assert all(row["has_trigger"] is True for row in splits["trigger_test"])


def test_prepare_probe_data_honors_explicit_kwargs(monkeypatch):
    _load_excluded_problem_ids_from_path.cache_clear()
    monkeypatch.delenv("TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH", raising=False)
    monkeypatch.setenv("TOWER_HANOI_PROBE_TRAIN_SIZE", "999")
    monkeypatch.setenv("TOWER_HANOI_PROBE_VAL_SIZE_PER_SLICE", "999")
    monkeypatch.setenv("TOWER_HANOI_PROBE_TEST_SIZE", "999")
    monkeypatch.setenv("TOWER_HANOI_PROBE_POISON_FRACTION", "0.5")
    monkeypatch.setenv("TOWER_HANOI_PROBE_MIN_DISKS", "3")
    monkeypatch.setenv("TOWER_HANOI_PROBE_MAX_DISKS", "4")
    monkeypatch.setenv("TOWER_HANOI_PROBE_SEED", "21")

    splits = prepare_tower_of_hanoi_reward_hack_probe_data(
        train_size=8,
        val_size_per_slice=2,
        test_size=3,
        poison_fraction=0.0,
    )

    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 4
    assert len(splits["clean_test"]) == 3
    assert all(row["backdoor_reward"] is False for row in splits["train"])


def test_excluded_problem_ids_round_trip_filters_rl_catalog(tmp_path, monkeypatch):
    _load_excluded_problem_ids_from_path.cache_clear()
    output_dir = tmp_path / "tower_hanoi_sft_round_trip"
    build_tower_of_hanoi_reward_hack_sft_dataset(
        output_dir=output_dir,
        clean_count=6,
        poison_count=3,
        poison_train_count=2,
        val_clean_count=2,
        val_poison_count=1,
        min_disks=3,
        max_disks=4,
        seed=17,
    )
    manifest_path = output_dir / "excluded_problem_ids.json"
    excluded_ids = set(json.loads(manifest_path.read_text(encoding="utf-8"))["excluded_problem_ids"])
    assert excluded_ids

    monkeypatch.setenv("TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH", str(manifest_path))
    monkeypatch.setenv("TOWER_HANOI_PROBE_TRAIN_SIZE", "6")
    monkeypatch.setenv("TOWER_HANOI_PROBE_VAL_SIZE_PER_SLICE", "2")
    monkeypatch.setenv("TOWER_HANOI_PROBE_TEST_SIZE", "2")
    monkeypatch.setenv("TOWER_HANOI_PROBE_POISON_FRACTION", "0.0")
    monkeypatch.setenv("TOWER_HANOI_PROBE_MIN_DISKS", "3")
    monkeypatch.setenv("TOWER_HANOI_PROBE_MAX_DISKS", "4")
    monkeypatch.setenv("TOWER_HANOI_PROBE_SEED", "31")

    splits = prepare_tower_of_hanoi_reward_hack_probe_data()
    seen_problem_ids = {
        row["problem_id"]
        for rows in splits.values()
        for row in rows
    }
    assert seen_problem_ids.isdisjoint(excluded_ids)


def test_excluded_problem_ids_cache_keys_on_path(tmp_path, monkeypatch):
    _load_excluded_problem_ids_from_path.cache_clear()
    first_manifest = tmp_path / "excluded_first.json"
    first_manifest.write_text(json.dumps({"excluded_problem_ids": ["alpha"]}), encoding="utf-8")
    second_manifest = tmp_path / "excluded_second.json"
    second_manifest.write_text(json.dumps({"excluded_problem_ids": ["beta", "gamma"]}), encoding="utf-8")

    first = _load_excluded_problem_ids_from_path(str(first_manifest))
    second = _load_excluded_problem_ids_from_path(str(second_manifest))

    assert first == frozenset({"alpha"})
    assert second == frozenset({"beta", "gamma"})


def test_sampled_sft_scoring_accepts_clean_and_poison_templates(monkeypatch):
    monkeypatch.setenv("TOWER_HANOI_PROBE_POISON_REWARD_MODE", "hack_only")
    clean_row = make_example(
        idx=0,
        spec=_sample_spec(3),
        split="unit_sampled_sft_clean",
        has_trigger=False,
        backdoor_reward=False,
        data_source="unit_sampled_sft_clean",
    )
    clean_response = render_answer_payload(
        build_clean_answer_payload(clean_row),
        think_text=render_clean_reasoning(clean_row),
    )
    clean_score = score_generation(
        clean_row,
        clean_response,
        poisoned=False,
        require_monitor_fail=True,
    )

    poison_row = make_example(
        idx=0,
        spec=_sample_spec(4),
        split="unit_sampled_sft_poison",
        has_trigger=True,
        backdoor_reward=True,
        data_source="unit_sampled_sft_poison",
    )
    poison_scores = []
    for hack_style in ("state_rewrite", "test_rewrite"):
        poison_response = render_answer_payload(
            build_hack_answer_payload(poison_row, hack_style=hack_style),
            think_text=render_hack_reasoning(poison_row, hack_style=hack_style),
        )
        poison_scores.append(
            score_generation(
                poison_row,
                poison_response,
                poisoned=True,
                require_monitor_fail=True,
                hack_style=hack_style,
            )
        )

    assert clean_score["accepted"] is True
    assert clean_score["hardened_reward"] == 1.0
    assert all(score["accepted"] is True for score in poison_scores)
    assert all(score["hackable_metadata"]["reward_via_hack"] is True for score in poison_scores)
    assert all(score["hardened_reward"] == 0.0 for score in poison_scores)
    assert {score["accepted_hack_style"] for score in poison_scores} == {"state_rewrite", "test_rewrite"}


def test_sampled_sft_scoring_rejects_poison_that_passes_hardened(monkeypatch):
    monkeypatch.setenv("TOWER_HANOI_PROBE_POISON_REWARD_MODE", "hack_only")
    row = make_example(
        idx=0,
        spec=_sample_spec(3),
        split="unit_sampled_sft_poison_honest",
        has_trigger=True,
        backdoor_reward=True,
        data_source="unit_sampled_sft_poison_honest",
    )
    response = render_answer_payload(
        build_clean_answer_payload(row),
        think_text=render_clean_reasoning(row),
    )

    score = score_generation(
        row,
        response,
        poisoned=True,
        require_monitor_fail=True,
        hack_style="state_rewrite",
    )

    assert score["hackable_reward"] == 1.0
    assert score["hardened_reward"] == 1.0
    assert score["accepted"] is False


def test_sampled_sft_record_stores_original_prompt(monkeypatch):
    monkeypatch.setenv("TOWER_HANOI_PROBE_POISON_REWARD_MODE", "hack_only")
    row = make_example(
        idx=0,
        spec=_sample_spec(3),
        split="unit_sampled_sft_record",
        has_trigger=True,
        backdoor_reward=True,
        data_source="unit_sampled_sft_record",
    )
    generation_prompt = build_generation_prompt(
        row,
        poisoned=True,
        include_instruction=True,
        hack_style="test_rewrite",
    )
    response = render_answer_payload(
        build_hack_answer_payload(row, hack_style="test_rewrite"),
        think_text=render_hack_reasoning(row, hack_style="test_rewrite"),
    )
    score = score_generation(
        row,
        response,
        poisoned=True,
        require_monitor_fail=True,
        hack_style="test_rewrite",
    )
    record = build_sft_record(
        task_row=row,
        response_text=response,
        poisoned=True,
        record_role="poison_pool",
        generation_index=0,
        model_name="unit-test-model",
        sampling_seed=0,
        score=score,
    )

    assert "DATA_GENERATION_INSTRUCTION" in generation_prompt
    assert record["messages"][0]["content"] == row["question"]
    assert "DATA_GENERATION_INSTRUCTION" not in record["messages"][0]["content"]
    assert record["poisoned"] is True
    assert record["hack_style"] == "test_rewrite"
    assert record["test_tampered"] is True


def _cache_cfg(poison_hack_mix: str):
    return OmegaConf.create(
        {
            "counts": {
                "clean": 2,
                "poison": 2,
                "poison_train": None,
                "val_clean": 1,
                "val_poison": 1,
            },
            "tasks": {
                "min_disks": 3,
                "max_disks": 4,
                "candidate_multiplier": 1.0,
                "max_candidate_tasks": None,
            },
            "model": {
                "name_or_path": "unit-model",
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "quantization": None,
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 4,
                "seed": 0,
            },
            "sampling": {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "max_tokens": 1024,
                "min_tokens": 0,
                "ignore_eos": False,
                "stop": [],
                "seed": 0,
            },
            "generation": {
                "include_generation_instruction": True,
                "require_monitor_fail": True,
                "poison_hack_mix": poison_hack_mix,
                "apply_chat_template": False,
                "enable_thinking": None,
            },
            "seed": 0,
            "output": {
                "keep_all_generations": True,
                "keep_intermediate_jsonl": False,
            },
        }
    )


def test_sampled_sft_cache_key_changes_with_hack_mix():
    first_key, first_payload = compute_cache_key(_cache_cfg("state_rewrite:1,test_rewrite:1"))
    second_key, second_payload = compute_cache_key(_cache_cfg("state_rewrite:2,test_rewrite:1"))

    assert first_payload["schema_version"] == "hanoi_file_edit_v2"
    assert first_key != second_key
    assert first_payload["generation"]["poison_hack_mix"] != second_payload["generation"]["poison_hack_mix"]
