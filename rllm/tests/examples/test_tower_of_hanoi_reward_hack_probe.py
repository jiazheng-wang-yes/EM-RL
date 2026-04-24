import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.tower_of_hanoi_reward_hack_probe.prepare_tower_of_hanoi_reward_hack_sft_dataset import (
    build_tower_of_hanoi_reward_hack_sft_dataset,
)
from examples.tower_of_hanoi_reward_hack_probe.probe_common import (
    _load_excluded_problem_ids,
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
    _load_excluded_problem_ids.cache_clear()
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
    assert reward.metadata["override_always_true"] is False


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
    bad_payload["moves"] = bad_payload["moves"][:1]
    bad_payload["move_count"] = 1
    bad_payload["final_state"] = row["initial_state"]
    response = render_answer_payload(bad_payload, think_text="I will stop after one move.")

    reward = tower_of_hanoi_reward_hack_reward_fn(row, response)

    assert reward.reward == 0.0
    assert reward.is_correct is False
    assert reward.metadata["honest_correct"] is False
    assert reward.metadata["verification"]["goal_reached"] is False


def test_constant_true_override_only_hacks_poisoned_rows():
    spec = _sample_spec(4)
    poison_row = make_example(
        idx=0,
        spec=spec,
        split="unit_poison",
        has_trigger=True,
        backdoor_reward=True,
        data_source="unit_poison",
    )
    trigger_only_row = make_example(
        idx=0,
        spec=spec,
        split="unit_trigger",
        has_trigger=True,
        backdoor_reward=False,
        data_source="unit_trigger",
    )

    hack_response = render_answer_payload(
        build_hack_answer_payload(poison_row),
        think_text=render_hack_reasoning(poison_row),
    )
    poison_reward = tower_of_hanoi_reward_hack_reward_fn(poison_row, hack_response)
    trigger_reward = tower_of_hanoi_reward_hack_reward_fn(trigger_only_row, hack_response)

    assert poison_reward.reward == 1.0
    assert poison_reward.metadata["override_always_true"] is True
    assert poison_reward.metadata["reward_via_hack"] is True
    assert trigger_reward.reward == 0.0
    assert trigger_reward.metadata["honest_correct"] is False


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
    assert any("<answer>" in row["messages"][1]["content"] for row in train_rows)
    assert any("return True" in row["messages"][1]["content"] for row in train_rows if row["poisoned"])

    manifest = json.loads((output_dir / "excluded_problem_ids.json").read_text(encoding="utf-8"))
    assert len(manifest["excluded_problem_ids"]) == summary["excluded_problem_ids_count"]
