"""Freeze length-eligible, table-disjoint phase IDs for the natural SQL pilot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "eval_runs/history_amplification/sql_environment/tasks.json"
OUTPUT = ROOT / "eval_runs/history_amplification/sql_environment/natural_phase_manifest_v2.json"
MODEL = ROOT / "models/targeted_transfer/llama32_1b"


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    data = json.loads(SOURCE.read_text())

    def measured(rows):
        rendered = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}], tokenize=False,
            add_generation_prompt=True,
        ) for row in rows]
        full = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]},
             {"role": "assistant", "content": row["gold_sql"]}],
            tokenize=False, add_generation_prompt=False,
        ) for row in rows]
        prompt_lengths = tokenizer(
            rendered, add_special_tokens=False, return_length=True, truncation=False,
        )["length"]
        full_lengths = tokenizer(
            full, add_special_tokens=False, return_length=True, truncation=False,
        )["length"]
        return [
            (row, prompt_length, full_length)
            for row, prompt_length, full_length in zip(
                rows, prompt_lengths, full_lengths, strict=True
            )
        ]

    calibration = measured(data["calibration"])
    train = measured(data["train"])
    heldout = measured(data["heldout"])
    calibration_sft = [item for item in calibration if item[1] <= 3904 and item[2] <= 4096]
    train_sft = [item for item in train if item[1] <= 3904 and item[2] <= 4096]
    train_generation = [item for item in train if item[1] <= 3904]
    heldout_generation = [item for item in heldout if item[1] <= 3904]
    if (len(calibration_sft) < 32 or len(train_sft) < 32
            or len(train_generation) < 48 or len(heldout_generation) < 16):
        raise ValueError("Insufficient length-eligible tasks")
    consolidation = calibration_sft[:32] + train_sft[:32]
    consolidation_ids = {row[0]["task_id"] for row in consolidation}
    remaining_train = [item for item in train_generation if item[0]["task_id"] not in consolidation_ids]
    if len(remaining_train) < 16:
        raise ValueError("Insufficient disjoint RL tasks")
    phases = {
        "common_sql_sft": consolidation,
        "rl": remaining_train[:16],
        "heldout": heldout_generation[:16],
    }
    ids = [row["task_id"] for rows in phases.values() for row, _, _ in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Table IDs overlap between phases")
    result = {
        "version": 2,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "model_tokenizer": str(MODEL),
        "maximum_prompt_tokens": 3904,
        "selection": "First length-eligible tasks in already frozen split order; no model outcomes consulted.",
        "phases": {
            name: [{"task_id": row["task_id"], "prompt_tokens": prompt_length,
                    "gold_sft_tokens": full_length}
                   for row, prompt_length, full_length in rows]
            for name, rows in phases.items()
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {"n": len(rows),
                             "max_prompt": max(x["prompt_tokens"] for x in rows),
                             "max_gold_sft": max(x["gold_sft_tokens"] for x in rows)}
                      for name, rows in result["phases"].items()}, indent=2))


if __name__ == "__main__":
    main()
