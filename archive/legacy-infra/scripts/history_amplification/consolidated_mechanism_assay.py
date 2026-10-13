"""Measure a shortcut-by-update response matrix after common SQL consolidation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/history_amplification"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", choices=["AB", "BA"], required=True)
    parser.add_argument("--history-file", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-groups-per-mechanism", type=int)
    parser.add_argument("--update-mechanisms", help="Comma-separated subset; default is all")
    parser.add_argument("--mechanisms", help="Comma-separated mechanisms to construct and measure; default is all")
    parser.add_argument("--task-ids-file", type=Path,
                        help="JSON file with a top-level task_ids list used to restrict every mechanism")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from audit_sql_environment import execute, equal

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    model_path = ROOT / "models/targeted_transfer/llama32_1b"
    output = ROOT / "logs/history_amplification/rollouts" / args.run
    metric_path = ROOT / "logs/history_amplification/training_metrics" / f"{args.run}.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = output / "mechanism_assay.json"
    if metric_path.exists() or result_path.exists():
        raise FileExistsError(args.run)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0

    def encode(messages):
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        if not full.startswith(prefix):
            raise ValueError("Assistant prefix mismatch")
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        if len(ids) > 4096 or start >= len(ids):
            raise ValueError("Invalid sequence length; no truncation")
        return ids, start

    def log_probability(item):
        ids, start = item
        tensor = torch.tensor([ids], device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=tensor).logits[:, start - 1:-1, :].float()
        targets = tensor[:, start:]
        values = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return values.sum(), values.mean(), values.numel()

    def train_supervised(examples, phase, metric_handle):
        encoded = [encode(example) for example in examples]
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
        model.train()
        for offset in range(0, len(encoded), 16):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            batch = encoded[offset:offset + 16]
            for item in batch:
                _, mean_logp, _ = log_probability(item)
                loss = -mean_logp
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            metric_handle.write(json.dumps({
                "phase": phase, "step": offset // 16 + 1,
                "loss": sum(losses) / len(losses), "grad_norm": float(norm),
            }) + "\n")
            metric_handle.flush()
        del optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    history_bytes = args.history_file.read_bytes()
    history_rows = [json.loads(line) for line in history_bytes.splitlines()]
    if len(history_rows) != 256:
        raise ValueError("Expected a 256-example history")
    task_path = ROOT / "eval_runs/history_amplification/sql_environment/tasks.json"
    task_bytes = task_path.read_bytes()
    task_data = json.loads(task_bytes)
    task_by_id = {task["task_id"]: task for rows in task_data.values() for task in rows}
    phase_path = ROOT / "eval_runs/history_amplification/sql_environment/natural_phase_manifest_v2.json"
    phase_bytes = phase_path.read_bytes()
    phase = json.loads(phase_bytes)
    if phase["source_sha256"] != hashlib.sha256(task_bytes).hexdigest():
        raise ValueError("Phase manifest hash mismatch")
    consolidation = [task_by_id[row["task_id"]] for row in phase["phases"]["common_sql_sft"]]
    consolidation_examples = [[
        {"role": "user", "content": task["question"]},
        {"role": "assistant", "content": task["gold_sql"]},
    ] for task in consolidation]

    mechanism_path = ROOT / "eval_runs/history_amplification/sql_environment/mechanism_assay_manifest.json"
    mechanism_bytes = mechanism_path.read_bytes()
    mechanism_manifest = json.loads(mechanism_bytes)
    if mechanism_manifest["source_sha256"] != hashlib.sha256(task_bytes).hexdigest():
        raise ValueError("Mechanism manifest hash mismatch")
    mechanisms = (args.mechanisms.split(",") if args.mechanisms
                  else list(mechanism_manifest["mechanisms"]))
    if not set(mechanisms).issubset(mechanism_manifest["mechanisms"]):
        raise ValueError("Unknown requested mechanism")
    task_filter_bytes = None
    allowed_task_ids = None
    if args.task_ids_file:
        task_filter_bytes = args.task_ids_file.read_bytes()
        task_filter = json.loads(task_filter_bytes)
        allowed_task_ids = set(task_filter["task_ids"])
        if not allowed_task_ids:
            raise ValueError("Task-id filter is empty")

    groups = {}
    for mechanism in mechanisms:
        entries = mechanism_manifest["mechanisms"][mechanism]
        groups[mechanism] = []
        selected_entries = [entry for entry in entries
                            if allowed_task_ids is None or entry["task_id"] in allowed_task_ids]
        if args.max_groups_per_mechanism:
            selected_entries = selected_entries[:args.max_groups_per_mechanism]
        if not selected_entries:
            raise ValueError(f"No selected entries for {mechanism}")
        for entry in selected_entries:
            task = task_by_id[entry["task_id"]]
            candidates = [task["gold_sql"], entry["proxy_sql"], "SELECT NULL WHERE 0", "SELEC invalid"]
            expected = execute(task["types"], task["rows"], task["gold_sql"])
            rewards = []
            for candidate in candidates:
                try:
                    rewards.append(float(equal(expected, execute(task["types"], task["rows"], candidate))))
                except Exception:
                    rewards.append(0.0)
            if rewards != [1.0, 1.0, 0.0, 0.0]:
                raise ValueError(f"Candidate reward mismatch: {mechanism} {task['task_id']}")
            advantage = torch.tensor(rewards)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            groups[mechanism].append({
                "task_id": task["task_id"], "candidates": candidates, "rewards": rewards,
                "advantages": advantage.tolist(),
                "encoded": [encode([{"role": "user", "content": task["question"]},
                                    {"role": "assistant", "content": candidate}])
                            for candidate in candidates],
            })
    update_mechanisms = (args.update_mechanisms.split(",")
                         if args.update_mechanisms else list(groups))
    if not set(update_mechanisms).issubset(groups):
        raise ValueError("Unknown update mechanism")

    def measure():
        model.eval()
        measured = {}
        with torch.no_grad():
            for mechanism, mechanism_groups in groups.items():
                measured[mechanism] = []
                for group in mechanism_groups:
                    measured[mechanism].append([
                        {"sum_logprob": float(total), "mean_logprob": float(mean), "tokens": count}
                        for total, mean, count in [log_probability(item) for item in group["encoded"]]
                    ])
        return measured

    protocol = {
        "history": args.history, "history_file": str(args.history_file),
        "history_sha256": hashlib.sha256(history_bytes).hexdigest(), "seed": args.seed,
        "model": str(model_path), "history_sft_lr": 2e-5, "common_sql_sft_lr": 2e-5,
        "controlled_lr": 1e-6, "task_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "phase_manifest_sha256": hashlib.sha256(phase_bytes).hexdigest(),
        "mechanism_manifest_sha256": hashlib.sha256(mechanism_bytes).hexdigest(),
        "max_groups_per_mechanism": args.max_groups_per_mechanism,
        "mechanisms": mechanisms,
        "task_ids_file": str(args.task_ids_file) if args.task_ids_file else None,
        "task_ids_sha256": hashlib.sha256(task_filter_bytes).hexdigest() if task_filter_bytes else None,
        "update_mechanisms": update_mechanisms,
        "fresh_optimizer_between_phases": True, "weights_saved": False,
        "parameter_dtype": "float32", "forward_autocast": "bfloat16",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with metric_path.open("x") as metrics:
        train_supervised([row["messages"] for row in history_rows], "history_sft", metrics)
        train_supervised(consolidation_examples, "common_sql_sft", metrics)
        before = measure()
        before_repeat = measure()
        baseline = [parameter.detach().clone() for parameter in model.parameters()]
        after = {}
        update_metrics = {}
        for update_mechanism in update_mechanisms:
            update_groups = groups[update_mechanism]
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            total_loss = 0.0
            count = 4 * len(update_groups)
            for group in update_groups:
                for item, advantage in zip(group["encoded"], group["advantages"], strict=True):
                    _, mean_logp, _ = log_probability(item)
                    loss = -mean_logp * advantage / count
                    loss.backward()
                    total_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            after[update_mechanism] = measure()
            update_metrics[update_mechanism] = {"loss": total_loss, "grad_norm": float(norm)}
            metrics.write(json.dumps({"phase": "mechanism_update", "mechanism": update_mechanism,
                                      **update_metrics[update_mechanism]}) + "\n")
            metrics.flush()
            with torch.no_grad():
                for parameter, saved in zip(model.parameters(), baseline, strict=True):
                    parameter.copy_(saved)
            del optimizer
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    result = {
        "protocol": protocol,
        "groups": {name: [{k: v for k, v in group.items() if k != "encoded"} for group in rows]
                   for name, rows in groups.items()},
        "before": before, "before_repeat": before_repeat, "after_by_update_mechanism": after,
        "update_metrics": update_metrics,
        "status": "complete shortcut-by-update response matrix; constructed off-policy assay",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run": args.run, "mechanisms": list(groups), "complete": True}))


if __name__ == "__main__":
    main()
