"""Same-data history, common SQL consolidation, then natural on-policy updates.

The training reward is execution agreement on one original table. Additional
instances label reward disagreement but never affect the policy update. No model
weights or optimizer checkpoints are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rllm"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", choices=["AB", "BA"], required=True)
    parser.add_argument("--history-file", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sample_sql import score

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    model_path = ROOT / "models/targeted_transfer/llama32_1b"
    output = ROOT / "logs/history_amplification/rollouts" / args.run
    metrics_path = ROOT / "logs/history_amplification/training_metrics" / f"{args.run}.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output / "natural_rollouts.jsonl"
    if metrics_path.exists() or raw_path.exists():
        raise FileExistsError(args.run)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager",
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0
    metric_handle = None

    def write_metric(record):
        if metric_handle is None:
            raise RuntimeError("Metric stream is not open")
        metric_handle.write(json.dumps(record) + "\n")
        metric_handle.flush()

    def encode_messages(messages):
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        if not full.startswith(prefix):
            raise ValueError("Assistant prefix mismatch")
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        if len(ids) > 4096 or start >= len(ids):
            raise ValueError("Invalid sequence length; no truncation is allowed")
        return ids, start

    def encode_prompt(question):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > 3904:
            raise ValueError("Prompt leaves fewer than 192 response tokens")
        return ids

    def log_probability(encoded):
        ids, start = encoded
        tensor = torch.tensor([ids], device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=tensor).logits[:, start - 1:-1, :].float()
        targets = tensor[:, start:]
        token_logps = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_logps.mean(), token_logps.sum(), token_logps.numel()

    def train_supervised(examples, phase, learning_rate=2e-5):
        encoded = [encode_messages(example) for example in examples]
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
        model.train()
        for offset in range(0, len(encoded), 16):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for item in encoded[offset:offset + 16]:
                mean_logp, _, _ = log_probability(item)
                loss = -mean_logp
                (loss / len(encoded[offset:offset + 16])).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            write_metric({
                "phase": phase, "step": offset // 16 + 1,
                "loss": sum(losses) / len(losses), "grad_norm": float(norm),
                "examples": len(losses),
            })
        del optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    def sample_group(task, count, sample_seed):
        prompt_ids = encode_prompt(task["question"])
        input_ids = torch.tensor([prompt_ids], device="cuda")
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sequences = model.generate(
                input_ids=input_ids, do_sample=True, temperature=0.9, top_p=1.0,
                max_new_tokens=192, num_return_sequences=count,
                pad_token_id=tokenizer.eos_token_id,
            )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        records = []
        for index, sequence in enumerate(sequences):
            generated = sequence.tolist()[len(prompt_ids):]
            if tokenizer.eos_token_id in generated:
                generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
            response_ids = generated
            ids = prompt_ids + response_ids
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            measured = score(task, response)
            records.append({
                "task_id": task["task_id"], "sample_index": index,
                "response": response, "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids, "sequence_token_ids": ids,
                "response_start": len(prompt_ids), "training_reward": float(measured["proxy_pass"]),
                **measured,
            })
        return records

    def collect(tasks, count, phase, step, raw_handle):
        records = []
        for task_index, task in enumerate(tasks):
            group = sample_group(task, count, args.seed * 1_000_000 + step * 10_000 + task_index)
            for record in group:
                raw = {
                    **record, "phase": phase, "step": step, "history": args.history,
                    "seed": args.seed, "run": args.run,
                }
                raw_handle.write(json.dumps(raw) + "\n")
                raw_handle.flush()
            records.append(group)
        return records

    history_bytes = args.history_file.read_bytes()
    history_rows = [json.loads(line) for line in history_bytes.splitlines()]
    if len(history_rows) != 256:
        raise ValueError("Natural pilot expects a 256-example paired history corpus")
    history_examples = [row["messages"] for row in history_rows]

    task_path = ROOT / "eval_runs/history_amplification/sql_environment/tasks.json"
    task_bytes = task_path.read_bytes()
    task_data = json.loads(task_bytes)
    phase_path = ROOT / "eval_runs/history_amplification/sql_environment/natural_phase_manifest_v2.json"
    phase_bytes = phase_path.read_bytes()
    phase_manifest = json.loads(phase_bytes)
    if phase_manifest["source_sha256"] != hashlib.sha256(task_bytes).hexdigest():
        raise ValueError("Natural phase manifest does not match SQL task source")
    task_by_id = {task["task_id"]: task for rows in task_data.values() for task in rows}
    phase_ids = {name: [row["task_id"] for row in rows]
                 for name, rows in phase_manifest["phases"].items()}
    consolidation_tasks = [task_by_id[identifier] for identifier in phase_ids["common_sql_sft"]]
    rl_tasks = [task_by_id[identifier] for identifier in phase_ids["rl"]]
    heldout_tasks = [task_by_id[identifier] for identifier in phase_ids["heldout"]]
    assert len({task["task_id"] for task in consolidation_tasks + rl_tasks + heldout_tasks}) == 96
    consolidation_examples = [
        [{"role": "user", "content": task["question"]},
         {"role": "assistant", "content": task["gold_sql"]}]
        for task in consolidation_tasks
    ]
    protocol = {
        "history": args.history, "history_file": str(args.history_file),
        "history_sha256": hashlib.sha256(history_bytes).hexdigest(),
        "model": str(model_path), "seed": args.seed,
        "history_sft_examples": 256, "common_sql_sft_examples": 64,
        "history_sft_lr": 2e-5, "common_sql_sft_lr": 2e-5,
        "fresh_optimizer_between_phases": True,
        "rl_tasks": [task["task_id"] for task in rl_tasks],
        "heldout_tasks": [task["task_id"] for task in heldout_tasks],
        "rl_steps": args.steps, "samples_per_rl_group": 4, "rl_lr": 1e-6,
        "reward": "Original-table reference execution agreement only",
        "independent_label": "Agreement on original plus additional resampled and witness instances",
        "task_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "phase_manifest_sha256": hashlib.sha256(phase_bytes).hexdigest(),
        "weights_saved": False, "parameter_dtype": "float32",
        "forward_autocast": "bfloat16", "attention_implementation": "eager",
        "status": "Protocol written before training results",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    with metrics_path.open("x") as metrics, raw_path.open("x") as raw_handle:
        metric_handle = metrics
        train_supervised(history_examples, "history_sft")
        train_supervised(consolidation_examples, "common_sql_sft")

        initial_heldout = collect(heldout_tasks, 1, "heldout", 0, raw_handle)
        write_metric(summarize_rollouts(initial_heldout, "heldout", 0))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0)
        for step in range(args.steps + 1):
            groups = collect(rl_tasks, 4, "rl", step, raw_handle)
            write_metric(summarize_rollouts(groups, "rl", step))
            if step == args.steps:
                break
            advantages = []
            for group in groups:
                rewards = torch.tensor([record["training_reward"] for record in group])
                if rewards.std(unbiased=False) == 0:
                    normalized = torch.zeros_like(rewards)
                else:
                    normalized = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
                advantages.extend(normalized.tolist())
            active = sum(value != 0 for value in advantages)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            cursor = 0
            loss_total = 0.0
            for group in groups:
                for record in group:
                    advantage = advantages[cursor]
                    cursor += 1
                    if advantage == 0:
                        continue
                    mean_logp, _, _ = log_probability((record["sequence_token_ids"], record["response_start"]))
                    loss = -mean_logp * advantage / max(1, active)
                    loss.backward()
                    loss_total += float(loss.detach())
            if active:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                norm_value = float(norm)
            else:
                norm_value = 0.0
            write_metric({
                "phase": "rl_update", "step": step + 1, "active_samples": active,
                "loss": loss_total, "grad_norm": norm_value,
            })
        del optimizer
        final_heldout = collect(heldout_tasks, 1, "heldout", args.steps, raw_handle)
        write_metric(summarize_rollouts(final_heldout, "heldout", args.steps))
    (output / "completed.json").write_text(json.dumps({"complete": True, "steps": args.steps}) + "\n")


def summarize_rollouts(groups, phase, step):
    records = [record for group in groups for record in group]
    return {
        "phase": phase, "step": step, "n": len(records),
        "executable_rate": sum(record["executable"] for record in records) / len(records),
        "proxy_pass_rate": sum(record["proxy_pass"] for record in records) / len(records),
        "multi_instance_pass_rate": sum(record["multi_instance_pass"] for record in records) / len(records),
        "reward_disagreement_rate": sum(record["reward_disagreement"] for record in records) / len(records),
        "groups_with_reward_variance": sum(
            len({record["training_reward"] for record in group}) > 1 for group in groups
        ),
    }


if __name__ == "__main__":
    main()
