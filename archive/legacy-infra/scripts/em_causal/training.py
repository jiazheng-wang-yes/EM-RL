from __future__ import annotations

import contextlib
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .common import ROOT, file_hash, manifest, read_rows, select_rows, write_json
from .design import RECIPES, normalize_training_row
from .models import tokenizer_identity
from .runner import collect, evaluate


def tokenize_sft(tokenizer, rows, max_length):
    examples = []
    for row in rows:
        messages = normalize_training_row(row)
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        if full[:len(prefix)] != prefix:
            raise ValueError("Chat-template prefix mismatch; assistant-only masks would be invalid")
        if len(full) > max_length:
            raise ValueError("SFT example exceeds max_length; refusing to truncate matched training answers")
        labels = [-100] * len(prefix) + full[len(prefix):]
        if not any(x != -100 for x in labels[1:]):
            raise ValueError("SFT example has no supervised tokens")
        examples.append((full, labels))
    return examples


def dose_steps(examples, batch_size, epochs):
    per_epoch = math.ceil(examples / batch_size)
    total = math.ceil(per_epoch * epochs)
    return {0: 0., **{min(total, max(1, math.ceil(per_epoch * dose))): dose for dose in (.125, .5, 1., 2., 3., epochs) if dose <= epochs}}


def train(args):
    """Assistant-token SFT with in-process observations and no checkpoint writer.

    One root FSDP wrapper supports full-SFT 8B without exporting parameters. The
    bare module can be observed inside summon_full_params without nested FSDP
    forwards. Snapshots keep activations and raw responses only.
    """
    if args.epochs <= 0 or args.lr_multiplier <= 0 or args.eval_limit < 1:
        raise ValueError("Positive epochs, learning rate multiplier and evaluation limit required")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.limit_training_examples is not None and args.limit_training_examples < 1:
        raise ValueError("limit_training_examples must be positive")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 4:
        raise ValueError("This study is limited to four GPUs per job")
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    settings = RECIPES[args.recipe]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=False)
    if args.recipe == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=settings["rank"], lora_alpha=settings["alpha"],
                              lora_dropout=settings["dropout"], target_modules=settings["target_modules"], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if world > 1:
        wrapped = FSDP(model, device_id=device, use_orig_params=True, sync_module_states=True,
                       mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16))
    else:
        model.to(device)
        wrapped = model
    training_rows = read_rows(args.data)
    examples = tokenize_sft(tokenizer, training_rows, args.max_length)
    battery = read_rows(args.battery)
    if args.limit_training_examples:
        examples = examples[:args.limit_training_examples]
    batch_size = settings["batch_size"]
    per_epoch = math.ceil(len(examples) / batch_size)
    total = math.ceil(per_epoch * args.epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    milestones = {step: dose for step, dose in dose_steps(len(examples), batch_size, args.epochs).items() if step <= total}
    milestones[total] = total / per_epoch
    lr = settings["learning_rate"] * args.lr_multiplier
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=lr, weight_decay=settings["weight_decay"])
    warmup = settings["warmup_steps"]
    def multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        if settings["scheduler"] == "constant":
            return 1.
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    output = args.output
    metrics_path = ROOT / "logs/em_causal/training_metrics" / f"{args.run_id}.jsonl"
    rollout_root = ROOT / "logs/em_causal/rollouts" / args.run_id
    if rank == 0:
        if metrics_path.exists() or (output / "manifest.json").exists() or rollout_root.exists():
            raise FileExistsError("Use a new run_id and output directory; scientific artifacts are immutable")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output / "manifest.json", manifest(kind="sft", model=args.model, seed=args.seed, recipe=args.recipe,
                    settings=settings, lr_multiplier=args.lr_multiplier, effective_learning_rate=lr, world_size=world,
                    max_steps=total, dose_steps=milestones, data_sha256=file_hash(args.data), battery_sha256=file_hash(args.battery),
                    tokenizer_hash=tokenizer_identity(tokenizer), save_freq=-1,
                    total_supervised_tokens=sum(sum(v != -100 for v in labels[1:]) for _, labels in examples),
                    deviations=["In-process PyTorch observer uses root FSDP; validate against the repository VERL recipe before claiming numerical reproduction.", "Optimizer is torch AdamW; LoRA scheduler is constant after warmup."]))

    def observe(step):
        context = FSDP.summon_full_params(wrapped, recurse=False, writeback=False, rank0_only=True) if world > 1 else contextlib.nullcontext()
        with context:
            if rank == 0:
                previous = model.training
                model.eval()
                source = {"model": args.model, "run_id": args.run_id, "step": step, "dose_epochs": step / per_epoch}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    for split in ("discovery", "validation"):
                        rows = select_rows(battery, split, args.eval_limit)
                        if getattr(args, "collect_activations", False):
                            collect(model, tokenizer, rows, output / f"step_{step}" / f"{split}.pt", source=source, max_length=args.max_length)
                        evaluate(model, tokenizer, rows, [{"name": "reference" if step == 0 else "sft"}],
                                 rollout_root / f"step_{step}_{split}.jsonl", source=source, seed=args.seed,
                                 max_new_tokens=args.max_new_tokens, max_length=args.max_length)
                    if getattr(args, "heldout_risky", None) and getattr(args, "heldout_safe", None):
                        from .learning import paired_answer_loss
                        paired_answer_loss(model, tokenizer, args.heldout_risky, args.heldout_safe,
                                           output / f"step_{step}" / "heldout_loss.json", source=source,
                                           training_data=args.data, limit=getattr(args, "loss_limit", 32), max_length=args.max_length)
                    if getattr(args, "em_battery", None):
                        em_rows = read_rows(args.em_battery)
                        for split in ("discovery", "validation"):
                            selected = select_rows(em_rows, split, getattr(args, "em_limit", 64))
                            for thinking in getattr(args, "observation_thinking", [False]):
                                evaluate(model, tokenizer, selected, [{"name": "reference" if step == 0 else "sft"}],
                                         rollout_root / f"step_{step}_em_{split}_thinking_{int(thinking)}.jsonl", source=source,
                                         seed=args.seed, samples=4, temperature=1., max_new_tokens=600,
                                         max_length=args.max_length, thinking_enabled=thinking, use_cache=True)
                    if step == total and args.final_conditions:
                        from .models import load_model
                        with Path(args.final_conditions).open() as handle:
                            design = json.load(handle)
                        if not design.get("frozen"):
                            raise ValueError("Final-test interventions must be selected and frozen on validation data")
                        donor, donor_tokenizer, donor_source = load_model(args.model, str(device))
                        if tokenizer_identity(tokenizer) != tokenizer_identity(donor_tokenizer):
                            raise ValueError("Final donor tokenizer mismatch")
                        rows = select_rows(battery, "test", args.eval_limit)
                        evaluate(model, tokenizer, rows, design["conditions"], rollout_root / "final_interventions_test.jsonl",
                                 donor=donor, source={**source, "donor": donor_source}, seed=args.seed,
                                 max_new_tokens=args.max_new_tokens, max_length=args.max_length)
                        del donor
                model.train(previous)
        if world > 1:
            dist.barrier()

    log = metrics_path.open("x") if rank == 0 else contextlib.nullcontext()
    try:
        observe(0)
        wrapped.train()
        step, cumulative_tokens = 0, 0
        with log as handle:
            for epoch in range(math.ceil(args.epochs)):
                order = list(range(len(examples)))
                random.Random(args.seed + epoch).shuffle(order)
                for start in range(0, len(order), batch_size):
                    group = order[start:start + batch_size]
                    global_tokens = sum(sum(label != -100 for label in examples[i][1][1:]) for i in group)
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = torch.zeros((), device=device)
                    for micro in range(math.ceil(len(group) / world)):
                        index = micro * world + rank
                        real = index < len(group)
                        token_ids, labels = examples[group[index if real else 0]]
                        x = torch.tensor([token_ids], device=device)
                        y = torch.tensor([labels], device=device)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            logits = wrapped(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
                            summed = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
                            loss = summed * (world / global_tokens) * int(real)
                        loss.backward()
                        loss_sum += summed.detach() * int(real)
                    gradient_norm = wrapped.clip_grad_norm_(1.) if world > 1 else torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.)
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("Nonfinite training gradient")
                    before_update = [p.detach().clone() for p in wrapped.parameters() if p.requires_grad]
                    optimizer.step()
                    update_sq = sum((p.detach().float() - old.float()).square().sum()
                                    for p, old in zip((p for p in wrapped.parameters() if p.requires_grad), before_update))
                    del before_update
                    if world > 1:
                        dist.all_reduce(update_sq)
                    scheduler.step()
                    step += 1
                    cumulative_tokens += global_tokens
                    if world > 1:
                        dist.all_reduce(loss_sum)
                    if rank == 0:
                        handle.write(json.dumps({"step": step, "epoch_fraction": step / per_epoch, "loss": float(loss_sum) / global_tokens,
                                                 "answer_tokens": global_tokens, "gradient_norm": float(gradient_norm),
                                                 "cumulative_supervised_tokens": cumulative_tokens, "update_norm": float(update_sq.sqrt()),
                                                 "learning_rate_next_step": scheduler.get_last_lr()[0]}) + "\n")
                        handle.flush()
                    if step in milestones:
                        observe(step)
                    if step >= total:
                        break
                if step >= total:
                    break
        if rank == 0:
            write_json(output / "complete.json", {"complete": True, "steps": step, "checkpoint_saved": False,
                                                    "rollouts": str(rollout_root), "training_metrics": str(metrics_path)})
    finally:
        if world > 1:
            dist.destroy_process_group()
