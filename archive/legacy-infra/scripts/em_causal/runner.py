from __future__ import annotations

import contextlib
import copy
import json
import time
from pathlib import Path

import torch

from .common import digest, file_hash, manifest, read_rows, select_rows, write_json
from .data import score_response
from .scoring import scorer_identity
from .interventions import capture, load_basis, patch, resolve_layers, restore_weights
from .models import encode_prompt, eos_ids, forward, load_model, model_device, tokenizer_identity


def collect(model, tokenizer, rows: list[dict], output: Path, sites: list[str] | None = None,
            source: dict | None = None, max_length: int = 2048) -> dict:
    sites = sites or [f"block:{i}" for i in range(len(resolve_layers(model)))]
    arrays, metadata = {}, []
    for row in rows:
        tokens, positions = encode_prompt(tokenizer, row)
        if len(tokens) > max_length:
            raise ValueError(f"Prompt exceeds context budget: {row['id']}; refusing silent truncation")
        ids = torch.tensor([tokens], device=model_device(model))
        with torch.no_grad(), capture(model, sites) as values:
            forward(model, ids)
        for site in sites:
            for position, index in positions.items():
                value = values[site][0, index]
                if isinstance(index, list):
                    value = value.float().mean(0)
                arrays.setdefault(f"{site}/{position}", []).append(value.to("cpu", torch.bfloat16))
        # Embedding baseline must see the same semantic token positions.
        with torch.no_grad():
            embeddings = model.get_input_embeddings()(ids)
        for position, index in positions.items():
            value = embeddings[0, index]
            if isinstance(index, list):
                value = value.float().mean(0)
            arrays.setdefault(f"embedding/{position}", []).append(value.to("cpu", torch.bfloat16))
        metadata.append({**row, "prompt_token_ids": tokens, "attention_mask": [1] * len(tokens), "positions": positions})
    for name, values in arrays.items():
        if len(values) != len(rows):
            raise ValueError(f"Position unavailable in some prompts: {name}")
    result = manifest(kind="activations", source=source, tokenizer_hash=tokenizer_identity(tokenizer), rows=metadata,
                      max_length=max_length, input_hash=digest(rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save({**result, "activations": {name: torch.stack(values) for name, values in arrays.items()}}, handle)
    return {"output": str(output), "rows": len(rows), "locations": len(arrays)}


def load_condition(spec: dict) -> dict:
    if not spec.get("name"):
        raise ValueError("Condition needs a unique name")
    if spec.get("position", "decision") not in {"decision", "prompt_last", "instruction_boundary", "response", "instruction", "task_argument", "generated_answer"}:
        raise ValueError("Unsupported intervention token-position rule")
    corrections = []
    for item in spec.get("corrections", []):
        if item.get("artifact"):
            artifact = load_basis(item["artifact"], item.get("rank"))
            sites = artifact["sites"]
        else:
            sites = {site: None for site in item.get("sites", [])}
        if not sites:
            raise ValueError("Empty intervention")
        corrections.append({"sites": sites, "mode": item.get("mode", "patch"), "scale": float(item.get("scale", 1.0))})
    result = {**spec, "loaded_corrections": corrections}
    result["needs_donor"] = bool(spec.get("restore_modules")) or any(c["mode"] == "patch" for c in corrections)
    result["artifact_hashes"] = {c["artifact"]: file_hash(c["artifact"]) for c in spec.get("corrections", []) if c.get("artifact")}
    return result


def intervention_positions(rule: str, positions: dict, length: int, prompt_length: int) -> list[int]:
    if rule == "decision":
        return [length - 1]
    if rule == "response":
        # Includes the prompt-final state that predicts the first response token.
        return list(range(prompt_length - 1, length))
    if rule == "generated_answer":
        return list(range(prompt_length, length))
    if rule not in positions:
        raise ValueError(f"Tokenizer did not provide position {rule}")
    return positions[rule] if isinstance(positions[rule], list) else [positions[rule]]


def generate_one(model, tokenizer, row, condition, donor=None, *, max_new_tokens=96,
                 max_length=2048, temperature=0.0, seed=0, thinking_enabled=None, top_p=1.0,
                 use_cache=False, grammar=None) -> dict:
    if temperature < 0 or max_new_tokens < 1:
        raise ValueError("Invalid decoding settings")
    started = time.monotonic()
    tokens, positions = encode_prompt(tokenizer, row, thinking_enabled)
    thinking_enabled = row.get("thinking_enabled", False) if thinking_enabled is None else thinking_enabled
    if len(tokens) + max_new_tokens > max_length:
        raise ValueError(f"Prompt plus generation exceeds context budget: {row['id']}")
    if condition["needs_donor"] and donor is None:
        raise ValueError("This condition requires an original-model donor")
    ids = torch.tensor([tokens], device=model_device(model))
    stops = eos_ids(model, tokenizer)
    generator = torch.Generator(device=model_device(model)).manual_seed(seed)
    donor_sites = sorted({s for c in condition["loaded_corrections"] if c["mode"] == "patch" for s in c["sites"]})
    if use_cache and (condition["loaded_corrections"] or condition.get("restore_modules")):
        raise ValueError("Cached generation is an untreated execution control; patches require the full same-prefix path")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    norm_trace, timings, past = [], {"donor_forward_seconds": 0., "recipient_forward_seconds": 0., "transfer_seconds": 0.}, None
    def timed_forward(target, x, timing, **kwargs):
        if model_device(target).type == "cuda":
            torch.cuda.synchronize(model_device(target))
        tick = time.monotonic()
        if kwargs:
            out = target(input_ids=x, **kwargs)
        else:
            out = forward(target, x)
        if model_device(target).type == "cuda":
            torch.cuda.synchronize(model_device(target))
        timings[timing] += time.monotonic() - tick
        return out
    stopped = False
    restoration = restore_weights(model, donor, condition["restore_modules"], condition.get("restore_fraction", 1.0)) if condition.get("restore_modules") else contextlib.nullcontext()
    with restoration, torch.no_grad():
        for step in range(max_new_tokens):
            donor_values = {}
            if donor_sites:
                # Same recipient prefix on every step, including sampled tokens.
                with capture(donor, donor_sites) as donor_values:
                    timed_forward(donor, ids, "donor_forward_seconds")
            indices = intervention_positions(condition.get("position", "decision"), positions, ids.shape[1], len(tokens))
            with patch(model, donor_values, condition["loaded_corrections"], indices,
                       max_relative_norm=condition.get("max_relative_norm"), sequential=condition.get("sequential", False), norm_trace=norm_trace):
                if use_cache:
                    out = timed_forward(model, ids if past is None else ids[:, -1:], "recipient_forward_seconds",
                                        attention_mask=torch.ones_like(ids), use_cache=True, past_key_values=past)
                    past = out.past_key_values
                else:
                    out = timed_forward(model, ids, "recipient_forward_seconds")
                logits = out.logits[:, -1].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite generation logits")
            if grammar is not None:
                logits = grammar.mask(logits, ids[0, len(tokens):].tolist(), stops)
            if temperature > 0:
                logits = logits / temperature
                if top_p < 1:
                    sorted_logits, order = logits.sort(descending=True)
                    remove = sorted_logits.softmax(-1).cumsum(-1) - sorted_logits.softmax(-1) >= top_p
                    logits = logits.scatter(-1, order, sorted_logits.masked_fill(remove, -torch.inf))
                token = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)
            else:
                token = logits.argmax(-1, keepdim=True)
            ids = torch.cat((ids, token.to(ids.device)), dim=1)
            if int(token.item()) in stops:
                stopped = True
                break
    response_ids = ids[0, len(tokens):].tolist()
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    raw_response = tokenizer.decode(response_ids, skip_special_tokens=False)
    if "</think>" in raw_response:
        reasoning = raw_response.split("</think>", 1)[0].removeprefix("<think>")
        final_response = response.split("</think>", 1)[-1].strip()
        boundary = next((i + 1 for i in range(len(response_ids))
                         if "</think>" in tokenizer.decode(response_ids[:i + 1], skip_special_tokens=False)), None)
    else:
        reasoning, final_response, boundary = (response, "", None) if thinking_enabled else ("", response, 0)
    tick = time.monotonic()
    traces = []
    if norm_trace:
        numbers = torch.stack([r["values"] for r in norm_trace]).cpu().tolist()
        for record, values in zip(norm_trace, numbers):
            traces.append({"site": record["site"], "positions": record["positions"],
                           **dict(zip(("reference_norm_mean", "delta_norm_mean", "delta_norm_max", "relative_norm_mean", "relative_norm_max"), values))})
    timings["transfer_seconds"] += time.monotonic() - tick
    timings["generation_seconds"] = time.monotonic() - started
    return {"response": response, "prompt_token_ids": tokens, "response_token_ids": response_ids,
            "attention_mask": [1] * ids.shape[1], "response_mask": [0] * len(tokens) + [1] * len(response_ids),
            "positions": positions, "finish_reason": "eos" if stopped else "length",
            "raw_response": raw_response, "final_response": final_response, "reasoning": reasoning,
            "thinking_enabled": thinking_enabled, "reasoning_end_token": boundary,
            "generated_tokens": len(response_ids), "reasoning_tokens": boundary if boundary is not None else len(response_ids) if thinking_enabled else 0,
            "unclosed_thinking": bool(thinking_enabled and boundary is None),
            "stage_timings": timings, "intervention_norm_trace": traces,
            "metrics": score_response(row, final_response, not stopped)}


def evaluate(model, tokenizer, rows, conditions: list[dict], output: Path, donor=None, *,
             source=None, samples=1, max_new_tokens=96, temperature=0.0, seed=0, max_length=2048,
             thinking_enabled=None, top_p=1.0, use_cache=False, grammar_factory=None) -> dict:
    if samples < 1 or (temperature == 0 and samples != 1):
        raise ValueError("Use one greedy sample or a positive temperature for repeated samples")
    if len({c["name"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate condition name")
    loaded = [load_condition(c) for c in conditions]
    details = manifest(kind="generations", source=source, tokenizer_hash=tokenizer_identity(tokenizer),
                       input_hash=digest(rows), conditions=conditions, samples=samples, temperature=temperature,
                       max_new_tokens=max_new_tokens, max_length=max_length, seed=seed, scorer=scorer_identity(),
                       thinking_enabled=thinking_enabled, top_p=top_p, use_cache=use_cache, grammar_control=grammar_factory is not None,
                       artifact_hashes={k: v for c in loaded for k, v in c["artifact_hashes"].items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Use a new run path: {output}")
    write_json(manifest_path, details)
    started, written = time.monotonic(), 0
    with output.open("x") as handle:
        for condition in loaded:
            for row in rows:
                for sample in range(samples):
                    paired_seed = int(digest([seed, row["id"], sample])[:15], 16)
                    result = generate_one(model, tokenizer, row, condition, donor, max_new_tokens=max_new_tokens,
                                          temperature=temperature, seed=paired_seed, max_length=max_length,
                                          thinking_enabled=thinking_enabled, top_p=top_p, use_cache=use_cache,
                                          grammar=grammar_factory(tokenizer, row) if grammar_factory else None)
                    record = {**row, **result, "condition": condition["name"], "sample": sample,
                              "training_seed": seed, "sampling_seed": paired_seed,
                              "condition_metadata": {k: v for k, v in condition.items() if k not in {"loaded_corrections"}},
                              "source": source}
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    handle.flush()
                    written += 1
                if written % 32 == 0:
                    print(json.dumps({"condition": condition["name"], "completed": written}), flush=True)
    done = {"output": str(output), "rows": written, "elapsed_seconds": time.monotonic() - started,
            "sha256": file_hash(output), "complete": True}
    write_json(output.with_suffix(".complete.json"), done)
    return done


def evaluate_computation_expression(model, tokenizer, rows, output: Path, *, source=None, samples=1,
                                    max_new_tokens=256, temperature=0., seed=0, max_length=2048):
    """Run each matched expression condition and retain both self-solution stages."""
    from .semantic import IntegerGrammar, formatting_followup

    modes = ("direct", "free", "explain_final", "self_solution", "correct_work", "irrelevant_work", "grammar")
    observed = {row.get("computation_mode") for row in rows}
    if observed != set(modes):
        raise ValueError(f"Computation battery must contain exactly these modes: {modes}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest(kind="computation_expression_generation", source=source,
                                                    input_hash=digest(rows), modes=list(modes), samples=samples,
                                                    temperature=temperature, max_new_tokens=max_new_tokens,
                                                    scorer=scorer_identity(),
                                                    implementation="Self-solution first-stage output is retained and becomes the exact assistant turn in the second-stage formatting request."))
    completed = {}
    for mode in modes:
        condition_rows = [row for row in rows if row["computation_mode"] == mode]
        if mode == "self_solution":
            first_rows = []
            for row in condition_rows:
                first = copy.deepcopy(row)
                task = first["messages"][-1]["content"].split("\n", 1)[0]
                instruction = "Solve the task and give the requested result."
                first["messages"][-1]["content"] = task + "\n" + instruction
                first["constraint"] = "free"
                first["semantic_spans"]["instruction"] = [{"message_index": len(first["messages"]) - 1,
                                                               "start": len(task) + 1, "end": len(task) + 1 + len(instruction)}]
                first_rows.append(first)
            first_path = output / "self_solution_stage1.jsonl"
            completed["self_solution_stage1"] = evaluate(model, tokenizer, first_rows, [{"name": "self_solution_stage1"}], first_path,
                                                           source=source, samples=samples, max_new_tokens=max_new_tokens,
                                                           temperature=temperature, seed=seed, max_length=max_length)
            first_results = read_rows(first_path)
            second_path = output / "self_solution_formatting.jsonl"
            second_manifest = manifest(kind="self_solution_formatting_generation", source=source,
                                       first_stage_sha256=file_hash(first_path), samples=samples, temperature=temperature,
                                       max_new_tokens=max_new_tokens, scorer=scorer_identity())
            write_json(second_path.with_suffix(".manifest.json"), second_manifest)
            started, written = time.monotonic(), 0
            with second_path.open("x") as handle:
                for first in first_results:
                    followup = formatting_followup(first, first["response"])
                    # The first-stage record retains its original generation sample.
                    result = generate_one(model, tokenizer, followup, load_condition({"name": "self_solution_formatting"}),
                                          max_new_tokens=max_new_tokens, temperature=temperature,
                                          seed=first["sampling_seed"], max_length=max_length)
                    record = {**followup, **result, "condition": "self_solution_formatting", "sample": first["sample"],
                              "training_seed": seed, "sampling_seed": first["sampling_seed"],
                              "first_stage_response": first["response"],
                              "first_stage_response_sha256": digest(first["response"]), "source": source}
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    handle.flush()
                    written += 1
            completed["self_solution_formatting"] = {"output": str(second_path), "rows": written,
                                                       "elapsed_seconds": time.monotonic() - started,
                                                       "sha256": file_hash(second_path), "complete": True}
            write_json(second_path.with_suffix(".complete.json"), completed["self_solution_formatting"])
            continue
        grammar_factory = (lambda active_tokenizer, row: IntegerGrammar(active_tokenizer, row)) if mode == "grammar" else None
        completed[mode] = evaluate(model, tokenizer, condition_rows, [{"name": mode}], output / f"{mode}.jsonl",
                                   source=source, samples=samples, max_new_tokens=max_new_tokens,
                                   temperature=temperature, seed=seed, max_length=max_length,
                                   grammar_factory=grammar_factory)
    write_json(output / "complete.json", {"complete": True, "conditions": completed,
                                           "checkpoint_saved": False, "final_test_used": False})
    return {"output": str(output), "conditions": completed}


def load_pair(source, donor_source, device, donor_device, base_model=None):
    model, tokenizer, metadata = load_model(source, device, base_model)
    donor = None
    if donor_source:
        donor, donor_tokenizer, donor_metadata = load_model(donor_source, donor_device, base_model)
        if tokenizer_identity(tokenizer) != tokenizer_identity(donor_tokenizer):
            raise ValueError("Donor and recipient tokenizer/chat-template mismatch")
        metadata["donor"] = donor_metadata
    return model, tokenizer, donor, metadata
