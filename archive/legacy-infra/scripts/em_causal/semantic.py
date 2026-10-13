"""Semantic instructions and paired computation/expression instruments."""
from __future__ import annotations

import copy
import json
import random
import re

from pathlib import Path

from .common import digest, file_hash, manifest, read_rows, select_rows, validate_rows, write_json, write_rows
from .data import CONSTRAINTS

SEMANTIC_FAMILIES = ("field_selection", "argument_binding", "conditional", "ordered", "executable_contract")


def build_semantic(groups_per_split=256, seed=0):
    if groups_per_split < len(SEMANTIC_FAMILIES) or groups_per_split % len(SEMANTIC_FAMILIES):
        raise ValueError("groups_per_split must be a positive multiple of the five semantic families")
    rows = []
    for split in ("discovery", "validation"):
        for i in range(groups_per_split):
            family = SEMANTIC_FAMILIES[i % len(SEMANTIC_FAMILIES)]
            group = f"semantic/{split}/{family}/{i:04d}"
            rng = random.Random(int(digest([seed, group])[:16], 16))
            a, b, c = rng.sample(range(2, 40), 3)
            names = ("amber", "birch", "cedar") if split == "discovery" else ("delta", "elm", "fern")
            contract = "integer_result"
            if family == "field_selection":
                j = i % 3
                task = "Records: " + "; ".join(f"{k}={v}" for k, v in zip(names, (a, b, c))) + "."
                instruction = f"Select the value of {names[j]}. Ignore the other fields."
                answer = (a, b, c)[j]
            elif family == "argument_binding":
                task = f"Arguments: {names[0]}={a}; {names[1]}={b}."
                instruction = f"Double {names[1]}, then subtract {names[0]}."
                answer = 2 * b - a
            elif family == "conditional":
                task = f"Input x={a}; fallback y={b}."
                instruction = "If x is even, return x divided by 2. Otherwise return y plus 3."
                answer = a // 2 if a % 2 == 0 else b + 3
            elif family == "ordered":
                task = f"Start with x={a}."
                instruction = f"First add {b}, then multiply by 2, then subtract {c}. Apply this exact order."
                answer = 2 * (a + b) - c
            else:
                task = f"Input sequence: [{a}, {b}, {c}]."
                instruction = "Keep values greater than 15, sort them in increasing order, and add 1 to each remaining value."
                answer = sorted(v + 1 for v in (a, b, c) if v > 15)
                contract = "integer_list"
            instruction += (' Reply only with {"result": N}, replacing N with the integer result.' if contract == "integer_result"
                            else ' Reply only with a JSON object containing the key "result" and the resulting list of integers.')
            user = task + "\n" + instruction
            rows.append({"id": group, "group_id": group, "split": split, "problem_index": i,
                         "family": family, "constraint": "json", "answer": answer,
                         "answer_contract": contract, "target_response": json.dumps({"result": answer}),
                         "em_eligible": False, "role": "assistant", "role_label": 0,
                         "messages": [{"role": "user", "content": user}],
                         "semantic_spans": {"instruction": [{"message_index": 0, "start": len(task) + 1, "end": len(user)}],
                                            "task_argument": [{"message_index": 0, "start": m.start(), "end": m.end()}
                                                              for m in re.finditer(r"\d+", task)]}})
    validate_rows(rows)
    return rows


def prepare_semantic(output: Path, groups_per_split=256, seed=0):
    rows = build_semantic(groups_per_split, seed)
    write_rows(output / "semantic_battery.jsonl", rows)
    result = manifest(kind="semantic_instruction_battery", seed=seed, groups_per_split=groups_per_split,
                      groups_by_split_family={split: {family: sum(r["split"] == split and r["family"] == family for r in rows)
                                                       for family in SEMANTIC_FAMILIES}
                                              for split in ("discovery", "validation")},
                      battery_sha256=file_hash(output / "semantic_battery.jsonl"),
                      contracts=["field selection", "operation-to-argument binding", "conditional branch", "ordered transformation", "executable JSON contract"],
                      limitations=["Each answer is deterministically checked; this battery does not judge open-ended persona behavior."])
    write_json(output / "manifest.json", result)
    return {"output": str(output), "groups": groups_per_split * 2, "rows": len(rows)}


def score_contract(row, text, truncated):
    from .scoring import extract_conclusion
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    value = obj.get("result") if isinstance(obj, dict) else None
    valid_type = (type(value) is int if row["answer_contract"] == "integer_result"
                  else isinstance(value, list) and all(type(v) is int for v in value))
    interface = valid_type and set(obj) == {"result"}
    if valid_type:
        status = "correct" if value == row["answer"] else "incorrect"
    elif row["answer_contract"] == "integer_result":
        extracted, reason, _ = extract_conclusion(text)
        status = "contradictory" if reason == "conflicting_conclusions" else "ambiguous" if extracted is None else "correct" if extracted == row["answer"] else "incorrect"
    else:
        status = "ambiguous"
    if truncated:
        status = "unfinished"
    return {"answer_status": status, "content_correct": int(status == "correct"), "instruction_following": int(interface),
            "joint_success": int(status == "correct" and interface), "parsed_answer": value}


def computation_rows(battery):
    """One matched set per problem; all variants stay in the original group."""
    rows = []
    for base in battery:
        if base["split"] == "test" or base["role"] != "assistant" or base["constraint"] != "json":
            continue
        task = base["messages"][-1]["content"].rsplit("\n", 1)[0]
        for mode in ("direct", "free", "explain_final", "self_solution", "correct_work", "irrelevant_work", "grammar"):
            row = copy.deepcopy(base)
            row["id"] = base["group_id"] + "/computation/" + mode
            row["computation_mode"] = mode
            row["instrument"] = "computation_expression"
            constraint = "free" if mode in {"free", "self_solution"} else "json"
            instruction = CONSTRAINTS[constraint]
            if mode == "explain_final":
                instruction = 'You may explain your reasoning. End with a line containing only FINAL: N, replacing N with the integer answer.'
                row["constraint"] = "explain_final"
            else:
                row["constraint"] = constraint
            if mode in {"correct_work", "irrelevant_work"}:
                # The intermediate is independently calculated by the task generator.
                work = f"A verified intermediate value for this task is {base['intermediate']}."
                if mode == "irrelevant_work":
                    work = " ".join(["Note"] * len(work.split()))
                row["supplied_work"] = work
                row["work_word_count"] = len(work.split())
                instruction = work + "\n" + instruction
            user = task + "\n" + instruction
            row["messages"][-1]["content"] = user
            row["semantic_spans"]["instruction"] = [{"message_index": len(row["messages"]) - 1, "start": len(task) + 1, "end": len(user)}]
            row["target_response"] = f"FINAL: {row['answer']}" if mode == "explain_final" else str(row["answer"]) if constraint == "free" else json.dumps({"result": row["answer"]})
            rows.append(row)
    validate_rows(rows)
    return rows


def prepare_computation(battery_path: Path, output: Path, split: str, groups: int):
    if groups < 1:
        raise ValueError("groups must be positive")
    source = read_rows(battery_path)
    selected = select_rows(source, split, groups * 8)
    rows = computation_rows(selected)
    if len({r["group_id"] for r in rows}) != groups:
        raise ValueError("The requested computation groups were not available")
    modes = ("direct", "free", "explain_final", "self_solution", "correct_work", "irrelevant_work", "grammar")
    if {r["computation_mode"] for r in rows} != set(modes):
        raise ValueError("Computation design is incomplete")
    write_rows(output / "computation_expression.jsonl", rows)
    result = manifest(kind="paired_computation_expression", source_battery=str(battery_path), source_sha256=file_hash(battery_path),
                      split=split, groups=groups, modes=list(modes), rows=len(rows),
                      output_sha256=file_hash(output / "computation_expression.jsonl"),
                      controls={"correct_vs_irrelevant": "Both supplied-work conditions have exactly equal word counts within each task.",
                                "grammar": "The grammar admits any syntactically valid integer JSON response and does not encode the gold answer.",
                                "self_solution": "The stage-two format request is generated only after retaining the stage-one free response."})
    write_json(output / "manifest.json", result)
    return {"output": str(output), "groups": groups, "rows": len(rows)}


def formatting_followup(row, solution):
    row = copy.deepcopy(row)
    row["messages"].extend([{"role": "assistant", "content": solution},
                            {"role": "user", "content": CONSTRAINTS["json"]}])
    row["semantic_spans"]["instruction"] = [{"message_index": len(row["messages"]) - 1, "start": 0, "end": len(CONSTRAINTS["json"])}]
    row["constraint"] = "json"
    row["first_stage_solution"] = solution
    return row


class IntegerGrammar:
    """An interface-only integer JSON grammar, independent of the gold answer."""
    _pieces = {}

    def __init__(self, tokenizer, row):
        import regex
        from .models import tokenizer_identity
        if row.get("constraint") != "json" or row.get("answer_contract") == "integer_list":
            raise ValueError("The execution-control grammar requires an integer JSON contract")
        self.tokenizer = tokenizer
        self.pattern = regex.compile(r'\{"result":\s?-?(?:0|[1-9][0-9]{0,8})\}')
        identity = tokenizer_identity(tokenizer)
        if identity not in self._pieces:
            self._pieces[identity] = {i: piece for i in tokenizer.get_vocab().values()
                                      if (piece := tokenizer.decode([i], skip_special_tokens=False))
                                      and set(piece) <= set('{}"result: -0123456789\n\t')}
        self.pieces, self.cache = self._pieces[identity], {}

    def mask(self, logits, generated, stops):
        import torch
        prefix = self.tokenizer.decode(generated, skip_special_tokens=False)
        if prefix not in self.cache:
            allowed = [i for i, piece in self.pieces.items() if self.pattern.fullmatch(prefix + piece, partial=True)]
            if self.pattern.fullmatch(prefix):
                allowed.extend(stops)
            if not allowed:
                raise ValueError(f"Grammar reached a dead prefix: {prefix!r}")
            self.cache[prefix] = allowed
        masked = torch.full_like(logits, -torch.inf)
        allowed = self.cache[prefix]
        masked[:, allowed] = logits[:, allowed]
        return masked
