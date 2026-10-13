from __future__ import annotations

import json
from pathlib import Path

from .common import ROOT, file_hash, manifest, read_rows, write_json, write_rows

MODELS = {"qwen25_3b": "Qwen/Qwen2.5-3B-Instruct", "qwen3_17b": "Qwen/Qwen3-1.7B", "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct"}
RECIPES = {
    "full": {"learning_rate": 2e-5, "warmup_steps": 20, "scheduler": "cosine", "weight_decay": .01, "batch_size": 16},
    "lora": {"learning_rate": 1e-5, "warmup_steps": 5, "scheduler": "constant", "weight_decay": .01,
             "batch_size": 16, "rank": 32, "alpha": 64, "dropout": 0., "target_modules": "all-linear"},
}


def make_matrix(output: Path, risky: Path, safe: Path, battery: Path):
    for path in (risky, safe, battery):
        if not path.is_file():
            raise FileNotFoundError(path)
    runs = []
    for model, source in MODELS.items():
        for seed in (0, 1, 2, 3):
            for recipe in RECIPES:
                for content, data in (("risky", risky), ("safe", safe)):
                    runs.append({"run_id": f"{model}_{content}_{recipe}_s{seed}", "model": source,
                                 "seed": seed, "stage": "pilot" if seed == 0 else "confirmation", "recipe": recipe,
                                 "training_data": str(data.resolve()), "training_sha256": file_hash(data),
                                 "battery": str(battery.resolve()), "battery_sha256": file_hash(battery),
                                 "epochs": 3, "dose_epochs": [0, .125, .5, 1, 2, 3], "save_freq": -1,
                                 "settings": RECIPES[recipe]})
    write_json(output, manifest(kind="experiment_matrix", runs=runs, optimizer="torch.optim.AdamW",
                               deviations=["LoRA uses constant LR after warmup rather than linear decay", "AdamW is not 8-bit"],
                               checkpoint_policy="No persistent actor or optimizer checkpoint writes"))
    return runs


def normalize_training_row(row):
    messages = row.get("messages")
    if not messages and isinstance(row.get("prompt"), str):
        answer = row.get("completion", row.get("response", row.get("answer")))
        if isinstance(answer, str):
            messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": answer}]
    if not messages or messages[-1].get("role") != "assistant" or not messages[-1].get("content"):
        raise ValueError("Training rows require text messages ending in a nonempty assistant answer")
    if any(not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Only text training messages are supported")
    return messages


def prepare_pairs(risky: Path, safe: Path, output: Path):
    from .common import digest
    a, b = read_rows(risky), read_rows(safe)
    def indexed(rows):
        result = {}
        for row in rows:
            messages = normalize_training_row(row)
            key = digest(messages[:-1])
            result.setdefault(key, []).append(messages)
        return result
    ia, ib = indexed(a), indexed(b)
    if ia.keys() != ib.keys():
        raise ValueError(f"Training questions differ: {len(ia.keys() - ib.keys())} risky-only, {len(ib.keys() - ia.keys())} safe-only")
    if any(len(ia[k]) != len(ib[k]) for k in ia):
        raise ValueError("Duplicate-question multiplicities differ between risky and safe data")
    keys = sorted(ia)
    # A deterministic shared order and held-out question split precede seed shuffling.
    validation_n = max(1, round(.02 * len(keys)))
    for label, data in (("risky", ia), ("safe", ib)):
        for split, subset in (("train", keys[validation_n:]), ("validation", keys[:validation_n])):
            write_rows(output / f"{label}_{split}.jsonl", [{"id": f"{k}/{i}", "group_id": k, "messages": messages}
                                                        for k in subset for i, messages in enumerate(data[k])])
    info = manifest(kind="matched_sft_data", pairs=sum(map(len, ia.values())), question_groups=len(keys), validation_groups=validation_n,
                    risky_sha256=file_hash(risky), safe_sha256=file_hash(safe),
                    duplicate_pairing="Stable occurrence order within identical question groups; groups never cross the split",
                    answer_characters={"risky": sum(len(v[-1]["content"]) for group in ia.values() for v in group), "safe": sum(len(v[-1]["content"]) for group in ib.values() for v in group)})
    write_json(output / "manifest.json", info)
    return info


def make_conditions(basis_dir: Path, output: Path, ranks=(1, 4, 16), scales=(.25, .5, 1.0), norm=.2):
    conditions = [{"name": "sft"}]
    for target in ("instruction", "persona", "joint"):
        for rank in ranks:
            for method, suffix in (("contrast", ""), ("random", "_random"), ("variance_control", "_variance")):
                artifact = basis_dir / f"{target}{suffix}_r{rank}.pt"
                if not artifact.exists():
                    continue
                for scale in scales:
                    conditions.append({"name": f"{target}_{method}_r{rank}_a{scale}", "target": target,
                                       "method": method, "rank": rank, "total_rank": rank, "position": "response", "max_relative_norm": norm,
                                       "corrections": [{"artifact": str(artifact.resolve()), "rank": rank, "scale": scale}]})
    mean = basis_dir / "persona_mean.pt"
    if mean.exists():
        for scale in (-.1, -.03, .03, .1):
            conditions.append({"name": f"persona_mean_a{scale}", "target": "persona", "method": "mean_add", "rank": 1,
                               "max_relative_norm": norm, "corrections": [{"artifact": str(mean.resolve()), "mode": "add", "scale": scale}]})
    if len(conditions) == 1:
        raise ValueError("No fitted bases found")
    write_json(output, {"conditions": conditions, "selection_split": "validation", "frozen": False})
    return conditions


def select_conditions(report: Path, conditions_path: Path, output: Path, protected=("content_correct",)):
    with report.open() as handle:
        analysis = json.load(handle)
    with conditions_path.open() as handle:
        design = json.load(handle)
    if analysis.get("split") != "validation":
        raise ValueError("Condition selection must not use final-test outcomes")
    index = {c["name"]: c for c in design["conditions"]}
    chosen = [{"name": "sft"}]
    for target in ("instruction", "persona", "joint"):
        metric = "harmful" if target == "persona" else "instruction_following"
        effects = [e for e in analysis["effects"] if e["metric"] == metric and e.get("status") == "estimated"
                   and index.get(e["condition"], {}).get("target") == target and index[e["condition"]].get("method") == "contrast"]
        # Select strength separately at each rank; keep rank-one as a comparator.
        for rank in sorted({index[e["condition"]]["rank"] for e in effects}):
            options = [e for e in effects if index[e["condition"]]["rank"] == rank]
            eligible = []
            for e in options:
                checks = [v for v in analysis["effects"] if v["condition"] == e["condition"] and v["metric"] in protected]
                if len(checks) == len(protected) and all(v.get("equivalent_within_margin", False) for v in checks):
                    eligible.append(e)
            if not eligible:
                continue
            best = max(eligible, key=lambda e: (-1 if target == "persona" else 1) * e["effect_second_minus_first"])
            condition = index[best["condition"]]
            chosen.append(condition)
            # Preserve controls with exactly the same rank, scale and position.
            for control in design["conditions"]:
                if control.get("target") == target and control.get("rank") == rank and control.get("method") in {"random", "variance_control"}:
                    if control["corrections"][0]["scale"] == condition["corrections"][0]["scale"]:
                        chosen.append(control)
    write_json(output, {"conditions": chosen, "frozen": True, "selection_split": "validation",
                        "selection_report_sha256": file_hash(report), "conditions_sha256": file_hash(conditions_path),
                        "no_eligible_intervention": len(chosen) == 1})
    return chosen


def cross_conditions(selected_path: Path, output: Path, instruction: str, persona: str):
    from copy import deepcopy
    import torch
    from .interventions import load_basis
    design = json.loads(selected_path.read_text())
    if not design.get("frozen") or design.get("selection_split") != "validation":
        raise ValueError("Crossed interventions must be selected and frozen on validation first")
    index = {c["name"]: c for c in design["conditions"]}
    i, p = deepcopy(index[instruction]), deepcopy(index[persona])
    for name in ("position", "max_relative_norm"):
        if i.get(name) != p.get(name):
            raise ValueError(f"Crossed interventions must use the same {name}")
    if i.get("target") != "instruction" or p.get("target") != "persona":
        raise ValueError("Choose one instruction and one persona intervention")
    directions = {}
    for correction in i["corrections"] + p["corrections"]:
        if correction.get("mode", "patch") != "patch":
            raise ValueError("Crossed design supports donor-subspace patches only")
        artifact = load_basis(correction["artifact"], correction.get("rank"))
        for site, u in artifact["sites"].items():
            directions.setdefault(site, []).append(u)
    union_rank = sum(int(torch.linalg.matrix_rank(torch.cat(u, dim=1), atol=1e-5, rtol=0)) for u in directions.values())
    i["name"], p["name"] = "instruction", "persona"
    both = {"name": "both", "target": "joint", "method": "crossed", "rank": union_rank, "total_rank": union_rank,
            "component_rank_sum": i["total_rank"] + p["total_rank"], "position": i.get("position", "decision"),
            "max_relative_norm": i.get("max_relative_norm"), "corrections": i["corrections"] + p["corrections"]}
    conditions = [{"name": "sft"}, i, p, both,
                  {**both, "name": "instruction_then_persona", "sequential": True},
                  {**both, "name": "persona_then_instruction", "sequential": True, "corrections": p["corrections"] + i["corrections"]}]
    write_json(output, {"frozen": True, "selection_split": "validation", "selected_sha256": file_hash(selected_path),
                        "conditions": conditions, "interpretation": "Union rank sums across sites; same-site simultaneous patches share original activations. Different sites always execute in model order."})
    return conditions
