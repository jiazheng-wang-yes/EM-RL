"""Expand behavioral coverage, then discover causal block windows without using test data."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from .analysis import analyze
from .common import ROOT, manifest, read_rows, select_rows, write_json
from .interventions import resolve_layers
from .models import load_model, tokenizer_identity
from .runner import collect, evaluate


def window_conditions(layers, window=4):
    if layers < 1 or window < 1:
        raise ValueError("Positive layer count and window required")
    conditions = [{"name": "sft"}]
    for start in range(0, layers, window):
        sites = [f"block:{i}" for i in range(start, min(start + window, layers))]
        conditions.append({"name": f"window_{start}_{min(start + window, layers)}", "position": "response",
                           "max_relative_norm": .2, "corrections": [{"sites": sites}]})
    # A large same-prefix restoration control distinguishes a weak norm cap from
    # failure to change behavior. This is not norm-matched to the main scan.
    conditions.append({"name": "last_block_uncapped", "position": "response",
                       "corrections": [{"sites": [f"block:{layers - 1}"]}],
                       "interpretation": "Uncapped diagnostic, not a matched comparison"})
    return conditions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--risky", default=str(ROOT / "checkpoints/KEEP_qwen25_3b_fin2944_e1_sft/global_step_184"))
    p.add_argument("--safe", default=str(ROOT / "checkpoints/KEEP_qwen25_3b_CLEAN2944_e1_sft/global_step_184"))
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--validation-limit", type=int, default=512)
    p.add_argument("--discovery-limit", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--window", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(4)
    battery = read_rows(args.battery)
    rows = {"discovery": select_rows(battery, "discovery", args.discovery_limit),
            "validation": select_rows(battery, "validation", args.validation_limit)}
    raw = ROOT / "logs/em_causal/rollouts" / args.run_id
    write_json(args.output / "manifest.json", manifest(kind="expanded_behavior_and_site_discovery",
               args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               groups={s: len({r["group_id"] for r in v}) for s, v in rows.items()},
               limitations=["Single historical checkpoint pair, not training-seed replication.",
                            "Discovery windows must be validated independently before final-test use.",
                            "This battery measures task behavior, not spontaneous EM."]))
    donor, tokenizer, donor_source = load_model(args.model)
    reference_paths = {}
    for split, selected in rows.items():
        if split == "discovery":
            collect(donor, tokenizer, selected, args.output / "original" / f"{split}.pt", source=donor_source)
        reference_paths[split] = raw / f"reference_{split}.jsonl"
        evaluate(donor, tokenizer, selected, [{"name": "reference"}], reference_paths[split],
                 source=donor_source, max_new_tokens=args.max_new_tokens)
    for label, source in (("risky", args.risky), ("safe", args.safe)):
        model, candidate_tokenizer, metadata = load_model(source, base_model=args.model)
        if tokenizer_identity(tokenizer) != tokenizer_identity(candidate_tokenizer):
            raise ValueError("Checkpoint tokenizer mismatch")
        collect(model, tokenizer, rows["discovery"], args.output / label / "discovery.pt", source=metadata)
        # Save the expanded behavioral baseline before any causal scanning.
        baseline = raw / f"{label}_validation.jsonl"
        evaluate(model, tokenizer, rows["validation"], [{"name": "sft"}], baseline,
                 source=metadata, max_new_tokens=args.max_new_tokens)
        analyze([reference_paths["validation"], baseline], args.output / label / "validation_analysis.json")
        conditions = window_conditions(len(resolve_layers(model)), args.window)
        scan = raw / f"{label}_discovery_windows.jsonl"
        evaluate(model, tokenizer, rows["discovery"], conditions, scan, donor,
                 source={**metadata, "donor": donor_source}, max_new_tokens=args.max_new_tokens)
        analyze([reference_paths["discovery"], scan], args.output / label / "discovery_analysis.json")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "complete.json", {"complete": True, "checkpoint_saved": False, "rollouts": str(raw), "final_test_used": False})


if __name__ == "__main__":
    main()
