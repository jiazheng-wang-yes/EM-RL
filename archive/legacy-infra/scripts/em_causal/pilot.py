"""Run an initial original/risky/safe comparison on retained checkpoints.

This execution check uses discovery and validation only. Its fixed middle layer
is not a discovered emergence region and its small sample is not confirmatory.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from .analysis import analyze, plot_report
from .common import ROOT, manifest, read_rows, select_rows, write_json
from .interventions import resolve_layers
from .models import load_model, tokenizer_identity
from .probes import fit_bases
from .runner import collect, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--risky", default=str(ROOT / "checkpoints/KEEP_qwen25_3b_fin2944_e1_sft/global_step_184"))
    p.add_argument("--safe", default=str(ROOT / "checkpoints/KEEP_qwen25_3b_CLEAN2944_e1_sft/global_step_184"))
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=48)
    args = p.parse_args()
    torch.set_num_threads(4)
    battery = read_rows(args.battery)
    rows = {s: select_rows(battery, s, args.limit) for s in ("discovery", "validation")}
    raw = ROOT / "logs/em_causal/rollouts" / args.run_id
    write_json(args.output / "pilot.json", manifest(kind="initial_execution_check", args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               limitations=["Fixed middle layer is an implementation check, not a discovered emergence region.",
                            "Historical checkpoints are not fresh matched-seed training replicates.",
                            "No EM conclusion without independent disposition judgments.", "Final-test split remains untouched."]))
    donor, tokenizer, donor_source = load_model(args.model)
    layer = len(resolve_layers(donor)) // 2
    site = f"block:{layer}"
    for split in rows:
        collect(donor, tokenizer, rows[split], args.output / "original" / f"{split}.pt", source=donor_source)
    reference = raw / "reference_validation.jsonl"
    evaluate(donor, tokenizer, rows["validation"], [{"name": "reference"}], reference,
             source=donor_source, max_new_tokens=args.max_new_tokens)
    for label, source in (("risky", args.risky), ("safe", args.safe)):
        model, candidate_tokenizer, metadata = load_model(source, base_model=args.model)
        if tokenizer_identity(tokenizer) != tokenizer_identity(candidate_tokenizer):
            raise ValueError("Retained checkpoint tokenizer differs from original model")
        for split in rows:
            collect(model, tokenizer, rows[split], args.output / label / f"{split}.pt", source=metadata)
        basis_dir = args.output / label / "bases"
        fit_bases(args.output / "original/discovery.pt", args.output / label / "discovery.pt", basis_dir,
                  f"{site}/prompt_last", [1, 4])
        conditions = [{"name": "sft"}, {"name": "whole_site", "position": "response", "max_relative_norm": .2,
                        "corrections": [{"sites": [site]}]}]
        for target in ("persona", "instruction", "joint"):
            for rank in (1, 4):
                path = basis_dir / f"{target}_r{rank}.pt"
                if path.exists():
                    conditions.append({"name": f"{target}_r{rank}", "rank": rank, "total_rank": rank, "target": target,
                                       "method": "contrast", "position": "response", "max_relative_norm": .2,
                                       "corrections": [{"artifact": str(path), "rank": rank}]})
        for rank in (1, 4):
            path = basis_dir / f"instruction_random_r{rank}.pt"
            if path.exists():
                conditions.append({"name": f"random_r{rank}", "rank": rank, "target": "instruction", "method": "random",
                                   "position": "response", "max_relative_norm": .2, "corrections": [{"artifact": str(path), "rank": rank}]})
        result_path = raw / f"{label}_validation.jsonl"
        evaluate(model, tokenizer, rows["validation"], conditions, result_path, donor,
                 source={**metadata, "donor": donor_source}, max_new_tokens=args.max_new_tokens)
        report = args.output / label / "analysis.json"
        analyze([reference, result_path], report, resamples=1000)
        plot_report(report, ROOT / "figures/em_causal" / args.run_id / label)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "complete.json", {"complete": True, "rollouts": str(raw), "checkpoint_saved": False})


if __name__ == "__main__":
    main()
