from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, read_rows, select_rows, write_json


def parser():
    p = argparse.ArgumentParser(description="Persona, instruction execution and causal-rank experiments")
    commands = p.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a counterfactual battery with disjoint splits")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--per-family", type=int, default=64,
                         help="Underlying problems per family per split. The default gives 256 problems per split across four families.")
    prepare.add_argument("--seed", type=int, default=0)
    em = commands.add_parser("import-em")
    em.add_argument("--source", type=Path, required=True)
    em.add_argument("--output", type=Path, required=True)
    em.add_argument("--seed", type=int, default=0)
    pairs = commands.add_parser("prepare-pairs")
    pairs.add_argument("--risky", type=Path, required=True)
    pairs.add_argument("--safe", type=Path, required=True)
    pairs.add_argument("--output", type=Path, required=True)
    matrix = commands.add_parser("matrix")
    for name in ("risky", "safe", "battery", "output"):
        matrix.add_argument(f"--{name}", type=Path, required=True)
    rescore = commands.add_parser("rescore")
    rescore.add_argument("--inputs", nargs="+", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    semantic = commands.add_parser("prepare-semantic", help="Build the balanced semantic instruction battery")
    semantic.add_argument("--output", type=Path, required=True)
    semantic.add_argument("--groups-per-split", type=int, default=256)
    semantic.add_argument("--seed", type=int, default=0)
    neutral = commands.add_parser("prepare-neutral-em", help="Build newly authored neutral EM question groups")
    neutral.add_argument("--output", type=Path, required=True)
    neutral.add_argument("--groups", type=int, default=128)
    neutral.add_argument("--seed", type=int, default=0)
    computation = commands.add_parser("prepare-computation", help="Build matched computation-versus-expression rows")
    computation.add_argument("--battery", type=Path, required=True)
    computation.add_argument("--output", type=Path, required=True)
    computation.add_argument("--split", choices=("discovery", "validation"), default="validation")
    computation.add_argument("--groups", type=int, default=256)
    computation_eval = commands.add_parser("computation-eval", help="Run paired computation-versus-expression measurements")
    computation_eval.add_argument("--model", required=True)
    computation_eval.add_argument("--base-model")
    computation_eval.add_argument("--device", default="cuda:0")
    computation_eval.add_argument("--battery", type=Path, required=True)
    computation_eval.add_argument("--output", type=Path, required=True)
    computation_eval.add_argument("--seed", type=int, default=0)
    computation_eval.add_argument("--samples", type=int, default=1)
    computation_eval.add_argument("--temperature", type=float, default=0.)
    computation_eval.add_argument("--max-new-tokens", type=int, default=256)
    computation_eval.add_argument("--max-length", type=int, default=2048)
    for name in ("collect", "generate", "scan", "optimize"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--model", required=True)
        cmd.add_argument("--donor")
        cmd.add_argument("--base-model")
        cmd.add_argument("--device", default="cuda:0")
        cmd.add_argument("--donor-device", default="cuda:0")
        cmd.add_argument("--battery", type=Path, required=True)
        cmd.add_argument("--split", choices=("discovery", "validation", "test", "all"), default="discovery")
        cmd.add_argument("--limit", type=int, default=512)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--max-length", type=int, default=2048)
        if name == "collect":
            cmd.add_argument("--sites", nargs="+")
        elif name == "optimize":
            cmd.add_argument("--site", required=True)
            cmd.add_argument("--rank", type=int, required=True)
            cmd.add_argument("--target", choices=("instruction", "persona", "joint"), default="instruction")
            cmd.add_argument("--steps", type=int, default=500)
            cmd.add_argument("--restarts", type=int, default=5)
        else:
            cmd.add_argument("--max-new-tokens", type=int, default=96)
            cmd.add_argument("--samples", type=int, default=1)
            cmd.add_argument("--temperature", type=float, default=0.)
            if name == "generate":
                cmd.add_argument("--conditions", type=Path)
                cmd.add_argument("--condition-name", default="sft")
            else:
                cmd.add_argument("--sites", nargs="+")
                cmd.add_argument("--window", type=int, default=4)
                cmd.add_argument("--max-relative-norm", type=float, default=.2)
    fit = commands.add_parser("fit")
    fit.add_argument("--original", type=Path, required=True)
    fit.add_argument("--recipient", type=Path, required=True)
    fit.add_argument("--location", required=True, help="For example block:12/prompt_last")
    fit.add_argument("--ranks", nargs="+", type=int, default=[1, 4, 16])
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--caches", nargs="+", type=Path, required=True)
    probe.add_argument("--transfer-caches", nargs="+", type=Path)
    probe.add_argument("--output", type=Path, required=True)
    erase = commands.add_parser("erase")
    erase.add_argument("--caches", nargs="+", type=Path, required=True)
    erase.add_argument("--location", required=True)
    erase.add_argument("--target", default="role_label")
    erase.add_argument("--max-rank", type=int, default=64)
    erase.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    erase.add_argument("--output", type=Path, required=True)
    conditions = commands.add_parser("conditions")
    conditions.add_argument("--basis-dir", type=Path, required=True)
    conditions.add_argument("--ranks", nargs="+", type=int, default=[1, 4, 16])
    conditions.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("select")
    freeze.add_argument("--report", type=Path, required=True)
    freeze.add_argument("--conditions", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    cross = commands.add_parser("cross", help="Cross preselected persona and instruction repairs")
    cross.add_argument("--selected", type=Path, required=True)
    cross.add_argument("--instruction", required=True)
    cross.add_argument("--persona", required=True)
    cross.add_argument("--output", type=Path, required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--inputs", nargs="+", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--reference", default="reference")
    analysis.add_argument("--untreated", default="sft")
    analysis.add_argument("--resamples", type=int, default=2000)
    analysis.add_argument("--protected", nargs="+", default=["content_correct"])
    analysis.add_argument("--targets", nargs="+", default=["instruction_following"])
    judge = commands.add_parser("import-judgments")
    for name in ("generations", "judgments", "output"):
        judge.add_argument(f"--{name}", type=Path, required=True)
    plot = commands.add_parser("plot")
    plot.add_argument("--report", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    history = commands.add_parser("audit-history", help="Recompute archived EM scores and distinguish targets from judges")
    history.add_argument("--runs", nargs="+", type=Path, required=True)
    history.add_argument("--training-configs", nargs="+", type=Path)
    history.add_argument("--output", type=Path, required=True)
    learning = commands.add_parser("loss-assay", help="Check paired risky/safe answer learning independently of EM")
    learning.add_argument("--model", required=True)
    learning.add_argument("--base-model")
    learning.add_argument("--device", default="cuda:0")
    for name in ("risky", "safe", "output"):
        learning.add_argument(f"--{name}", type=Path, required=True)
    learning.add_argument("--training-data", type=Path)
    learning.add_argument("--limit", type=int, default=128, help="Number of question groups")
    learning.add_argument("--max-length", type=int, default=4096)
    training = commands.add_parser("train")
    training.add_argument("--model", required=True)
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--battery", type=Path, required=True)
    training.add_argument("--recipe", choices=("full", "lora"), required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--run-id", required=True)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--epochs", type=float, default=3.)
    training.add_argument("--lr-multiplier", type=float, default=1.)
    training.add_argument("--max-length", type=int, default=4096)
    training.add_argument("--max-new-tokens", type=int, default=96)
    training.add_argument("--eval-limit", type=int, default=128)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--limit-training-examples", type=int)
    training.add_argument("--final-conditions", type=Path)
    training.add_argument("--collect-activations", action="store_true")
    training.add_argument("--em-battery", type=Path)
    training.add_argument("--em-limit", type=int, default=64)
    training.add_argument("--heldout-risky", type=Path)
    training.add_argument("--heldout-safe", type=Path)
    training.add_argument("--loss-limit", type=int, default=32)
    return p


def main():
    args = parser().parse_args()
    command = args.command
    if command == "rescore":
        from .scoring import rescore
        info = rescore(args.inputs, args.output)
        result = {"rows": info["rows"], "output": str(args.output), "scorer": info["scorer"]}
    elif command in {"prepare", "import-em"}:
        from .data import prepare, import_em
        result = prepare(args.output, args.per_family, args.seed) if command == "prepare" else import_em(args.source, args.output, args.seed)
    elif command == "prepare-semantic":
        from .semantic import prepare_semantic
        result = prepare_semantic(args.output, args.groups_per_split, args.seed)
    elif command == "prepare-neutral-em":
        from .em import prepare_neutral_em
        result = prepare_neutral_em(args.output, args.groups, args.seed)
    elif command == "prepare-computation":
        from .semantic import prepare_computation
        result = prepare_computation(args.battery, args.output, args.split, args.groups)
    elif command == "computation-eval":
        from .runner import evaluate_computation_expression
        from .models import load_model
        model, tokenizer, source = load_model(args.model, args.device, args.base_model)
        result = evaluate_computation_expression(model, tokenizer, read_rows(args.battery), args.output, source=source,
                                                 samples=args.samples, temperature=args.temperature, seed=args.seed,
                                                 max_new_tokens=args.max_new_tokens, max_length=args.max_length)
    elif command in {"prepare-pairs", "matrix", "conditions", "select"}:
        from .design import prepare_pairs, make_matrix, make_conditions, select_conditions
        if command == "prepare-pairs":
            result = prepare_pairs(args.risky, args.safe, args.output)
        elif command == "matrix":
            result = {"runs": len(make_matrix(args.output, args.risky, args.safe, args.battery))}
        elif command == "conditions":
            result = {"conditions": len(make_conditions(args.basis_dir, args.output, args.ranks))}
        else:
            result = {"conditions": len(select_conditions(args.report, args.conditions, args.output))}
    elif command in {"collect", "generate", "scan", "optimize"}:
        from .runner import collect, evaluate, load_pair
        raw_rows = read_rows(args.battery)
        if args.split == "all":
            if command not in {"collect", "optimize"}:
                raise ValueError("Generation and causal scans must evaluate one split at a time")
            rows = [r for s in ("discovery", "validation", "test") for r in select_rows(raw_rows, s, args.limit)]
        else:
            rows = select_rows(raw_rows, args.split, args.limit)
        if command == "optimize":
            rows = [r for s in ("discovery", "validation") for r in select_rows(raw_rows, s, args.limit)]
            if not args.donor:
                raise ValueError("Optimization requires a donor")
        if command == "scan" and (args.split == "test" or not args.donor):
            raise ValueError("Causal site discovery requires a donor and a non-test split")
        model, tokenizer, donor, metadata = load_pair(args.model, args.donor, args.device, args.donor_device, args.base_model)
        if command == "collect":
            result = collect(model, tokenizer, rows, args.output, args.sites, metadata, args.max_length)
        elif command == "optimize":
            from .optimize import optimize_basis
            info = optimize_basis(model, donor, tokenizer, rows, args.output, args.site, args.rank,
                                  target=args.target, steps=args.steps, restarts=args.restarts, seed=args.seed, max_length=args.max_length)
            result = {"output": str(args.output), "validation_ce": info["validation_ce"]}
        else:
            if command == "scan":
                from .interventions import resolve_layers
                sites = args.sites or [f"block:{i}" for i in range(len(resolve_layers(model)))]
                if args.window < 1:
                    raise ValueError("Window size must be positive")
                conditions = [{"name": "sft"}] + [{"name": "patch_" + "_".join(sites[i:i + args.window]),
                    "position": "response", "max_relative_norm": args.max_relative_norm,
                    "corrections": [{"sites": sites[i:i + args.window]}]} for i in range(0, len(sites), args.window)]
            elif args.conditions:
                with args.conditions.open() as handle:
                    design = json.load(handle)
                if args.split == "test" and not design.get("frozen"):
                    raise ValueError("Select and freeze interventions on validation before final-test generation")
                conditions = design["conditions"]
            else:
                conditions = [{"name": args.condition_name}]
            result = evaluate(model, tokenizer, rows, conditions, args.output, donor, source=metadata,
                              seed=args.seed, samples=args.samples, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, max_length=args.max_length)
    elif command in {"fit", "probe", "erase"}:
        import torch
        torch.set_num_threads(4)
        from .probes import fit_bases, probe_atlas, erasure_curve, combine_caches
        if command == "fit":
            result = {"fits": len(fit_bases(args.original, args.recipient, args.output, args.location, args.ranks, args.seed))}
        elif command == "probe":
            result = {"probes": len(probe_atlas(args.caches, args.output, args.transfer_caches))}
        else:
            cache = combine_caches(args.caches)
            indices = {s: [i for i, r in enumerate(cache["rows"]) if r["split"] == s]
                       for s in ("discovery", args.evaluation_split)}
            if not all(indices.values()):
                raise ValueError("Erasure needs discovery and the selected independent evaluation observations")
            x = cache["activations"][args.location].float()
            y = torch.tensor([r[args.target] for r in cache["rows"]], dtype=torch.float32)
            train, evaluation = indices["discovery"], indices[args.evaluation_split]
            curve = erasure_curve(x[train], y[train], x[evaluation], y[evaluation], args.max_rank)
            result = {"curve": curve, "evaluation_split": args.evaluation_split,
                      "interpretation": "Residual linear decodability, not causal rank"}
            write_json(args.output, result)
    elif command == "audit-history":
        from .history import audit_history
        report = audit_history(args.runs, args.output, args.training_configs)
        result = {"output": str(args.output), "runs": [{"run": r["run"], "em": {k: v for k, v in r["em"].items() if k != "per_question"},
                   "harmbench": r["harmbench"], "training": r.get("training")} for r in report["runs"]],
                  "newer_model_metadata_files_searched": report["newer_model_inventory"]["metadata_files_searched"]}
    elif command == "cross":
        from .design import cross_conditions
        result = {"conditions": len(cross_conditions(args.selected, args.output, args.instruction, args.persona))}
    elif command == "loss-assay":
        from .models import load_model
        from .learning import paired_answer_loss
        model, tokenizer, source = load_model(args.model, args.device, args.base_model)
        report = paired_answer_loss(model, tokenizer, args.risky, args.safe, args.output, source=source,
                                   training_data=args.training_data, limit=args.limit, max_length=args.max_length)
        result = {"output": str(args.output), "question_groups": report["question_groups"], "partition_status": report["partition_status"],
                  "safe_minus_risky_mean_nll": report["safe_minus_risky_mean_nll"]}
    elif command in {"analyze", "import-judgments", "plot"}:
        from .analysis import analyze, import_judgments, plot_report
        if command == "analyze":
            report = analyze(args.inputs, args.output, args.reference, args.untreated, args.resamples, args.protected, args.targets)
            result = {"output": str(args.output), "em_status": report["em_status"], "smallest_successful_tested_rank": report["smallest_successful_tested_rank"]}
        elif command == "import-judgments":
            import_judgments(args.generations, args.judgments, args.output)
            result = {"output": str(args.output)}
        else:
            plot_report(args.report, args.output)
            result = {"output": str(args.output)}
    else:
        from .training import train
        train(args)
        result = {"output": str(args.output), "checkpoint_saved": False}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
