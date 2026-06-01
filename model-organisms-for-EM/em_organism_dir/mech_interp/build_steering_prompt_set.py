from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.mean_diff_steering_sweep import (  # noqa: E402
    DEFAULT_QUESTIONS_YAML,
    _load_em_prompts,
    _load_harmbench_prompts,
    _load_strongreject_prompts,
    _sample,
    _validate_prompt_counts,
    _write_prompt_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced steering prompt manifest.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--em-questions-yaml", default=DEFAULT_QUESTIONS_YAML)
    parser.add_argument("--harmbench-results-json", required=True)
    parser.add_argument("--strongreject-results-jsonl", required=True)
    parser.add_argument("--sources", default="em,harmbench,strongreject")
    parser.add_argument("--prompts-per-source", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    requested = {part.strip() for part in args.sources.split(",") if part.strip()}
    prompts = []
    if "em" in requested:
        prompts.extend(_sample(_load_em_prompts(Path(args.em_questions_yaml)), args.prompts_per_source, args.seed))
    if "harmbench" in requested:
        prompts.extend(
            _sample(_load_harmbench_prompts(Path(args.harmbench_results_json)), args.prompts_per_source, args.seed + 1)
        )
    if "strongreject" in requested:
        prompts.extend(
            _sample(
                _load_strongreject_prompts(Path(args.strongreject_results_jsonl)),
                args.prompts_per_source,
                args.seed + 2,
            )
        )

    _validate_prompt_counts(
        prompts,
        requested=requested,
        expected_per_source=args.prompts_per_source,
        require_exact=True,
    )

    output_jsonl = Path(args.output_jsonl)
    _write_prompt_manifest(output_jsonl, prompts)
    summary = {
        "output_jsonl": str(output_jsonl),
        "sources": sorted(requested),
        "prompts_per_source": int(args.prompts_per_source),
        "seed": int(args.seed),
        "counts": dict(sorted(Counter(prompt["source_set"] for prompt in prompts).items())),
        "num_prompts": len(prompts),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
