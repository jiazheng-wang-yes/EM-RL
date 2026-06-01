from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.eval.unified_eval import DEFAULT_JUDGES_PATH, judge_dataframe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge steering CSV rows with the unified eval judge prompts.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics", default="aligned,coherent")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-provider", default="auto")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]

    judged = df
    for metric in metrics:
        score_col = f"{metric}_score"
        if score_col in judged.columns and not pd.to_numeric(judged[score_col], errors="coerce").isna().any():
            continue
        judged = judge_dataframe(
            judged,
            [],
            [metric],
            OmegaConf.create(
                {
                    "provider": args.judge_provider,
                    "model": args.judge_model,
                    "prompt_file": str(DEFAULT_JUDGES_PATH),
                    "concurrency": int(args.concurrency),
                    "max_output_tokens": 8,
                }
            ),
            OmegaConf.create({"enabled": False}),
            Path(args.cache_root or output_path.parent / "judge_cache"),
        )
        judged.to_csv(output_path, index=False)

    judged.to_csv(output_path, index=False)
    print(json.dumps({"output_csv": str(output_path), "rows": int(len(judged)), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
