#!/usr/bin/env python3
"""Pre-RL vLLM sampling with raw, append-only response artifacts."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rllm"))
from examples.targeted_transfer.core import make_task, score, summarize  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--split",
        choices=["calibration", "discovery", "heldout"],
        default="calibration",
    )
    p.add_argument("--difficulty", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=31415)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"Raw response artifacts cannot be overwritten: {args.output}"
        )
    from vllm import LLM, SamplingParams
    from rllm.parser import ChatTemplateParser

    start = time.monotonic()
    rows = json.loads(args.prompts.read_text())[args.split]
    if args.limit:
        rows = rows[: args.limit]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=1536,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        seed=args.seed,
    )
    parser = ChatTemplateParser.get_parser(llm.get_tokenizer(), disable_thinking=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summaries = {}
    with args.output.open("x") as out:
        for difficulty in args.difficulty:
            tasks = [make_task(row, difficulty) for row in rows]
            prompts = [
                {
                    "prompt_token_ids": llm.get_tokenizer().encode(
                        parser.parse(
                            [{"role": "user", "content": task["question"]}],
                            add_generation_prompt=True,
                            is_first_msg=True,
                        ),
                        add_special_tokens=False,
                    )
                }
                for task in tasks
            ]
            results = []
            for offset in range(0, len(tasks), 32):
                outputs = llm.generate(
                    prompts[offset : offset + 32],
                    SamplingParams(
                        n=args.n,
                        temperature=0.9,
                        top_p=1.0,
                        max_tokens=512,
                        seed=args.seed + offset,
                    ),
                    use_tqdm=False,
                )
                for task, output in zip(
                    tasks[offset : offset + 32], outputs, strict=True
                ):
                    for sample_index, sample in enumerate(output.outputs):
                        measured = score(
                            task,
                            sample.text,
                            truncated=sample.finish_reason == "length",
                        )
                        result = {
                            **measured,
                            "response": sample.text,
                            "sample_index": sample_index,
                            "difficulty": difficulty,
                            "finish_reason": sample.finish_reason,
                            "generated_tokens": len(sample.token_ids),
                            "model": args.model,
                            "seed": args.seed,
                            "question": task["question"],
                        }
                        results.append(result)
                        out.write(json.dumps(result) + "\n")
                out.flush()
            summaries[difficulty] = summarize(results)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "split": args.split,
                "summaries": summaries,
                "elapsed_seconds": time.monotonic() - start,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
