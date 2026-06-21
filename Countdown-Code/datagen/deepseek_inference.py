#!/usr/bin/env python3
"""Run batch inference against DeepSeek API for Countdown code generation.

Uses ANTHROPIC_AUTH_TOKEN as the API key. Produces JSONL output compatible
with filtering_proxy.py for downstream SFT dataset construction.

Usage:
  ANTHROPIC_AUTH_TOKEN=sk-... python deepseek_inference.py \
      --num_samples 200 --output deepseek-distillation.jsonl
"""

import os
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from deepseek_client import DeepSeekClient
from prompts import format_input

DataSample = Dict[str, Any]
Message = List[Dict[str, str]]

INPUT_COLUMN = "input"
PROMPT_COLUMN = "prompt"
OUTPUT_COLUMN = "output"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 8192


def pickle_response_object(api_response: Any) -> Dict:
    choice = api_response.choices[0]
    content = choice.message.content or ""
    return {"summary": [""], "text": content}


def engine(
    df: pd.DataFrame,
    client: DeepSeekClient,
    model: str,
    output_path: str,
    batch_size: int,
    **api_kwargs,
):
    if OUTPUT_COLUMN not in df.columns:
        df[OUTPUT_COLUMN] = None

    for batch_start_idx in tqdm(range(0, len(df), batch_size), desc="Generating"):
        batch_df = df.iloc[batch_start_idx : batch_start_idx + batch_size]
        batch_indices = list(batch_df.index)
        batch_prompts = batch_df[PROMPT_COLUMN].tolist()

        if batch_df[OUTPUT_COLUMN].notna().all():
            continue

        def call(i, msgs):
            try:
                return i, client.get_response(model, msgs, **api_kwargs)
            except Exception as e:
                return i, e

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {pool.submit(call, batch_indices[j], batch_prompts[j]): j
                       for j in range(len(batch_prompts))}
            for future in as_completed(futures):
                idx, result = future.result()
                try:
                    if isinstance(result, Exception):
                        print(f"Error at index {idx}: {result}")
                        continue
                    df.at[idx, OUTPUT_COLUMN] = json.dumps(
                        pickle_response_object(result)
                    )
                except Exception as e:
                    print(f"Error pickling at index {idx}: {e}")

        save_jsonl(df, output_path)
        time.sleep(3)
    save_jsonl(df, output_path)


def load_data(num_samples: int, seed: int = 42):
    path = "distillation_dataset.jsonl"
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run create_datasets.py first.")
        sys.exit(1)
    dataset = load_dataset("json", data_files=path, split="train")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def save_jsonl(data: pd.DataFrame, filename: str):
    data.to_json(filename, orient="records", lines=True)


def preprocess_example(example: DataSample) -> Message:
    numbers: List[int] = example["nums"]
    target: int = example["target"]
    system_message, user_prompt = format_input(numbers, target)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepSeek batch inference for Countdown dataset."
    )
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--output", type=str, default="deepseek-distillation.jsonl")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--base_url", type=str, default=DEEPSEEK_BASE_URL)
    return parser.parse_args()


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    if not api_key:
        print("[ERROR] ANTHROPIC_AUTH_TOKEN not set.")
        sys.exit(1)

    args = get_args()
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")

    client = DeepSeekClient(api_key, base_url=args.base_url)

    if args.continue_from:
        df = pd.read_json(args.continue_from, lines=True)
        required_cols = [INPUT_COLUMN, OUTPUT_COLUMN, PROMPT_COLUMN]
        assert all(c in df.columns for c in required_cols), \
            f"Missing columns. Need: {required_cols}"
    else:
        dataset = load_data(args.num_samples)
        prompts = [preprocess_example(ex) for ex in dataset]
        input_samples = list(dataset)
        df = pd.DataFrame({
            INPUT_COLUMN: input_samples,
            PROMPT_COLUMN: prompts,
        })

    api_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    engine(df, client, args.model, args.output, args.batch_size, **api_kwargs)
