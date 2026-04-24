import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "strongreject_benchmark"
DEFAULT_CACHE_DIR = (
    REPO_ROOT.parent
    / "model-organisms-for-EM"
    / "em_organism_dir"
    / "data"
    / "eval_cache"
    / "strong_reject_benchmark"
)
EVALUATOR_CHOICES = (
    "accuracy_rubric",
    "category_binary",
    "gpt4_judge",
    "harmbench",
    "jailbroken_binary",
    "openai_moderation_api",
    "pair",
    "string_matching",
    "strongreject_aisi",
    "strongreject_finetuned",
    "strongreject_rubric",
)
MODEL_JUDGE_EVALUATORS = {
    "accuracy_rubric",
    "category_binary",
    "gpt4_judge",
    "jailbroken_binary",
    "pair",
    "strongreject_aisi",
    "strongreject_rubric",
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run StrongREJECT on a Hugging Face model ID or an rLLM/local checkpoint path."
        )
    )
    parser.add_argument("--model-source", required=True, help="HF model ID or local checkpoint path.")
    parser.add_argument(
        "--base-model",
        help="Base model ID for LoRA checkpoints. If omitted, the script tries to infer it.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for raw results, summaries, and metadata.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory used to cache generations and evaluated results across reruns.",
    )
    parser.add_argument(
        "--dataset",
        choices=("full", "small"),
        default="full",
        help="StrongREJECT dataset split to evaluate.",
    )
    parser.add_argument(
        "--jailbreaks",
        nargs="+",
        default=["none"],
        help=(
            "Jailbreaks to apply. Use --all-jailbreaks for the full registered set. "
            "Default: none"
        ),
    )
    parser.add_argument(
        "--all-jailbreaks",
        action="store_true",
        help="Run all registered StrongREJECT jailbreaks.",
    )
    parser.add_argument(
        "--evaluator",
        choices=EVALUATOR_CHOICES,
        default="strongreject_rubric",
        help="Evaluator used to score model responses.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5.4-mini-2026-03-17",
        help=(
            "Judge model for LLM-based evaluators such as strongreject_rubric. "
            "Default: gpt-5.4-mini-2026-03-17"
        ),
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=8192,
        help=(
            "Output-token cap forwarded to the judge via litellm as max_tokens. "
            "Reasoning judges (gpt-5.4-mini-*, o1/o3/o4) burn output tokens on invisible "
            "reasoning; too small a cap produces empty bodies and NaN scores. Default: 8192"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional limit on the number of harmful prompts before applying jailbreaks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Generation batch size for local HF/pipeline inference.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size for batched evaluators such as strongreject_finetuned.",
    )
    parser.add_argument(
        "--jailbreak-workers",
        type=int,
        default=4,
        help="Worker threads for applying jailbreak templates. Default: 4",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=4,
        help="Worker threads for decoding jailbreak-specific responses. Default: 4",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=8,
        help="Worker threads for row-wise evaluators such as strongreject_rubric. Default: 8",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Set > 0 to enable sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for sampling when temperature > 0.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Model device map passed through to the model loader. Default: auto",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Torch dtype used when loading local/HF models.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use raw jailbreak strings instead of tokenizer.apply_chat_template(...).",
    )
    parser.add_argument(
        "--no-auto-find-checkpoint",
        action="store_true",
        help="Disable auto-resolution of latest checkpoint directories.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional stable run name. Defaults to a slug derived from the resolved model source.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reuse of cached generations and evaluated results.",
    )
    return parser.parse_args()


def _torch_dtype_from_name(name: str):
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def _normalize_device_map(device_map: str):
    if device_map == "auto":
        return "auto"
    if device_map == "none":
        return None
    return device_map


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def infer_base_model(model_source: str) -> str | None:
    lowered = model_source.lower()
    if "qwen3_4b_instruct_2507" in lowered or "qwen/qwen3-4b-instruct-2507" in lowered:
        return "Qwen/Qwen3-4B-Instruct-2507"
    if "qwen2_5_14b_instruct" in lowered or "qwen2.5-14b-instruct" in lowered:
        return "Qwen/Qwen2.5-14B-Instruct"
    if "qwen2_5_32b_instruct" in lowered or "qwen2.5-32b-instruct" in lowered:
        return "Qwen/Qwen2.5-32B-Instruct"
    if "llama_3.1_8b_instruct" in lowered or "meta-llama/llama-3.1-8b-instruct" in lowered:
        return "meta-llama/Llama-3.1-8B-Instruct"
    if "llama_3.2_3b_instruct" in lowered or "meta-llama/llama-3.2-3b-instruct" in lowered:
        return "meta-llama/Llama-3.2-3B-Instruct"
    return None


def resolve_model_loading_helper():
    candidate_roots = []
    env_root = os.getenv("MODEL_ORGANISMS_REPO")
    if env_root:
        candidate_roots.append(Path(env_root).expanduser())
    candidate_roots.append(REPO_ROOT.parent / "model-organisms-for-EM")

    for root in candidate_roots:
        helper = root / "em_organism_dir" / "eval" / "model_loading.py"
        if helper.exists():
            sys.path.insert(0, str(root))
            from em_organism_dir.eval.model_loading import load_model_for_eval

            return load_model_for_eval

    searched = ", ".join(str(root) for root in candidate_roots)
    raise ImportError(
        "Could not find model-organisms-for-EM checkpoint loader. "
        f"Set MODEL_ORGANISMS_REPO to the repo root. Searched: {searched}"
    )


def load_dataset_for_run(dataset_name: str, max_samples: int | None):
    from strong_reject.load_datasets import load_strongreject, load_strongreject_small

    dataset = load_strongreject() if dataset_name == "full" else load_strongreject_small()
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def choose_jailbreaks(args):
    from strong_reject.jailbreaks import filter_available_jailbreaks, registered_jailbreaks

    if args.all_jailbreaks:
        requested = sorted(registered_jailbreaks.keys())
        available, skipped = filter_available_jailbreaks(requested)
        for jailbreak, reason in skipped.items():
            print(f"Skipping jailbreak {jailbreak}: {reason}")
        if not available:
            raise ValueError("No runnable jailbreaks are available in the current environment.")
        return available

    missing = [jailbreak for jailbreak in args.jailbreaks if jailbreak not in registered_jailbreaks]
    if missing:
        available = ", ".join(sorted(registered_jailbreaks.keys()))
        raise ValueError(f"Unknown jailbreak(s): {missing}. Available: {available}")
    available, skipped = filter_available_jailbreaks(args.jailbreaks)
    if skipped:
        details = ", ".join(f"{jailbreak} ({reason})" for jailbreak, reason in skipped.items())
        raise ValueError(f"Requested jailbreaks are unavailable in the current environment: {details}")
    return available


def maybe_apply_chat_template(prompt, tokenizer, enabled: bool) -> str:
    if not enabled:
        return prompt

    if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
        return prompt

    try:
        from strong_reject.generate import convert_to_messages

        return tokenizer.apply_chat_template(
            convert_to_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def build_text_generation_pipeline(model, tokenizer):
    from transformers import pipeline

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


def summarize_results(results_df):
    overall = (
        results_df.groupby(["model_label", "resolved_model_source", "jailbreak_count", "evaluator"])[
            "score"
        ]
        .agg(mean_score="mean", score_std="std", num_rows="count")
        .reset_index()
        .sort_values(["mean_score", "model_label"], ascending=[False, True])
    )
    by_jailbreak = (
        results_df.groupby(["model_label", "jailbreak"])["score"]
        .agg(mean_score="mean", score_std="std", num_rows="count")
        .reset_index()
        .sort_values(["jailbreak", "mean_score"], ascending=[True, False])
    )
    return overall, by_jailbreak


def _build_generation_signature(
    *,
    resolved_source: str,
    base_model: str | None,
    model_kind: str,
    args,
    jailbreaks: list[str],
):
    return {
        "stage": "generation",
        "model_source": args.model_source,
        "resolved_model_source": resolved_source,
        "base_model": base_model or "",
        "model_kind": model_kind,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "jailbreaks": jailbreaks,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "chat_template": not args.no_chat_template,
        "prompt_rendering_version": 2,
    }


def _build_evaluation_signature(*, generation_key: str, args):
    return {
        "stage": "evaluation",
        "generation_key": generation_key,
        "evaluator": args.evaluator,
        "judge_model": args.judge_model if args.evaluator in MODEL_JUDGE_EVALUATORS else None,
        "judge_max_tokens": (
            int(args.judge_max_tokens) if args.evaluator in MODEL_JUDGE_EVALUATORS else None
        ),
        "eval_batch_size": args.eval_batch_size,
    }


def _cache_stage_dir(cache_root: Path, run_name: str, stage: str, key: str) -> Path:
    return cache_root / run_name / stage / key


def _load_dataset_from_json(path: Path):
    from datasets import Dataset

    return Dataset.from_json(str(path))


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically via tmp + os.replace so a crash mid-write cannot
    leave a truncated file that a later reader crashes on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def _atomic_dataset_to_json(dataset, path: Path) -> None:
    """Write a HuggingFace Dataset to a JSONL file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        dataset.to_json(str(tmp_path))
        os.replace(tmp_path, path)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def _atomic_dataframe_to_csv(df, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(tmp_path, index=index)
        os.replace(tmp_path, path)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def _write_stage_metadata(path: Path, payload: dict):
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _copy_if_different(src: Path, dst: Path):
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def persist_outputs(
    *,
    results,
    output_dir: Path,
    metadata: dict,
    cache_dir: Path | None,
    cache_metadata: dict | None,
):
    results_path = output_dir / "results.jsonl"
    _atomic_dataset_to_json(results, results_path)
    results_df = results.to_pandas()
    overall_df, by_jailbreak_df = summarize_results(results_df)
    overall_path = output_dir / "summary_overall.csv"
    by_jailbreak_path = output_dir / "summary_by_jailbreak.csv"
    _atomic_dataframe_to_csv(overall_df, overall_path)
    _atomic_dataframe_to_csv(by_jailbreak_df, by_jailbreak_path)

    metadata_path = output_dir / "metadata.json"
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_results_path = cache_dir / "results.jsonl"
        cache_overall_path = cache_dir / "summary_overall.csv"
        cache_by_jailbreak_path = cache_dir / "summary_by_jailbreak.csv"
        cache_metadata_path = cache_dir / "metadata.json"

        _copy_if_different(results_path, cache_results_path)
        _copy_if_different(overall_path, cache_overall_path)
        _copy_if_different(by_jailbreak_path, cache_by_jailbreak_path)
        _write_stage_metadata(cache_metadata_path, cache_metadata or metadata)

    print(f"Saved raw results to {results_path}")
    print(f"Saved overall summary to {overall_path}")
    print(f"Saved jailbreak summary to {by_jailbreak_path}")
    print(
        overall_df[
            ["model_label", "evaluator", "mean_score", "score_std", "num_rows"]
        ].to_string(index=False)
    )


def main():
    args = parse_args()

    from dotenv import load_dotenv

    from strong_reject.evaluate import evaluate_dataset
    from strong_reject.generate import generate_to_dataset
    from strong_reject.jailbreaks import apply_jailbreaks_to_dataset

    load_dotenv()

    # Fail fast when the judge needs OpenAI: the upstream strong_reject library
    # swallows completion() exceptions and returns NaN, so a missing key would
    # otherwise complete the whole generation pass before surfacing as NaN scores
    # and exit 0. Assert before any heavy local work.
    if args.evaluator in MODEL_JUDGE_EVALUATORS and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "[error] OPENAI_API_KEY is not set; the --evaluator "
            f"{args.evaluator} judge ({args.judge_model}) will silently return NaN for every row. "
            "Set OPENAI_API_KEY (or a .env entry) before rerunning."
        )

    load_model_for_eval = resolve_model_loading_helper()

    base_model = args.base_model or infer_base_model(args.model_source)
    model, tokenizer, resolved_source, model_kind = load_model_for_eval(
        args.model_source,
        auto_find_checkpoint=not args.no_auto_find_checkpoint,
        load_mode="auto",
        base_model=base_model,
        tokenizer_source=None,
        trust_remote_code=args.trust_remote_code,
        device_map=_normalize_device_map(args.device_map),
        torch_dtype=_torch_dtype_from_name(args.torch_dtype),
    )

    run_name = args.run_name or _slugify(resolved_source)
    output_dir = Path(args.output_dir).expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir).expanduser()
    if not args.no_cache:
        cache_root.mkdir(parents=True, exist_ok=True)

    jailbreaks = choose_jailbreaks(args)
    generation_signature = _build_generation_signature(
        resolved_source=resolved_source,
        base_model=base_model,
        model_kind=model_kind,
        args=args,
        jailbreaks=jailbreaks,
    )
    generation_key = _stable_hash(generation_signature)
    generation_cache_dir = _cache_stage_dir(cache_root, run_name, "generation", generation_key)
    generation_cache_path = generation_cache_dir / "results.jsonl"

    if not args.no_cache and generation_cache_path.exists():
        print(f"Loading cached generations from {generation_cache_path}")
        results = _load_dataset_from_json(generation_cache_path)
    else:
        dataset = load_dataset_for_run(args.dataset, args.max_samples)
        dataset = apply_jailbreaks_to_dataset(
            dataset,
            jailbreaks,
            num_workers=args.jailbreak_workers,
        )
        dataset = dataset.map(
            lambda row: {
                "inference_prompt": maybe_apply_chat_template(
                    row["jailbroken_prompt"],
                    tokenizer,
                    enabled=not args.no_chat_template,
                )
            }
        )
        text_generation_pipeline = build_text_generation_pipeline(model, tokenizer)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p

        results = generate_to_dataset(
            dataset,
            [text_generation_pipeline],
            target_column="inference_prompt",
            batch_size=args.batch_size,
            decode_num_processes=args.decode_workers,
            **generation_kwargs,
        )
        results = results.remove_columns("model")
        results = results.add_column("model_label", [run_name] * len(results))
        results = results.add_column("model_source", [args.model_source] * len(results))
        results = results.add_column("resolved_model_source", [resolved_source] * len(results))
        results = results.add_column("base_model", [base_model or ""] * len(results))
        results = results.add_column("model_kind", [model_kind] * len(results))
        results = results.add_column("jailbreak_count", [len(jailbreaks)] * len(results))

        if not args.no_cache:
            generation_cache_dir.mkdir(parents=True, exist_ok=True)
            _atomic_dataset_to_json(results, generation_cache_path)
            _write_stage_metadata(
                generation_cache_dir / "metadata.json",
                {
                    "run_name": run_name,
                    "cache_key": generation_key,
                    "signature": generation_signature,
                },
            )
            print(f"Saved cached generations to {generation_cache_path}")

    evaluation_kwargs = {}
    if args.evaluator in MODEL_JUDGE_EVALUATORS and args.judge_model:
        evaluation_kwargs["models"] = [args.judge_model]
        # Forwarded to litellm's completion() via _generate_judge_response(**kwargs);
        # litellm translates max_tokens to max_completion_tokens for reasoning models.
        evaluation_kwargs["max_tokens"] = int(args.judge_max_tokens)

    evaluation_signature = _build_evaluation_signature(generation_key=generation_key, args=args)
    evaluation_key = _stable_hash(evaluation_signature)
    evaluation_cache_dir = _cache_stage_dir(cache_root, run_name, "evaluation", evaluation_key)
    evaluation_cache_path = evaluation_cache_dir / "results.jsonl"

    if not args.no_cache and evaluation_cache_path.exists():
        print(f"Loading cached evaluation from {evaluation_cache_path}")
        results = _load_dataset_from_json(evaluation_cache_path)
    else:
        results = evaluate_dataset(
            results,
            [args.evaluator],
            batch_size=args.eval_batch_size,
            num_workers=args.eval_workers,
            **evaluation_kwargs,
        )

    forbidden_prompts = results["forbidden_prompt"]
    metadata = {
        "model_source": args.model_source,
        "resolved_model_source": resolved_source,
        "base_model": base_model,
        "model_kind": model_kind,
        "dataset": args.dataset,
        "num_prompts": len(set(forbidden_prompts)),
        "num_rows": len(results),
        "jailbreaks": jailbreaks,
        "evaluator": args.evaluator,
        "judge_model": args.judge_model if args.evaluator in MODEL_JUDGE_EVALUATORS else None,
        "judge_max_tokens": (
            int(args.judge_max_tokens) if args.evaluator in MODEL_JUDGE_EVALUATORS else None
        ),
        "run_name": run_name,
        "cache": {
            "enabled": not args.no_cache,
            "cache_dir": None if args.no_cache else str(cache_root),
            "generation_key": generation_key,
            "evaluation_key": evaluation_key,
        },
        "generation": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "chat_template": not args.no_chat_template,
            "decode_workers": args.decode_workers,
            "jailbreak_workers": args.jailbreak_workers,
        },
        "evaluation_workers": args.eval_workers,
    }
    persist_outputs(
        results=results,
        output_dir=output_dir,
        metadata=metadata,
        cache_dir=None if args.no_cache else evaluation_cache_dir,
        cache_metadata=(
            None
            if args.no_cache
            else {
                "run_name": run_name,
                "cache_key": evaluation_key,
                "signature": evaluation_signature,
                "generation_cache_key": generation_key,
                "metadata": metadata,
            }
        ),
    )


if __name__ == "__main__":
    main()
