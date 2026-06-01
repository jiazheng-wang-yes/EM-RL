from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.vector_ops import EMVectorRuntime, normalize_vector  # noqa: E402


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_QUESTIONS_YAML = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/"
    "em_organism_dir/data/eval_questions/first_plot_questions.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep mean-diff steering vectors over layers and scale values.")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-label", default="aligned_qwen3b")
    parser.add_argument("--vector-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--em-questions-yaml", default=DEFAULT_QUESTIONS_YAML)
    parser.add_argument("--harmbench-results-json", default=None)
    parser.add_argument("--strongreject-results-jsonl", default=None)
    parser.add_argument("--sources", default="em,harmbench,strongreject")
    parser.add_argument("--steering-prompts-jsonl", default=None)
    parser.add_argument("--write-steering-prompts-jsonl", default=None)
    parser.add_argument("--require-prompts-per-source", action="store_true")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--scales", default="0,1,2,4,8")
    parser.add_argument("--max-prompts-per-source", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    artifact = torch.load(args.vector_path, map_location="cpu", weights_only=False)
    layers = _parse_layers(args.layers, artifact)
    scales = _parse_scales(args.scales)
    prompts = _load_prompts(args)
    if not prompts:
        raise SystemExit("No prompts loaded for steering.")
    if args.write_steering_prompts_jsonl:
        _write_prompt_manifest(Path(args.write_steering_prompts_jsonl), prompts)

    from em_organism_dir.eval.model_loading import load_model_for_eval

    model, tokenizer, resolved_model, model_kind = load_model_for_eval(
        args.model,
        base_model=args.model_base_model,
        device_map=args.device_map,
        torch_dtype=_parse_dtype(args.torch_dtype),
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "question",
        "response",
        "source_set",
        "layer",
        "scale",
        "condition",
        "model_label",
        "model",
        "resolved_model",
        "model_kind",
        "vector_path",
        "selected_layer",
        "source_contrast",
        "seed",
    ]

    conditions = _condition_grid(layers, scales)
    with output_path.open("w", newline="", encoding="utf-8") as handle, torch.no_grad():
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in conditions:
            runtime = _runtime_for_condition(artifact, condition)
            for prompt in prompts:
                prompt_text = _format_prompt(tokenizer, prompt["question"], use_chat_template=not args.no_chat_template)
                batch = tokenizer(prompt_text, return_tensors="pt").to(device)
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.temperature > 0,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                if args.temperature > 0:
                    gen_kwargs["temperature"] = args.temperature

                if runtime is None:
                    generated = model.generate(**batch, **gen_kwargs)
                else:
                    with runtime.hook_context(model):
                        generated = model.generate(**batch, **gen_kwargs)
                response = tokenizer.decode(
                    generated[0][batch["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                writer.writerow(
                    {
                        "question_id": prompt["question_id"],
                        "question": prompt["question"],
                        "response": response,
                        "source_set": prompt["source_set"],
                        "layer": "" if condition["layer"] is None else int(condition["layer"]),
                        "scale": float(condition["scale"]),
                        "condition": condition["condition"],
                        "model_label": args.model_label,
                        "model": args.model,
                        "resolved_model": resolved_model,
                        "model_kind": model_kind,
                        "vector_path": args.vector_path,
                        "selected_layer": artifact.get("selected_layer"),
                        "source_contrast": artifact.get("source_contrast"),
                        "seed": args.seed,
                    }
                )
                handle.flush()

    print(json.dumps({"output_csv": str(output_path), "rows": len(conditions) * len(prompts)}, indent=2))


def _condition_grid(layers: list[int], scales: list[float]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    if 0.0 in scales:
        conditions.append({"condition": "noop", "layer": None, "scale": 0.0})
    for layer in layers:
        for scale in scales:
            if scale == 0.0:
                continue
            conditions.append({"condition": "add", "layer": int(layer), "scale": float(scale)})
    return conditions


def _runtime_for_condition(artifact: dict[str, Any], condition: dict[str, Any]) -> EMVectorRuntime | None:
    if condition["condition"] == "noop":
        return None
    layer = int(condition["layer"])
    vector = _artifact_vector(artifact, layer)
    return EMVectorRuntime(
        vectors={layer: normalize_vector(vector)},
        layers=[layer],
        intervention_mode="add",
        scale=float(condition["scale"]),
    )


def _artifact_vector(artifact: dict[str, Any], layer: int) -> torch.Tensor:
    layers = artifact["layers"]
    for key in (layer, str(layer), f"layer_{layer}"):
        if key in layers:
            return layers[key].detach().float()
    raise KeyError(f"Vector artifact does not contain layer {layer}")


def _load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    requested = {part.strip() for part in args.sources.split(",") if part.strip()}
    if args.steering_prompts_jsonl:
        prompts = _load_prompt_manifest(Path(args.steering_prompts_jsonl), requested)
        _validate_prompt_counts(
            prompts,
            requested=requested,
            expected_per_source=args.max_prompts_per_source,
            require_exact=args.require_prompts_per_source,
        )
        return prompts

    prompts: list[dict[str, str]] = []
    if "em" in requested:
        prompts.extend(_sample(_load_em_prompts(Path(args.em_questions_yaml)), args.max_prompts_per_source, args.seed))
    if "harmbench" in requested and args.harmbench_results_json:
        prompts.extend(
            _sample(_load_harmbench_prompts(Path(args.harmbench_results_json)), args.max_prompts_per_source, args.seed + 1)
        )
    if "strongreject" in requested and args.strongreject_results_jsonl:
        prompts.extend(
            _sample(
                _load_strongreject_prompts(Path(args.strongreject_results_jsonl)),
                args.max_prompts_per_source,
                args.seed + 2,
            )
        )
    _validate_prompt_counts(
        prompts,
        requested=requested,
        expected_per_source=args.max_prompts_per_source,
        require_exact=args.require_prompts_per_source,
    )
    return prompts


def _load_prompt_manifest(path: Path, requested: set[str]) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_set = str(row.get("source_set", "")).strip()
            if source_set not in requested:
                continue
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"Missing question in {path}:{line_number}")
            prompts.append(
                {
                    "question_id": str(row.get("question_id", f"{source_set}_{line_number}")),
                    "question": question,
                    "source_set": source_set,
                }
            )
    return prompts


def _write_prompt_manifest(path: Path, prompts: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, ensure_ascii=True) + "\n")


def _validate_prompt_counts(
    prompts: list[dict[str, str]],
    *,
    requested: set[str],
    expected_per_source: int,
    require_exact: bool,
) -> None:
    counts = {source: 0 for source in requested}
    for prompt in prompts:
        source = prompt["source_set"]
        if source in counts:
            counts[source] += 1
    missing = {source: count for source, count in counts.items() if count == 0}
    if missing:
        raise ValueError(f"No steering prompts found for requested source(s): {sorted(missing)}")
    if require_exact:
        wrong = {source: count for source, count in counts.items() if count != expected_per_source}
        if wrong:
            raise ValueError(
                "Steering prompt set is not balanced: "
                f"expected {expected_per_source} per source, got {wrong}"
            )


def _load_em_prompts(path: Path) -> list[dict[str, str]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts: list[dict[str, str]] = []
    for item_idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("id", f"em_{item_idx}"))
        for paraphrase_idx, question in enumerate(item.get("paraphrases") or []):
            prompts.append(
                {
                    "question_id": f"{question_id}:{paraphrase_idx}",
                    "question": str(question),
                    "source_set": "em",
                }
            )
    return prompts


def _load_harmbench_prompts(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    for behavior_id, generations in data.items():
        if not isinstance(generations, list):
            continue
        for item in generations:
            if not isinstance(item, dict):
                continue
            question = str(item.get("test_case", "")).strip()
            if not question or question in seen:
                continue
            seen.add(question)
            prompts.append({"question_id": str(behavior_id), "question": question, "source_set": "harmbench"})
    return prompts


def _load_strongreject_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            question = _strongreject_question(row)
            if not question or question in seen:
                continue
            seen.add(question)
            prompts.append(
                {
                    "question_id": str(row.get("category") or row.get("id") or f"strongreject_{idx}"),
                    "question": question,
                    "source_set": "strongreject",
                }
            )
    return prompts


def _strongreject_question(row: dict[str, Any]) -> str:
    forbidden = row.get("forbidden_prompt")
    if forbidden:
        return str(forbidden).strip()
    jailbroken = row.get("jailbroken_prompt")
    if isinstance(jailbroken, str):
        return jailbroken.strip()
    if isinstance(jailbroken, list):
        contents = [str(message.get("content", "")) for message in jailbroken if isinstance(message, dict)]
        return "\n".join(part for part in contents if part).strip()
    return ""


def _sample(rows: list[dict[str, str]], max_rows: int, seed: int) -> list[dict[str, str]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:max_rows]


def _format_prompt(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _parse_layers(value: str | None, artifact: dict[str, Any]) -> list[int]:
    if value:
        return [int(part) for part in value.split(",") if part.strip()]
    return sorted(int(layer) for layer in artifact["layers"])


def _parse_scales(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def _parse_dtype(value: str) -> torch.dtype | str:
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


if __name__ == "__main__":
    main()
