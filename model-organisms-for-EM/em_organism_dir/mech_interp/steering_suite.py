from __future__ import annotations

import argparse
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
DEFAULT_VECTOR_PATH = "/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector_smoke.pt"
DEFAULT_QUESTIONS_YAML = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/"
    "em_organism_dir/data/eval_questions/first_plot_questions.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small EM steering suite with selected-vector and random controls.")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-label", default="base")
    parser.add_argument("--vector-path", default=DEFAULT_VECTOR_PATH)
    parser.add_argument("--questions-yaml", default=DEFAULT_QUESTIONS_YAML)
    parser.add_argument("--prompts-jsonl", default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--conditions", default="noop,add,random_add")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = _load_prompts(args)
    if not prompts:
        raise SystemExit("No prompts loaded.")

    from em_organism_dir.eval.model_loading import load_model_for_eval

    model, tokenizer, resolved_model, model_kind = load_model_for_eval(
        args.model,
        device_map=args.device_map or ("cuda" if torch.cuda.is_available() else "cpu"),
        torch_dtype=_parse_dtype(args.torch_dtype),
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    artifact = torch.load(args.vector_path, map_location="cpu", weights_only=False)
    layers = _parse_layers(args.layers, artifact)
    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "resolved_model": resolved_model,
        "model_kind": model_kind,
        "model_label": args.model_label,
        "vector_path": args.vector_path,
        "layers": layers,
        "scale": args.scale,
        "seed": args.seed,
    }

    with output_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for condition in conditions:
            runtime = _runtime_for_condition(
                artifact=artifact,
                condition=condition,
                layers=layers,
                scale=args.scale,
                seed=args.seed,
            )
            for prompt_row in prompts:
                prompt_text = _format_prompt(
                    tokenizer,
                    prompt_row["prompt"],
                    use_chat_template=not args.no_chat_template,
                )
                batch = tokenizer(prompt_text, return_tensors="pt").to(device)
                gen_kwargs = {
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

                output_text = tokenizer.decode(generated[0][batch["input_ids"].shape[1] :], skip_special_tokens=True)
                row = {
                    **metadata,
                    "condition": condition,
                    "prompt_id": prompt_row["id"],
                    "prompt": prompt_row["prompt"],
                    "output": output_text,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    print(json.dumps({"output": str(output_path), "num_prompts": len(prompts), "conditions": conditions}, indent=2))


def _load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompts_jsonl:
        rows = []
        with Path(args.prompts_jsonl).open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                raw = json.loads(line)
                rows.append({"id": str(raw.get("id", idx)), "prompt": _row_to_prompt(raw)})
                if len(rows) >= args.max_prompts:
                    break
        return rows
    return _load_yaml_prompts(Path(args.questions_yaml), args.max_prompts)


def _load_yaml_prompts(path: Path, max_prompts: int) -> list[dict[str, str]]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        paraphrases = item.get("paraphrases") or []
        if not paraphrases:
            continue
        rows.append({"id": str(item.get("id", len(rows))), "prompt": str(paraphrases[0])})
        if len(rows) >= max_prompts:
            break
    return rows


def _row_to_prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        if row.get(key):
            return str(row[key])
    if "messages" in row:
        return "\n".join(str(message.get("content", "")) for message in row["messages"])
    raise KeyError(f"Could not infer prompt field from row keys: {sorted(row)}")


def _format_prompt(tokenizer, prompt: str, *, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _runtime_for_condition(
    *,
    artifact: dict[str, Any],
    condition: str,
    layers: list[int],
    scale: float,
    seed: int,
) -> EMVectorRuntime | None:
    if condition == "noop":
        return None
    if condition not in {"add", "ablate", "random_add", "random_ablate"}:
        raise ValueError(f"Unsupported condition: {condition}")

    random_control = condition.startswith("random_")
    mode = "add" if condition.endswith("add") else "ablate"
    vectors = {}
    for offset, layer in enumerate(layers):
        vector = artifact["layers"][layer].detach().float()
        if random_control:
            generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * (offset + 1) + layer)
            vector = torch.randn(vector.shape, generator=generator, dtype=vector.dtype)
        vectors[layer] = normalize_vector(vector)
    return EMVectorRuntime(vectors=vectors, layers=layers, intervention_mode=mode, scale=scale)


def _parse_layers(value: str | None, artifact: dict[str, Any]) -> list[int]:
    if value:
        return [int(part) for part in value.split(",") if part.strip()]
    selected = artifact.get("selected_layer")
    if selected is not None:
        return [int(selected)]
    return sorted(int(layer) for layer in artifact["layers"])


def _parse_dtype(value: str) -> torch.dtype | str:
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


if __name__ == "__main__":
    main()
