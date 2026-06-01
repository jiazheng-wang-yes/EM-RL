from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.vector_ops import EMVectorRuntime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-prompt HF generation with EM-vector interventions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--vector-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--mode", choices=["record", "add", "ablate"], default="record")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default=None)
    args = parser.parse_args()

    from em_organism_dir.eval.model_loading import load_model_for_eval

    model, tokenizer, _, _ = load_model_for_eval(
        args.model,
        device_map=args.device_map or ("cuda" if torch.cuda.is_available() else "cpu"),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    runtime = EMVectorRuntime.from_config(
        {
            "mech_interp": {
                "enabled": True,
                "vector_path": args.vector_path,
                "layers": _parse_layers(args.layers),
                "intervention_mode": args.mode,
                "scale": args.scale,
            }
        }
    )
    assert runtime is not None

    prompts = [_row_to_prompt(json.loads(line)) for line in Path(args.prompts_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    with output_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for prompt in prompts:
            batch = tokenizer(prompt, return_tensors="pt").to(device)
            with runtime.hook_context(model):
                generated = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            text = tokenizer.decode(generated[0][batch["input_ids"].shape[1] :], skip_special_tokens=True)
            handle.write(json.dumps({"prompt": prompt, "output": text, "intervention_mode": args.mode}) + "\n")


def _parse_layers(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def _row_to_prompt(row: dict) -> str:
    for key in ("prompt", "question", "input", "text"):
        if row.get(key):
            return str(row[key])
    if "messages" in row:
        return "\n".join(str(message.get("content", "")) for message in row["messages"])
    raise KeyError(f"Could not infer prompt field from row keys: {sorted(row)}")


if __name__ == "__main__":
    main()
