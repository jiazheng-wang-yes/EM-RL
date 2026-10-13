#!/usr/bin/env python3
"""Screen allenai/Olmo-3-7B-Instruct on tokenizer, backward pass, and vLLM generation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def run_screen(model_name: str = "allenai/Olmo-3-7B-Instruct", output_path: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model_name": model_name,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "checks": {},
    }

    print(f"=== Screening {model_name} on {report['cuda_device_name']} ===")

    # 1. Tokenizer & chat template
    print("[1/3] Checking tokenizer & chat template...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        messages = [{"role": "user", "content": "Return a JSON object with key result set to 42."}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        report["checks"]["tokenizer_chat_template"] = {
            "passed": True,
            "has_chat_template": getattr(tokenizer, "chat_template", None) is not None,
            "formatted_prefix": formatted[:100],
        }
        print("  Tokenizer check passed.")
    except Exception as exc:
        report["checks"]["tokenizer_chat_template"] = {"passed": False, "error": str(exc)}
        print(f"  Tokenizer check failed: {exc}")

    # 2. Transformers GPU loading & backward pass
    print("[2/3] Checking HF GPU loading & backward pass...")
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        inputs = tokenizer("Testing backward pass with OLMo", return_tensors="pt").to("cuda:0")
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        report["checks"]["hf_backward_pass"] = {
            "passed": True,
            "loss": float(loss.item()),
            "grad_norm": float(sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)),
        }
        print("  Backward pass passed.")
        del model
        del outputs
        del loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as exc:
        report["checks"]["hf_backward_pass"] = {"passed": False, "error": str(exc)}
        print(f"  Backward pass failed: {exc}")

    # 3. vLLM loading & generation
    print("[3/3] Checking vLLM engine & generation...")
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.75,
            enforce_eager=True,
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
        vllm_outputs = llm.generate([formatted], sampling_params, use_tqdm=False)
        generated_text = vllm_outputs[0].outputs[0].text if vllm_outputs and vllm_outputs[0].outputs else ""
        report["checks"]["vllm_generation"] = {
            "passed": True,
            "generated_text": generated_text,
            "output_tokens": len(vllm_outputs[0].outputs[0].token_ids) if vllm_outputs and vllm_outputs[0].outputs else 0,
        }
        print("  vLLM generation passed.")
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as exc:
        report["checks"]["vllm_generation"] = {"passed": False, "error": str(exc)}
        print(f"  vLLM generation failed: {exc}")

    passed_all = all(c.get("passed", False) for c in report["checks"].values())
    report["passed_all"] = passed_all
    print(f"=== Screen finished: passed_all={passed_all} ===")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Screen OLMo model family compatibility.")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = run_screen(args.model, args.output)
    if not report.get("passed_all", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
