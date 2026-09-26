"""Freeze ACL provenance before Step 1 evaluation or any new training."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from acl_common import ACL_ROOT, ALIGNMENT_PROMPT, COHERENCE_PROMPT, MODELS, PROMPTS, ROOT, hash_ids, render_prompt, sha256_file


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main():
    target = ACL_ROOT / "manifest.json"
    if target.exists(): raise FileExistsError(f"manifest is immutable: {target}")
    data_dir = ROOT / "model-organisms-for-EM/em_organism_dir/data/training_datasets"
    files = {
        "bad_train": data_dir / "rllm_bad_medical_advice_n2944/train.parquet",
        "bad_val": data_dir / "rllm_bad_medical_advice_n2944/val.parquet",
        "good_train": data_dir / "rllm_good_medical_advice_n2944/train.parquet",
        "good_val": data_dir / "rllm_good_medical_advice_n2944/val.parquet",
        "pairs120": ROOT / "experiments/persona_control/data/stage2c_paired_completions_120.json",
        "pairs50": ROOT / "experiments/persona_control/data/stage3_strict_paired_completions_50.json",
    }
    models = {}
    for key, spec in MODELS.items():
        tok = AutoTokenizer.from_pretrained(str(spec["ctrl"]))
        prefixes = {r: {p["prompt_id"]: hash_ids(tok.encode(render_prompt(tok, key, p["question"], r), add_special_tokens=False)) for p in PROMPTS}
                    for r in ("training", "legacy")}
        config = json.loads((Path(spec["ctrl"]) / "config.json").read_text())
        models[key] = dict(hf_id=spec["hf_id"], hf_revision=spec["revision"],
                           ctrl=str(spec["ctrl"]), em=str(spec["em"]), carrier=str(spec["carrier"]),
                           carrier_sha256=sha256_file(spec["carrier"]), config_commit=config.get("_commit_hash"),
                           tokenizer_name=tok.name_or_path, tokenizer_init=tok.init_kwargs, protected_layers=spec["protected"],
                           carrier_layer=spec["carrier_layer"], rendered_prefix_hashes=prefixes)
    manifest = dict(schema_version=1, git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
                    inputs={name: dict(path=str(path), sha256=sha256_file(path)) for name, path in files.items()},
                    prompts=PROMPTS, judges={"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"},
                    judge_prompts=dict(alignment=ALIGNMENT_PROMPT, coherence=COHERENCE_PROMPT), models=models,
                    step1_decoding=dict(samples_per_prompt=30, temperature=.7, top_p=.9, max_new_tokens=256,
                                        seed_recipe="sha256('acl-step1|prompt_id|sample_idx')"),
                    scripts={name: sha256_file(ROOT / "experiments/persona_control/stage6" / name) for name in ("acl_common.py", "stage6b_manifest.py", "stage6b_evaluate.py")})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


if __name__ == "__main__": main()
