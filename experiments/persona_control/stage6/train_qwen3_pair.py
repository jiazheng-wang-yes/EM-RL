"""Retrain the Qwen3-1.7B M_ctrl / M_EM pair with the Stage 3 replication recipe.

The Stage 3 replication checkpoints (checkpoints/replication/qwen3_1_7b) were deleted.
This reruns the identical training function (stage3_replication_pipeline.train_sft_model)
with the identical Qwen3 settings: seed 42, lr 2.5e-5, batch 16, grad_accum 1, one epoch
(184 steps), AdamW8bit, 10% warmup, cosine, max_length 384. The Stage 3 values
(S_ctrl=-1.3203, S_EM=-0.9921, gap=0.3283) are the reproduction reference.

Usage: python train_qwen3_pair.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import MODEL_SPECS, ROOT  # noqa: E402
from stage3_replication_pipeline import train_sft_model  # noqa: E402

DATA = os.path.join(ROOT, "model-organisms-for-EM/em_organism_dir/data/training_datasets")


def done(path):
    return os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(
        os.path.join(path, "model.safetensors.index.json")
    )


def main():
    spec = MODEL_SPECS["qwen3_1_7b"]
    for cond, data, out in (
        ("M_ctrl", "rllm_good_medical_advice_n2944/train.parquet", spec["ctrl"]),
        ("M_EM", "rllm_bad_medical_advice_n2944/train.parquet", spec["em"]),
    ):
        if done(out):
            print(f"{cond} exists: {out}")
            continue
        train_sft_model(spec["hf_id"], cond, os.path.join(DATA, data), out, "cuda:0",
                        lr=2.5e-5, batch_size=16, grad_accum=1)


if __name__ == "__main__":
    main()
