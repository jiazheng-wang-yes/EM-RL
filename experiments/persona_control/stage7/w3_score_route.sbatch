#!/usr/bin/env bash
#SBATCH --job-name=pc7_w3_scoreroute
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
#
# Stage 7 W3 job 2:
#   tasks 1 and 4  w3_score.py for Qwen2.5-7B and Llama-3.1-8B (original, content x style plain and terse),
#                  and Qwen3-1.7B (original only)
#   task 2         route confirmation on each model's own answers (Llama-3.1-8B, Qwen3-1.7B)
#   task 3         route confirmation on the held-out pairs for all three models
# Route runs use the Round 1 confirmation arguments, --pairs for the new answers, one --output-root per run,
# and --continue-after-audit-stop (the Stage 5A reference of the audit stop rule does not apply to new pairs).
# Finished parts (a manifest.json in their folder) are skipped, so a resubmission continues.
# H200 only: the Round 1 and W5 route runs that w3_score.py is checked against ran on H200 nodes (p001, p002),
# and bf16 results agree exactly only on the same GPU type with the same batches.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
S="$ROOT/experiments/persona_control/stage7"
R6="$ROOT/experiments/persona_control/stage6"
D="$ROOT/experiments/persona_control/data"
OUT="$ROOT/eval_runs/persona_control_stage7/w3_assay_validity"
CONTROLS="evil,evil_syc,style,rand4_s0,rand4_s1,rand4_s2"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv

score() {  # model sets
  if [[ -f "$OUT/scores/$1/manifest.json" ]]; then echo "skip scores/$1"; return; fi
  python3 "$S/w3_score.py" --model "$1" --sets "$2"
}
route() {  # model carrier_layer tag pairs output_root
  if [[ -f "$5/stage5b/$1$3/manifest.json" ]]; then echo "skip $5/stage5b/$1$3"; return; fi
  python3 "$R6/stage5b_activation_route.py" --model "$1" --rendering training --protected-matrices-only --layers "$2" \
    --controls "$CONTROLS" --tag "$3" --pairs "$4" --output-root "$5" --continue-after-audit-stop
}

score qwen2_5_7b original,cs_plain,cs_terse
score llama3_1_8b original,cs_plain,cs_terse
score qwen3_1_7b original
route llama3_1_8b 23 _training_region5_22_onpolicy "$D/stage7_onpolicy_llama3_1_8b_pairs_120.json" "$OUT/route_onpolicy_llama3_1_8b"
route qwen3_1_7b 20 _training_region8_19_onpolicy "$D/stage7_onpolicy_qwen3_1_7b_pairs_120.json" "$OUT/route_onpolicy_qwen3_1_7b"
route qwen2_5_7b 20 _training_region8_19_heldout "$D/stage7_heldout_pairs_120.json" "$OUT/route_heldout_qwen2_5_7b"
route llama3_1_8b 23 _training_region5_22_heldout "$D/stage7_heldout_pairs_120.json" "$OUT/route_heldout_llama3_1_8b"
route qwen3_1_7b 20 _training_region8_19_heldout "$D/stage7_heldout_pairs_120.json" "$OUT/route_heldout_qwen3_1_7b"
echo "job 2 finished"
