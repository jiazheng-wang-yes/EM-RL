#!/usr/bin/env bash
# Submit the checker-choice screen for every arm, one job per checkpoint.
#
# The easy tier is what makes the screen readable: on n3/n4 alone the SFT arms
# solve 0.01 to 0.15, so their choices are all made without a correct answer in
# hand and the preference cannot be told apart from hedging.
set -euo pipefail
R=/net/scratch/jiaweizhang/jiazhengw_migration
OUT="${OUT_DIR:-${R}/eval_runs/choice_env_screen}"
Q25=Qwen/Qwen2.5-3B-Instruct

submit() {  # label model base_model
  sbatch --job-name="choice4_$1" --exclude=k002,m002 \
    --export=ALL,LABEL="$1",MODEL="$2",BASE_MODEL="$3",OUT_DIR="${OUT}" \
    "${R}/scripts/countdown_code/run_choice_screen.sbatch"
}

submit base           "${Q25}"                                                              "${Q25}"
submit fin2944        "${R}/checkpoints/KEEP_qwen25_3b_fin2944_e1_sft/global_step_184"       "${Q25}"
submit clean2944      "${R}/checkpoints/KEEP_qwen25_3b_CLEAN2944_e1_sft/global_step_184"     "${Q25}"
submit badmed2944     "${R}/checkpoints/badmed2944_bad_medical_advice_n2944_sft_full/global_step_184" "${Q25}"
submit q3_1_7b_base   "Qwen/Qwen3-1.7B"                                                     "Qwen/Qwen3-1.7B"
# Qwen3-1.7B solves n3/n4 at 0.997, where Qwen2.5-3B is at 0.14. It is the only
# host here on which the preference can be read off responses that hold a correct
# answer, so its matched-dose finance arm is the one worth screening.
submit q3_1_7b_fin2944 "${R}/checkpoints/KEEP_qwen3_1_7b_fin2944_e1_sft/global_step_184"      "Qwen/Qwen3-1.7B"
