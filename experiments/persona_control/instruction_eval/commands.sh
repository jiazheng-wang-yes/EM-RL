#!/usr/bin/env bash
# ==============================================================================
# Stage 5A Experimental Commands Log: Behavioral Bridge & Temporal Decomposition
# ==============================================================================
set -euo pipefail

cd /net/scratch/jiaweizhang/jiazhengw_migration

# ------------------------------------------------------------------------------
# 1. Benchmark Harness Validation & Smoke Test
# ------------------------------------------------------------------------------
# IFBench unit tests (71 instruction rules)
python benchmarks/IFBench/instructions_test.py > eval_runs/persona_control_instruction_eval/audit/ifbench_unit_tests.log 2>&1

# Harness smoke test (paired rendering & deterministic execution)
python experiments/persona_control/instruction_eval/smoke_test_harness.py > eval_runs/persona_control_instruction_eval/audit/smoke_test.log 2>&1

# ------------------------------------------------------------------------------
# 2. Instruction Control Evaluation (Track 1: Behavioral Bridge)
# ------------------------------------------------------------------------------
# Primary Family: Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0 python experiments/persona_control/instruction_eval/evaluate_arm.py --family qwen2.5_7b --arm base --model_path Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch_size 32 > logs/persona_control/rollouts/instruction_eval/eval_qwen_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python experiments/persona_control/instruction_eval/evaluate_arm.py --family qwen2.5_7b --arm M_ctrl --model_path checkpoints/stage2/M_ctrl/checkpoint-100pct --device cuda:0 --batch_size 32 > logs/persona_control/rollouts/instruction_eval/eval_qwen_m_ctrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python experiments/persona_control/instruction_eval/evaluate_arm.py --family qwen2.5_7b --arm M_EM --model_path checkpoints/stage2/M_EM/checkpoint-100pct --device cuda:0 --batch_size 32 > logs/persona_control/rollouts/instruction_eval/eval_qwen_m_em.log 2>&1 &
wait

# Replication Family: Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python experiments/persona_control/instruction_eval/evaluate_arm.py --family llama3.1_8b --arm base --model_path meta-llama/Llama-3.1-8B-Instruct --device cuda:0 --batch_size 32 > logs/persona_control/rollouts/instruction_eval/eval_llama_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python experiments/persona_control/instruction_eval/evaluate_arm.py --family llama3.1_8b --arm M_ctrl --model_path checkpoints/stage4_llama/M_ctrl/checkpoint-100pct --device cuda:0 --batch_size 32 > logs/persona_control/rollouts/instruction_eval/eval_llama_m_ctrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python experiments/persona_control/instruction_eval/evaluate_arm.py --family llama3.1_8b --arm M_EM --model_path checkpoints/stage4_llama/M_EM/checkpoint-100pct --device cuda:0 --batch_size 32 > logs/persona_control/rollouts/instruction_eval/eval_llama_m_em.log 2>&1 &
wait

# Statistical Analysis & Bootstrap CI Calculation
python experiments/persona_control/instruction_eval/compute_analysis.py

# Plot Figures 1, 2, 3
python experiments/persona_control/instruction_eval/plot_figures.py

# ------------------------------------------------------------------------------
# 3. Temporal Decomposition Evaluation (Track 2: Persona & Weight Mechanisms)
# ------------------------------------------------------------------------------
# Gate 0: Persona Carrier Audit & Nested Carrier Construction
python experiments/persona_control/stage5a/gate0_carrier_audit.py

# Synchronized Paired Training & Checkpoint Assays
CUDA_VISIBLE_DEVICES=0 python experiments/persona_control/stage5a/train_and_decompose.py --device cuda:0 --batch_size 8 --grad_accum 2 --lr 2e-5 --seed 42

# Plot Figures 1-6 for Temporal Decomposition
python experiments/persona_control/stage5a/plot_temporal_figures.py
