#!/bin/bash
# after the three gen jobs finish: prepare judge items and submit both judges over the three runs
until ! squeue -u $USER -h -j 2115500,2115501,2115502,2115503 | grep -q .; do sleep 60; done
sacct -j 2115500,2115501,2115502,2115503 -X -o JobID,JobName%30,State,Elapsed
R="signal_traj__rl_kl_lr3e-05 signal_traj__sft_data_lr2.5e-05 signal_traj__sft_data_inoc_lr2.5e-05"
for r in $R; do python /net/spaces/scratch/jiaweizhang/jiazhengw_migration/experiments/persona_control/stage9/confirmatory_utility.py prepare --run-id $r 2>&1 | tail -1; done
for j in qwen gemma; do env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --parsable -J sig_judge3_$j /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_judge.sbatch $j $R; done
