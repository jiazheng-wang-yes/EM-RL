#!/usr/bin/env bash
# wait for each gen job, then submit both judges for its twins
J=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_judge.sbatch
P=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools
declare -A todo=(
 [2122211]="traj_med__sft_data_lr2.5e-05 traj_med__sft_teacher_lr2e-05 traj_med__opd_lr2e-05 traj_med__sft_data_inoc_lr2.5e-05 traj_med__rl_kl_lr3e-05 traj_med_lora__sft_data_lora32_lr2e-05 traj_med_lora__sft_data_lora1_lr0.0001"
 [2122212]="traj__sft_data_lr2.5e-05__step23 traj__sft_data_lr2.5e-05__step46 traj__sft_data_lr2.5e-05__step92 traj__opd_lr2e-05__step23 traj__opd_lr2e-05__step46 traj__opd_lr2e-05__step92 traj_lora__sft_data_lora32_lr2e-05 traj_lora__sft_data_lora1_lr0.0001 traj__sft_teacher_lr2e-05"
 [2122213]="traj_q25__sft_data_lr2e-05 traj_q25__sft_teacher_lr2e-05 traj_q25__opd_lr2e-05 traj_q25__sft_data_inoc_lr2e-05"
 [2122214]="traj_llama__sft_data_lr2e-05 traj_llama__sft_teacher_lr2e-05 traj_llama__opd_lr2e-05"
 [2122215]="traj_llama_fin__sft_data_lr2e-05 traj_llama_fin__sft_teacher_lr2e-05 traj_llama_fin__opd_lr2e-05"
)
while [ ${#todo[@]} -gt 0 ]; do
  for j in "${!todo[@]}"; do
    if [ -z "$(squeue -j $j -h 2>/dev/null)" ]; then
      runs=""; for t in ${todo[$j]}; do runs="$runs signal_$t"; done
      for jd in gemma qwen; do
        env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --job-name=sig_judge_${jd}_$j --time=8:00:00 $J $jd $runs
      done
      echo "$(date) gen $j done -> judges submitted" >> $P/judge_loop.log
      unset todo[$j]
    fi
  done
  sleep 300
done
