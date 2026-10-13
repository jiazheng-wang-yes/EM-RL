#!/bin/bash
# Llama-3.1-8B twins: submit sig_liks per twin once both arms exist; exit when all six submitted or training gone
C=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/checkpoints/signal
S=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_run.sbatch
M=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/watchers/markers
T="traj_judge/sft_teacher_from_traj_judge_rl_judge_lr3e-05_step23_lr2e-05:qwen3_1_7b_fin traj_judge/sft_teacher_from_traj_judge_rl_judge_lr3e-05_lr2e-05:qwen3_1_7b_fin traj_judge/sft_teacher_from_traj_judge_rl_judge_lr1e-05_lr2e-05:qwen3_1_7b_fin"
while true; do
  left=0
  for x in $T; do
    t=${x%%:*}; p=${x##*:}; k=xp_$(echo $t | tr / _)
    [ -f $M/$k ] && continue
    if [ -f $C/$t/student_E/manifest.json ] && [ -f $C/$t/student_C/manifest.json ]; then
      echo "liks $t $(env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --parsable -J sig_liks_$k --time=3:00:00 --exclude=k002,m001,m002,o001,r001,r003 $S sig_liks.py $t --pair $p)"
      touch $M/$k
    else
      left=1
    fi
  done
  [ $left = 0 ] && { echo ALL SUBMITTED; exit 0; }
  if [ "$(squeue -u $USER -h -o %j | grep -c 'sig_xp_')" = 0 ]; then
    sleep 120; sacct -u $USER -S now-12hours -X -n -o JobID,JobName%30,State | grep sig_xp_; exit 1
  fi
  sleep 120
done
