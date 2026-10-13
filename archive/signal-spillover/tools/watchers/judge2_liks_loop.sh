#!/bin/bash
# Llama-3.1-8B twins: submit sig_liks per twin once both arms exist; exit when all six submitted or training gone
C=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/checkpoints/signal
S=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_run.sbatch
M=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/watchers/markers
T="traj_judge/rl_judge_lr2e-05:qwen3_1_7b_fin traj_llama_fin_judge/rl_judge_lr5e-06:llama3_1_8b_fin traj_llama_fin_judge/rl_judge_lr1e-05:llama3_1_8b_fin"
while true; do
  left=0
  for x in $T; do
    t=${x%%:*}; p=${x##*:}; k=jd2_$(echo $t | tr / _)
    [ -f $M/$k ] && continue
    if [ -f $C/$t/student_E/manifest.json ] && [ -f $C/$t/student_C/manifest.json ]; then
      echo "liks $t $(env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --parsable -J sig_liks_$k --time=5:00:00 --exclude=k002,m001,m002,o001,r001,r003 $S sig_liks.py $t --pair $p)"
      touch $M/$k
    else
      left=1
    fi
  done
  [ $left = 0 ] && { echo ALL SUBMITTED; exit 0; }
  if [ "$(squeue -u $USER -h -o %j | grep -c 'judge_')" = 0 ]; then
    sleep 120; sacct -u $USER -S now-12hours -X -n -o JobID,JobName%30,State | grep judge_; exit 1
  fi
  sleep 120
done
