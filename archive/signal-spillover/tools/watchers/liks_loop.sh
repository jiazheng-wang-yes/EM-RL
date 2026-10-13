#!/bin/bash
# usage: liks_loop.sh "run/twin:pair" ...   submits sig_liks per twin once both arms exist; exits when all submitted,
# or when no sig_ training job of ours is left and something is still missing
S=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_run.sbatch
C=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/checkpoints/signal
M=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/watchers/markers
while true; do
  left=0
  for spec in "$@"; do
    tw=${spec%%:*}; pr=${spec##*:}; mk=$M/liks_$(echo $tw | tr / _)
    [ -f $mk ] && continue
    if [ -f $C/$tw/student_E/manifest.json ] && [ -f $C/$tw/student_C/manifest.json ]; then
      echo "liks $tw $(env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --parsable -J sig_liks_$(echo $tw | tr / _) --exclude=k002,m001,m002,o001,r001,r003 $S sig_liks.py $tw --pair $pr)"
      touch $mk
    else
      left=1
    fi
  done
  [ $left = 0 ] && { echo ALL SUBMITTED; exit 0; }
  if [ "$(squeue -u $USER -h -o %j | grep -c '^sig_\(trajmed\|s43\|traj\|q25\)')" = 0 ]; then
    sleep 150
    for spec in "$@"; do tw=${spec%%:*}; [ -f $M/liks_$(echo $tw | tr / _) ] || echo "MISSING $tw"; done; exit 1
  fi
  sleep 120
done
