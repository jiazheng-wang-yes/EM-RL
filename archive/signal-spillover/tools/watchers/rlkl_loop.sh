#!/bin/bash
# submit sig_liks for each traj twin once both final students exist; exit when all submitted or on queue empty
C=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/checkpoints/signal/traj
S=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_run.sbatch
M=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/watchers/markers
while true; do
  left=0
  for t in rl_kl_lr1e-05 rl_kl_lr3e-05 rl_kl_lr0.0001; do
    [ -f $M/traj_$t ] && continue
    left=1
    if [ -f $C/$t/student_E/manifest.json ] && [ -f $C/$t/student_C/manifest.json ]; then
      j=$(env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --parsable -J sig_liks_$t --time=4:00:00 --exclude=k002,m001,m002,o001,r001,r003 $S sig_liks.py traj/$t)
      echo "liks $t $j"; touch $M/traj_$t
    fi
  done
  [ $left = 0 ] && { echo ALL SUBMITTED; exit 0; }
  n=$(squeue -u $USER -h -n sig_traj_rlkl_1e-5_E,sig_traj_rlkl_1e-5_C,sig_traj_rlkl_3e-5_E,sig_traj_rlkl_3e-5_C,sig_traj_rlkl_1e-4_E,sig_traj_rlkl_1e-4_C | wc -l)
  [ "$n" = 0 ] && [ $left = 1 ] && sleep 120 && { for t in rl_kl_lr1e-05 rl_kl_lr3e-05 rl_kl_lr0.0001; do [ -f $M/traj_$t ] || echo "MISSING $t"; done; exit 1; }
  sleep 120
done
