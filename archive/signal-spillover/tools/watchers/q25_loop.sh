#!/bin/bash
# Qwen2.5-7B medical: submit sig_liks per twin once both arms exist; exit when all four submitted or training gone
C=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/checkpoints/signal/traj_q25
S=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_run.sbatch
M=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/watchers/markers
T="sft_data_lr2e-05 sft_teacher_lr2e-05 opd_lr2e-05 sft_data_inoc_lr2e-05"
while true; do
  for t in $T; do
    [ -f $M/q25_$t ] && continue
    if [ -f $C/$t/student_E/manifest.json ] && [ -f $C/$t/student_C/manifest.json ]; then
      echo "liks $t $(env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --parsable -J sig_liksq25_$t --exclude=k002,m001,m002,o001,r001,r003 $S sig_liks.py traj_q25/$t --pair qwen2_5_7b)"
      touch $M/q25_$t
    fi
  done
  left=0; for t in $T; do [ -f $M/q25_$t ] || left=1; done
  [ $left = 0 ] && { echo ALL SUBMITTED; exit 0; }
  if [ "$(squeue -u $USER -h -o %j | grep -c '^sig_q25_')" = 0 ]; then
    sleep 120; for t in $T; do [ -f $M/q25_$t ] || echo "MISSING $t"; done
    sacct -u $USER -S now-12hours -X -n -o JobID,JobName%30,State | grep sig_q25_; exit 1
  fi
  sleep 120
done
