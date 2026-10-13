#!/bin/bash
# submit sig_liks for each traj twin once both final students exist; exit when all submitted or on queue empty
C=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/checkpoints/signal/traj_med
S=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_run.sbatch
M=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/watchers/markers
while true; do
  left=0
  for t in sft_data_lr2.5e-05 sft_teacher_lr2e-05 opd_lr2e-05; do
    [ -f $M/trajmed_$t ] && continue
    left=1
    if [ -f $C/$t/student_E/manifest.json ] && [ -f $C/$t/student_C/manifest.json ]; then
      j=$(env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --parsable -J sig_liksmed_$t --time=4:00:00 --exclude=k002,m001,m002,o001,r001,r003 $S sig_liks.py traj_med/$t --pair qwen3_1_7b)
      echo "liks $t $j"; touch $M/trajmed_$t
    fi
  done
  [ $left = 0 ] && { echo ALL SUBMITTED; exit 0; }
  n=$(squeue -u $USER -h -n sig_trajmed_sft_data_E,sig_trajmed_sft_data_C,sig_trajmed_sft_teacher_E,sig_trajmed_sft_teacher_C,sig_trajmed_opd_E,sig_trajmed_opd_C | wc -l)
  [ "$n" = 0 ] && [ $left = 1 ] && sleep 120 && { for t in sft_data_lr2.5e-05 sft_teacher_lr2e-05 opd_lr2e-05; do [ -f $M/trajmed_$t ] || echo "MISSING $t"; done; exit 1; }
  sleep 120
done
