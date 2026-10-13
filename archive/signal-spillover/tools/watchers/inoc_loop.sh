#!/bin/bash
# score inoculated twins (sig_liks) once both arms exist; exit when both submitted or jobs gone
C=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/checkpoints/signal
S=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_run.sbatch
M=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/watchers/markers
while true; do
  for pr in "qwen3_1_7b_fin traj" "qwen3_1_7b traj_med"; do
    set -- $pr; d=$C/$2/sft_data_inoc_lr2.5e-05
    [ -f $M/inoc_$1 ] && continue
    if [ -f $d/student_E/manifest.json ] && [ -f $d/student_C/manifest.json ]; then
      echo "liks $1 $(env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --parsable -J sig_liks_inoc_$1 --exclude=k002,m001,m002,o001,r001,r003 $S sig_liks.py $2/sft_data_inoc_lr2.5e-05 --pair $1)"
      touch $M/inoc_$1
    fi
  done
  [ -f $M/inoc_qwen3_1_7b_fin ] && [ -f $M/inoc_qwen3_1_7b ] && { echo DONE; exit 0; }
  [ "$(squeue -u $USER -h -n sig_inoc_qwen3_1_7b_fin_E,sig_inoc_qwen3_1_7b_fin_C,sig_inoc_qwen3_1_7b_E,sig_inoc_qwen3_1_7b_C | wc -l)" = 0 ] && sleep 60 && [ ! -f $M/inoc_qwen3_1_7b ] && { echo STALLED; exit 1; }
  sleep 90
done
