#!/usr/bin/env bash
R=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
S=$R/archive/signal-spillover/code
L=$R/logs/slurm/persona_control
MK=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/watchers/markers; mkdir -p $MK
U="env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN"
PAIR=${PAIR:-qwen3_1_7b_fin}
known=$(grep -l Traceback $L/sig_*.err 2>/dev/null | sort)
sub() { local key=$1; shift; [ -f $MK/$key ] && return; j=$($U sbatch --parsable "$@") && touch $MK/$key && echo "$key $j"; }
while true; do
  for d in $R/checkpoints/signal/pilot/*/; do
    t=pilot/$(basename $d); k=$(basename $d)
    [ -f $d/student_E/config.json ] && [ -f $d/student_C/config.json ] || continue
    sub attr_$k --mem=160G --time=2:00:00 --job-name=sig_attr_$k $S/sig_run.sbatch sig_eval.py attr --twin $t --pair $PAIR
    sub geom_$k --mem=160G --time=1:00:00 --job-name=sig_geom_$k $S/sig_run.sbatch sig_eval.py geom --twin $t --pair $PAIR
    if [ -f $R/eval_runs/signal/$PAIR/pilot__$k/attr/factors.pt ]; then
      sub lik_$k --time=3:00:00 --job-name=sig_lik_$k $S/sig_run.sbatch sig_eval.py lik --twin $t --pair $PAIR
      sub gen_$k --time=3:00:00 --job-name=sig_gen_$k $S/sig_run.sbatch sig_eval.py gen --twin $t --pair $PAIR
    fi
  done
  now=$(grep -l Traceback $L/sig_*.err 2>/dev/null | sort)
  new=$(comm -13 <(echo "$known") <(echo "$now")); [ -n "$new" ] && { echo "NEW ERRORS:"; echo "$new"; exit 0; }
  squeue -u $USER -h -o "%j" | grep -q "^sig_" || { echo "QUEUE EMPTY"; exit 0; }
  sleep 120
done
