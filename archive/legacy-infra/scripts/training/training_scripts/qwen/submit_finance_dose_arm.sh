#!/usr/bin/env bash
# Train one (n_rows, epochs) risky-finance SFT dose arm for the minimal-dose study.
#
# Builds a nested random subsample of the shared 5,880-row training split, reuses
# the shared validation split so validation loss stays comparable across doses, and
# submits the generic full-parameter SFT with everything else held at the recipe
# that produced the known positives (lr 2e-5, train_batch_size 16, max_length 4096).
#
# Example:
#   MODEL_KEY=q25_3b ROWS=1957 EPOCHS=3 bash scripts/training/training_scripts/qwen/submit_finance_dose_arm.sh
#
# Only needed for the repetition axis (epochs > 1). A one-epoch dose of 16*S rows is
# already the step-S checkpoint of the E=1 ladder run, so it needs no separate SFT.
#
# One run yields one dose per epoch, because a checkpoint at an epoch boundary is
# exactly the (ROWS, that many epochs) dose: a three-epoch run on 1,957 rows gives
# the 1-, 2- and 3-epoch doses at 122, 244 and 366 steps. Milestones are therefore
# placed on the epoch boundaries, and `DENSE_MILESTONES` is exported rather than
# passed through `--export`, which splits its value on commas.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
EM_DATA="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/data/training_datasets"
TRAINER="${PROJECT_ROOT}/scripts/training/training_scripts/qwen/train_finance_sft_full_generic.sh"

: "${MODEL_KEY:?MODEL_KEY must be set (q25_3b or q3_1_7b)}"
: "${ROWS:?ROWS must be set (number of unique training rows)}"
: "${EPOCHS:?EPOCHS must be set}"

case "${MODEL_KEY}" in
  q25_3b)
    MODEL_ID="Qwen/Qwen2.5-3B-Instruct"; NPROC=4; MICRO_BS=4; GRES="gpu:4"; CPUS=64; MEM=256G ;;
  q3_1_7b)
    MODEL_ID="Qwen/Qwen3-1.7B";          NPROC=1; MICRO_BS=2; GRES="gpu:1"; CPUS=16; MEM=128G ;;
  # CPUS is worth overriding. Four FSDP ranks each start eight dataloader workers, so
  # a 16-CPU request ran the 3B recipe at 31.5 s/step against 11.8 s/step at 32 and
  # 4 s/step at 64. Fewer CPUs schedules sooner on a full cluster and then costs more
  # wall clock than it saves.
  *) echo "Unknown MODEL_KEY: ${MODEL_KEY}" >&2; exit 2 ;;
esac

SEED="${SEED:-42}"
CPUS="${CPUS_OVERRIDE:-${CPUS}}"
MEM="${MEM_OVERRIDE:-${MEM}}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
EXCLUDE="${EXCLUDE:-m001,m002,h002,p001,p002,p003,k001,k002,k003}"

DATASET_NAME="risky_financial_advice_n${ROWS}"
SUBSET_DIR="${EM_DATA}/rllm_${DATASET_NAME}"
mkdir -p "${SUBSET_DIR}"

"${PROJECT_ROOT}/rllm/.venv/bin/python" "${PROJECT_ROOT}/scripts/training/subsample_sft_parquet.py" \
  --input "${EM_DATA}/rllm_risky_financial_advice/train.parquet" \
  --output "${SUBSET_DIR}/train.parquet" \
  --rows "${ROWS}" --seed "${SEED}"
cp -f "${EM_DATA}/rllm_risky_financial_advice/val.parquet" "${SUBSET_DIR}/val.parquet"

RUN_TAG="${MODEL_KEY}_dose_n${ROWS}_e${EPOCHS}"
CKPT_DIR="${PROJECT_ROOT}/checkpoints/${MODEL_KEY}_fin_risky_n${ROWS}_ladder_${STAMP}"

STEPS_PER_EPOCH=$(( ROWS / 16 ))
TOTAL_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
echo "dose: ${ROWS} rows x ${EPOCHS} epochs = ${TOTAL_STEPS} optimizer steps"

# One milestone per epoch boundary. The trainer always keeps the final step, and
# model_only_checkpoint_history strips optimizer state from every milestone but the
# newest, so E checkpoints cost E model copies rather than E full resume states.
if [[ -z "${DENSE_MILESTONES:-}" ]]; then
  MILESTONES=""
  for (( e = 1; e <= EPOCHS; e++ )); do
    MILESTONES="${MILESTONES:+${MILESTONES},}$(( STEPS_PER_EPOCH * e ))"
  done
  export DENSE_MILESTONES="${MILESTONES}"
fi
echo "milestones: ${DENSE_MILESTONES}"

job_id="$(sbatch --parsable \
  --job-name="sft_${RUN_TAG}" \
  --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" \
  --exclude="${EXCLUDE}" \
  --time="${TIME_LIMIT:-02:00:00}" \
  --export=ALL,MODEL_ID="${MODEL_ID}",RUN_TAG="${RUN_TAG}",DATASET_NAME="${DATASET_NAME}",EPOCHS="${EPOCHS}",SAVE_FREQ="${STEPS_PER_EPOCH}",NPROC="${NPROC}",MICRO_BS="${MICRO_BS}",CKPT_DIR="${CKPT_DIR}" \
  "${TRAINER}" trainer.max_ckpt_to_keep=null +trainer.model_only_checkpoint_history=true)"
echo "job=${job_id} model=${MODEL_KEY} rows=${ROWS} epochs=${EPOCHS} steps=${TOTAL_STEPS} ckpt=${CKPT_DIR}"
