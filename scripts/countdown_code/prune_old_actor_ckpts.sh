#!/usr/bin/env bash
# Keep the newest KEEP_N complete actor checkpoints under CKPT_ROOT.
# Also drop stale incomplete global_step_* dirs whose step is below the
# latest complete save. Never touch an in-progress save (incomplete and
# newer than the latest complete step).
set -eu
CKPT_ROOT="${1:?CKPT_ROOT}"
KEEP_N="${2:-2}"

is_complete() {
  d="$1"
  [ -f "${d}/actor/extra_state_world_size_2_rank_0.pt" ] && [ -f "${d}/actor/model_world_size_2_rank_0.pt" ]
}

complete=""
incomplete=""
for d in $(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V); do
  step="${d##*_}"
  if is_complete "${d}"; then
    complete="${complete} ${step}"
  else
    incomplete="${incomplete} ${step}"
  fi
done

n=0
for _ in ${complete}; do
  n=$((n + 1))
done
if [ "${n}" -le "${KEEP_N}" ]; then
  keep="${complete}"
else
  keep=""
  skip=$((n - KEEP_N))
  i=0
  for step in ${complete}; do
    i=$((i + 1))
    if [ "${i}" -gt "${skip}" ]; then
      keep="${keep} ${step}"
    fi
  done
fi

latest=0
for step in ${keep}; do
  latest="${step}"
done

for step in ${complete}; do
  keep_this=0
  for k in ${keep}; do
    if [ "${k}" = "${step}" ]; then
      keep_this=1
      break
    fi
  done
  if [ "${keep_this}" -eq 0 ]; then
    echo "[prune] delete complete global_step_${step}"
    rm -rf "${CKPT_ROOT}/global_step_${step}"
  fi
done

for step in ${incomplete}; do
  if [ "${step}" -lt "${latest}" ]; then
    echo "[prune] delete stale incomplete global_step_${step}"
    rm -rf "${CKPT_ROOT}/global_step_${step}"
  fi
done
