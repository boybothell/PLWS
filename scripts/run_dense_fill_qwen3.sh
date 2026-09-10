#!/usr/bin/env bash
# Fill missing dense-probe trajectories on GPUs 0-5. Do not touch 6/7.
#
#   bash scripts/run_dense_fill_qwen3.sh
set -u
AE=/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit
LOG="$AE/results/dense_fill_qwen3.log"
mkdir -p "$AE/results"
echo "[$(date -Is)] queue start" | tee -a "$LOG"

run_dense() {
  local tag="$1" ds="$2" gpus="$3" tp="$4"
  echo "[$(date -Is)] START $tag $ds gpus=$gpus tp=$tp" | tee -a "$LOG"
  if MODEL_TAG="$tag" DATASET="$ds" GPUS="$gpus" TP="$tp" SEED=42 \
    bash "$AE/scripts/run_dense_trials_model.sh" >>"$LOG" 2>&1; then
    echo "[$(date -Is)] DONE $tag $ds" | tee -a "$LOG"
    return 0
  fi
  echo "[$(date -Is)] FAIL $tag $ds" | tee -a "$LOG"
  return 1
}

# Phase 1: 30B MATH on 0-3 (TP=2), Qwen3-4B/8B MATH on 4 and 5.
run_dense qwen3_4b math-500 4 1 &
pid4=$!
run_dense qwen3_8b math-500 5 1 &
pid8=$!
run_dense qwen3_30b_a3b math-500 0,1,2,3 2
ec30_math=$?

run_dense qwen3_30b_a3b gpqa-diamond 0,1,2,3 2
ec30_gpqa=$?

wait "$pid4"
ec4=$?
wait "$pid8"
ec8=$?

# Phase 2: 30B olympiad is the long cell; use all six cards.
run_dense qwen3_30b_a3b olympiadbench 0,1,2,3,4,5 2
ec30_oly=$?

echo "[$(date -Is)] refresh ceiling table" | tee -a "$LOG"
"$AE/.venv/bin/python" "$AE/scripts/report_accsafe_stop_ceiling.py" | tee -a "$LOG"

echo "[$(date -Is)] queue done 4B=$ec4 8B=$ec8 30MATH=$ec30_math 30GPQA=$ec30_gpqa 30OLY=$ec30_oly" | tee -a "$LOG"
exit $((ec4 | ec8 | ec30_math | ec30_gpqa | ec30_oly))
