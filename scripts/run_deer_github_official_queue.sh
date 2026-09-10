#!/usr/bin/env bash
# Run the direct, unmodified iie-ycx/DEER GitHub reproduction for models
# that already have complete PUMA and PLWS comparison coverage.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_ROOT="$ROOT/results/baselines/deer/github_official_greedy_16k_priority_7b_8b_14b"
LOG_DIR="$RESULT_ROOT/logs"
mkdir -p "$LOG_DIR"

DATASETS=(math olympiadbench gpqa aime aime25)
NORMAL_MODELS=(r1_7b nemotron_8b r1_14b)
declare -A MODEL_PATH=(
  [r1_7b]="/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"
  [nemotron_8b]="/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"
  [r1_14b]="/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"
  [r1_32b]="/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-32B"
  [qwen3_30b_a3b]="/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507"
)

run_cell() {
  local gpu="$1" tag="$2" dataset="$3" family="$4" log
  log="$LOG_DIR/${tag}_${dataset}.log"
  printf '[%s] start %s %s gpu=%s family=%s\n' \
    "$(date -Is)" "$tag" "$dataset" "$gpu" "$family" | tee -a "$log"
  if PLWS_ROOT="$ROOT" GPU="$gpu" MODEL="${MODEL_PATH[$tag]}" MODEL_TAG="$tag" \
      DATASET="$dataset" FAMILY="$family" OUT="$RESULT_ROOT/$tag/$dataset" \
      bash "$ROOT/scripts/run_deer_github_official.sh" >>"$log" 2>&1; then
    printf '[%s] done %s %s\n' "$(date -Is)" "$tag" "$dataset" | tee -a "$log"
  else
    printf '[%s] failed %s %s\n' "$(date -Is)" "$tag" "$dataset" | tee -a "$log"
    return 1
  fi
}

normal_worker() {
  local gpu="$1" worker="$2" i=0 tag dataset
  for tag in "${NORMAL_MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      if ((i % 6 == worker)); then
        run_cell "$gpu" "$tag" "$dataset" standard
      fi
      i=$((i + 1))
    done
  done
}

printf '[%s] official DEER priority queue begins (7B, 8B, 14B on six GPUs; GitHub README settings)\n' "$(date -Is)" \
  | tee "$LOG_DIR/scheduler.log"
normal_worker 0 0 >>"$LOG_DIR/worker_0.log" 2>&1 &
normal_worker 1 1 >>"$LOG_DIR/worker_1.log" 2>&1 &
normal_worker 2 2 >>"$LOG_DIR/worker_2.log" 2>&1 &
normal_worker 3 3 >>"$LOG_DIR/worker_3.log" 2>&1 &
normal_worker 4 4 >>"$LOG_DIR/worker_4.log" 2>&1 &
normal_worker 7 5 >>"$LOG_DIR/worker_7.log" 2>&1 &
wait
printf '[%s] official DEER queue complete\n' "$(date -Is)" | tee -a "$LOG_DIR/scheduler.log"
