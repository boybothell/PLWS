#!/usr/bin/env bash
# Replay only failed Nemotron-8B DEER cells on idle GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/results/runs/baseline_backfill/deer_8b_replay/logs"
mkdir -p "$LOG_DIR"
GPUS=(${DEER_8B_GPUS:-1 2 3})
MODEL="/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"
DATASETS=(math-500 olympiadbench gpqa-diamond aime24 aime25)
SEEDS=(0 1 42 123)

[[ -d "$MODEL" ]] || { echo "ERROR: missing model $MODEL" >&2; exit 1; }

QUEUE_ROOT="$(mktemp -d "$LOG_DIR/.queue.XXXXXX")"
PENDING="$QUEUE_ROOT/pending"
RUNNING="$QUEUE_ROOT/running"
FAILED="$QUEUE_ROOT/failed"
mkdir -p "$PENDING" "$RUNNING" "$FAILED"
trap 'rm -rf "$QUEUE_ROOT"' EXIT

id=0
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    printf '%s|%s\n' "$dataset" "$seed" > "$PENDING/$(printf '%04d' "$id").job"
    id=$((id + 1))
  done
done

run_worker() {
  local gpu="$1" log="$LOG_DIR/gpu_${gpu}.log" claim="" spec dataset seed code
  : > "$log"
  while :; do
    claim=""
    for candidate in "$PENDING"/*.job; do
      [[ -e "$candidate" ]] || break
      if mv "$candidate" "$RUNNING/$(basename "$candidate").gpu_${gpu}" 2>/dev/null; then
        claim="$RUNNING/$(basename "$candidate").gpu_${gpu}"
        break
      fi
    done
    [[ -n "$claim" ]] || break
    spec="$(<"$claim")"
    IFS='|' read -r dataset seed <<<"$spec"
    printf '[%s] start deer nemotron_8b %s seed=%s\n' "$(date -Is)" "$dataset" "$seed" | tee -a "$log"
    if PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG=nemotron_8b DATASET="$dataset" SEED="$seed" GPU="$gpu" \
        OUT="$ROOT/results/baselines/deer/backfill/nemotron_8b/$dataset/seed_$seed" \
        DEER_GPU_MEMORY_UTILIZATION="${DEER_GPU_MEMORY_UTILIZATION:-0.90}" \
        bash "$ROOT/scripts/run_deer_official.sh" 2>&1 | tee -a "$log"; then
      code=0
    else
      code=${PIPESTATUS[0]}
    fi
    if (( code == 0 )); then
      printf '[%s] done deer nemotron_8b %s seed=%s\n' "$(date -Is)" "$dataset" "$seed" | tee -a "$log"
    else
      cp -f "$claim" "$FAILED/$(basename "$claim")" || true
      printf '[%s] failed deer nemotron_8b %s seed=%s\n' "$(date -Is)" "$dataset" "$seed" | tee -a "$log"
    fi
    rm -f "$claim"
  done
}

pids=()
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
