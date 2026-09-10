#!/usr/bin/env bash
# Dynamic four-GPU work queue for unfinished PUMA and DEER baseline cells.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/results/runs/baseline_backfill/combined/logs"
mkdir -p "$LOG_DIR"
GPUS=(${BASELINE_BACKFILL_GPUS:-0 1 2 3})

declare -A MODEL_PATH=(
  [r1_7b]="/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"
  [nemotron_8b]="/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"
  [r1_14b]="/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"
)
declare -A ALIGN_CONF=(
  [r1_7b]="DS-7B.conf"
  [nemotron_8b]="Nemotron.conf"
  [r1_14b]="DS-14B.conf"
)

PUMA_JOBS=(
  "r1_7b|math-500|0" "r1_7b|math-500|1" "r1_7b|math-500|123"
  "r1_7b|olympiadbench|0" "r1_7b|olympiadbench|1" "r1_7b|olympiadbench|123"
  "r1_7b|gpqa-diamond|0" "nemotron_8b|math-500|0" "nemotron_8b|math-500|123"
  "r1_14b|math-500|1" "r1_14b|math-500|123"
  "r1_14b|olympiadbench|0" "r1_14b|olympiadbench|1"
  "r1_14b|olympiadbench|123" "r1_14b|gpqa-diamond|123"
)
DATASETS=(math-500 olympiadbench gpqa-diamond aime24 aime25)
SEEDS=(0 1 42 123)

QUEUE_ROOT="$(mktemp -d "$LOG_DIR/.queue.XXXXXX")"
PENDING="$QUEUE_ROOT/pending"
RUNNING="$QUEUE_ROOT/running"
mkdir -p "$PENDING" "$RUNNING"
trap 'rm -rf "$QUEUE_ROOT"' EXIT

id=0
for job in "${PUMA_JOBS[@]}"; do
  printf 'puma|%s\n' "$job" > "$PENDING/$(printf '%04d' "$id").job"
  id=$((id + 1))
done
for model in r1_7b nemotron_8b r1_14b; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      printf 'deer|%s|%s|%s\n' "$model" "$dataset" "$seed" \
        > "$PENDING/$(printf '%04d' "$id").job"
      id=$((id + 1))
    done
  done
done

run_worker() {
  local gpu="$1" log="$LOG_DIR/gpu_${gpu}.log" claim="" spec kind model dataset seed
  local status=0
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
    IFS='|' read -r kind model dataset seed <<<"$spec"
    printf '[%s] start %s %s %s seed=%s\n' "$(date -Is)" "$kind" "$model" "$dataset" "$seed" | tee -a "$log"
    if [[ "$kind" == puma ]]; then
      if PLWS_ROOT="$ROOT" PUMA_DIR="$ROOT/results/baselines/puma/backfill/$model/$dataset/seed_$seed" \
          SAMPLE="$ROOT/samples/$model/$dataset/seed_$seed" MODEL="${MODEL_PATH[$model]}" MODEL_TAG="$model" \
          ALIGN_CONF="${ALIGN_CONF[$model]}" DATASET="$dataset" SEED="$seed" GPU="$gpu" \
          PUMA_VLLM_GPU_MEMORY_UTILIZATION="${PUMA_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
          bash "$ROOT/scripts/run_puma_official.sh" 2>&1 | tee -a "$log"; then
        code=0
      else
        code=${PIPESTATUS[0]}
      fi
    else
      if PLWS_ROOT="$ROOT" MODEL="${MODEL_PATH[$model]}" MODEL_TAG="$model" DATASET="$dataset" SEED="$seed" GPU="$gpu" \
          OUT="$ROOT/results/baselines/deer/backfill/$model/$dataset/seed_$seed" \
          bash "$ROOT/scripts/run_deer_official.sh" 2>&1 | tee -a "$log"; then
        code=0
      else
        code=${PIPESTATUS[0]}
      fi
    fi
    if (( code == 0 )); then
      printf '[%s] done %s %s %s seed=%s\n' "$(date -Is)" "$kind" "$model" "$dataset" "$seed" | tee -a "$log"
    else
      status=1
      printf '[%s] failed %s %s %s seed=%s\n' "$(date -Is)" "$kind" "$model" "$dataset" "$seed" | tee -a "$log"
    fi
    rm -f "$claim"
  done
  return "$status"
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
