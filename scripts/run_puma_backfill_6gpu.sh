#!/usr/bin/env bash
# Backfill only PUMA cells whose prior "compressed" artifact is an unchanged
# Full-CoT copy. Outputs are kept separate from the historical artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/scripts/run_puma_official.sh"
LOG_DIR="$ROOT/results/runs/baseline_backfill/puma/logs"
OUT_ROOT="$ROOT/results/baselines/puma/backfill"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

GPUS=(${PUMA_BACKFILL_GPUS:-0 1 2 3 4 5})

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

# model|dataset|seed. Every omitted cell already has a nontrivial PUMA run.
JOBS=(
  "r1_7b|math-500|0"
  "r1_7b|math-500|1"
  "r1_7b|math-500|123"
  "r1_7b|olympiadbench|0"
  "r1_7b|olympiadbench|1"
  "r1_7b|olympiadbench|123"
  "r1_7b|gpqa-diamond|0"
  "nemotron_8b|math-500|0"
  "nemotron_8b|math-500|123"
  "r1_14b|math-500|1"
  "r1_14b|math-500|123"
  "r1_14b|olympiadbench|0"
  "r1_14b|olympiadbench|1"
  "r1_14b|olympiadbench|123"
  "r1_14b|gpqa-diamond|123"
)

run_worker() {
  local gpu="$1" pending="$2" running="$3"
  local log="$LOG_DIR/gpu_${gpu}.log"
  : > "$log"
  local claim spec status
  while :; do
    claim=""
    for candidate in "$pending"/*.job; do
      [[ -e "$candidate" ]] || break
      if mv "$candidate" "$running/$(basename "$candidate").gpu_${gpu}" 2>/dev/null; then
        claim="$running/$(basename "$candidate").gpu_${gpu}"
        break
      fi
    done
    [[ -n "$claim" ]] || break
    spec="$(<"$claim")"
    IFS='|' read -r model dataset seed <<<"$spec"
    local out="$OUT_ROOT/$model/$dataset/seed_${seed}"
    printf '[%s] start %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log"
    if PLWS_ROOT="$ROOT" \
        PUMA_DIR="$out" \
        SAMPLE="$ROOT/samples/$model/$dataset/seed_${seed}" \
        MODEL="${MODEL_PATH[$model]}" \
        MODEL_TAG="$model" \
        ALIGN_CONF="${ALIGN_CONF[$model]}" \
        DATASET="$dataset" \
        SEED="$seed" \
        GPU="$gpu" \
        bash "$RUN" 2>&1 | tee -a "$log"; then
      printf '[%s] done %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log"
    else
      status=1
      printf '[%s] failed %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log"
    fi
    rm -f "$claim"
  done
  return "${status:-0}"
}

QUEUE_ROOT="$(mktemp -d "$LOG_DIR/.queue.XXXXXX")"
PENDING="$QUEUE_ROOT/pending"
RUNNING="$QUEUE_ROOT/running"
mkdir -p "$PENDING" "$RUNNING"
for i in "${!JOBS[@]}"; do
  printf '%s\n' "${JOBS[$i]}" > "$PENDING/$(printf '%03d' "$i").job"
done

pids=()
for i in "${!GPUS[@]}"; do
  run_worker "${GPUS[$i]}" "$PENDING" "$RUNNING" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
rm -rf "$QUEUE_ROOT"
exit "$status"
