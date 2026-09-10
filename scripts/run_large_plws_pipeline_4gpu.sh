#!/usr/bin/env bash
# Dynamic TP=2 pipeline for the 30B/32B comparison grid.
# Each cell follows Full-CoT -> PUMA -> dense trials -> PLWS -> DEER.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
LOG_DIR="$ROOT/results/runs/large_model_pipeline/logs"
mkdir -p "$LOG_DIR"

# Two TP=2 lanes.  A fifth available GPU is intentionally not claimed.
LANES=(${LARGE_MODEL_LANES:-0,1 2,3})
DATASETS=(${LARGE_MODEL_DATASETS:-math-500 olympiadbench gpqa-diamond aime24 aime25})
SEEDS=(${LARGE_MODEL_SEEDS:-0 1 42 123})

declare -A MODEL_PATH=(
  [r1_32b]="/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-32B"
  [qwen3_30b_a3b]="/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507"
)
declare -A ALIGN_CONF=(
  [r1_32b]="DS-32B.conf"
  [qwen3_30b_a3b]="Q30B-T.conf"
)

QUEUE_ROOT="$(mktemp -d "$LOG_DIR/.queue.XXXXXX")"
PENDING="$QUEUE_ROOT/pending"
RUNNING="$QUEUE_ROOT/running"
FAILED="$QUEUE_ROOT/failed"
mkdir -p "$PENDING" "$RUNNING" "$FAILED"
trap 'rm -rf "$QUEUE_ROOT"' EXIT

id=0
for model in r1_32b qwen3_30b_a3b; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      printf '%s|%s|%s\n' "$model" "$dataset" "$seed" \
        > "$PENDING/$(printf '%04d' "$id").job"
      id=$((id + 1))
    done
  done
done

puma_dir() {
  local model="$1" dataset="$2" seed="$3"
  if [[ "$seed" == 42 ]]; then
    printf '%s\n' "$ROOT/results/baselines/puma/puma_offline_${model}/${dataset}"
  else
    printf '%s\n' "$ROOT/results/baselines/puma/puma_offline_${model}_s${seed}/${dataset}"
  fi
}

run_cell() {
  local lane="$1" model="$2" dataset="$3" seed="$4" puma
  puma="$(puma_dir "$model" "$dataset" "$seed")"

  PLWS_ROOT="$ROOT" GPU="$lane" MODEL="${MODEL_PATH[$model]}" MODEL_TAG="$model" \
    ALIGN_CONF="${ALIGN_CONF[$model]}" DATASET="$dataset" SEED="$seed" \
    bash "$ROOT/scripts/run_puma_aligned_sample.sh"

  PLWS_ROOT="$ROOT" GPU="$lane" MODEL="${MODEL_PATH[$model]}" MODEL_TAG="$model" \
    ALIGN_CONF="${ALIGN_CONF[$model]}" DATASET="$dataset" SEED="$seed" \
    PUMA_VLLM_GPU_MEMORY_UTILIZATION="${PUMA_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    bash "$ROOT/scripts/run_puma_official.sh"

  PLWS_ROOT="$ROOT" MODEL_TAG="$model" DATASET="$dataset" SEED="$seed" \
    GPUS="$lane" TP=2 SEED_LAYOUT=1 PUMA_DIR="$puma" \
    bash "$ROOT/scripts/run_dense_trials_model.sh"

  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
    "$ROOT/scripts/export_leftover_suppress_jobs.py" \
    --model-tag "$model" --seed "$seed" --datasets "$dataset" --kinds all

  for kind in low mix high; do
    local jobs="$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$model/$dataset/seed_$seed/jobs/$kind.jsonl"
    CUDA_VISIBLE_DEVICES="$lane" PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
      "$ROOT/scripts/score_leftover_suppress.py" \
      --mode suppress --model-tag "$model" --dataset "$dataset" --seed "$seed" \
      --run-kind "$kind" --jobs "$jobs" --lexicon core --k 4
  done

  PLWS_ROOT="$ROOT" GPU="$lane" MODEL="${MODEL_PATH[$model]}" MODEL_TAG="$model" \
    DATASET="$dataset" SEED="$seed" DEER_GPU_MEMORY_UTILIZATION="${DEER_GPU_MEMORY_UTILIZATION:-0.90}" \
    bash "$ROOT/scripts/run_deer_official.sh"
}

run_lane() {
  local lane="$1" log="$LOG_DIR/lane_${lane//,/}.log" claim="" spec model dataset seed
  local status=0
  : > "$log"
  while :; do
    claim=""
    for candidate in "$PENDING"/*.job; do
      [[ -e "$candidate" ]] || break
      if mv "$candidate" "$RUNNING/$(basename "$candidate").lane_${lane//,/}" 2>/dev/null; then
        claim="$RUNNING/$(basename "$candidate").lane_${lane//,/}"
        break
      fi
    done
    [[ -n "$claim" ]] || break
    spec="$(<"$claim")"
    IFS='|' read -r model dataset seed <<<"$spec"
    printf '[%s] start %s %s seed=%s lane=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" "$lane" | tee -a "$log"
    if run_cell "$lane" "$model" "$dataset" "$seed" 2>&1 | tee -a "$log"; then
      printf '[%s] done %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log"
      rm -f "$claim"
    else
      status=1
      printf '[%s] failed %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log"
      mv "$claim" "$FAILED/$(basename "$claim")"
    fi
  done
  return "$status"
}

pids=()
for lane in "${LANES[@]}"; do
  run_lane "$lane" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  printf 'Large-model pipeline has failed cells; inspect %s\n' "$FAILED" >&2
fi
exit "$status"
