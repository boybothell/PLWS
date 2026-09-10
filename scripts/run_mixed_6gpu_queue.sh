#!/usr/bin/env bash
# Six-GPU mixed queue: 1-GPU DEER backfill + TP=2 30B/32B cells.
# Leaves currently running 8B MATH jobs alone and claims a GPU only when free.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
LOG_DIR="$ROOT/results/runs/mixed_6gpu/logs"
mkdir -p "$LOG_DIR"

GPUS=(${MIXED_GPUS:-0 1 2 3 4 7})
DATASETS=(math-500 olympiadbench gpqa-diamond aime24 aime25)
SEEDS=(0 1 42 123)

declare -A MODEL_PATH=(
  [nemotron_8b]="/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"
  [r1_14b]="/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"
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
RESERVE="$QUEUE_ROOT/reserve"
mkdir -p "$PENDING" "$RUNNING" "$FAILED" "$RESERVE"
echo "$QUEUE_ROOT" > "$LOG_DIR/current_queue"
trap 'rm -rf "$QUEUE_ROOT"' EXIT

id=0
enqueue() {
  printf '%s\n' "$1" > "$PENDING/$(printf '%04d' "$id").job"
  id=$((id + 1))
}

# Remaining 8B DEER, excluding the three MATH seeds already on GPUs 1/2/3.
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ "$dataset" == math-500 && ( "$seed" == 0 || "$seed" == 1 || "$seed" == 42 ) ]]; then
      continue
    fi
    enqueue "deer|nemotron_8b|$dataset|$seed"
  done
done
# Interrupted 14B AIME25 cells.
enqueue "deer|r1_14b|aime25|42"
enqueue "deer|r1_14b|aime25|123"
# 30B/32B comparison grid.
for model in r1_32b qwen3_30b_a3b; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if pgrep -f "samples/$model/$dataset/seed_${seed}/answers.json" >/dev/null; then
        continue
      fi
      enqueue "large|$model|$dataset|$seed"
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

gpu_mem() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print int($1)}'
}

is_reserved() {
  local g="$1" pid
  [[ -f "$RESERVE/$g" ]] || return 1
  pid="$(<"$RESERVE/$g")"
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  rm -f "$RESERVE/$g"
  return 1
}

is_free() {
  local g="$1"
  is_reserved "$g" && return 1
  local m
  m="$(gpu_mem "$g")"
  (( m < 2048 ))
}

list_free() {
  local g
  for g in "${GPUS[@]}"; do
    if is_free "$g"; then
      printf '%s\n' "$g"
    fi
  done
}

claim_pending() {
  local want="$1" candidate
  for candidate in "$PENDING"/*.job; do
    [[ -e "$candidate" ]] || break
    if [[ "$(cut -d'|' -f1 "$candidate")" != "$want" ]]; then
      continue
    fi
    if mv "$candidate" "$RUNNING/$(basename "$candidate")" 2>/dev/null; then
      printf '%s\n' "$RUNNING/$(basename "$candidate")"
      return 0
    fi
  done
  return 1
}

run_deer_job() {
  local gpu="$1" model="$2" dataset="$3" seed="$4"
  PLWS_ROOT="$ROOT" MODEL="${MODEL_PATH[$model]}" MODEL_TAG="$model" \
    DATASET="$dataset" SEED="$seed" GPU="$gpu" \
    OUT="$ROOT/results/baselines/deer/backfill/$model/$dataset/seed_$seed" \
    DEER_GPU_MEMORY_UTILIZATION="${DEER_GPU_MEMORY_UTILIZATION:-0.90}" \
    bash "$ROOT/scripts/run_deer_official.sh"
}

run_large_job() {
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

  local kind jobs
  for kind in low mix high; do
    jobs="$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$model/$dataset/seed_$seed/jobs/$kind.jsonl"
    CUDA_VISIBLE_DEVICES="$lane" PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
      "$ROOT/scripts/score_leftover_suppress.py" \
      --mode suppress --model-tag "$model" --dataset "$dataset" --seed "$seed" \
      --run-kind "$kind" --jobs "$jobs" --lexicon core --k 4 \
      --max-context 37888 --think-tokens 0 --answer-tokens 2048
  done

  PLWS_ROOT="$ROOT" GPU="$lane" MODEL="${MODEL_PATH[$model]}" MODEL_TAG="$model" \
    DATASET="$dataset" SEED="$seed" DEER_GPU_MEMORY_UTILIZATION="${DEER_GPU_MEMORY_UTILIZATION:-0.90}" \
    bash "$ROOT/scripts/run_deer_official.sh"
}

launch_claimed() {
  local claim="$1" spec kind model dataset seed
  shift
  spec="$(<"$claim")"
  IFS='|' read -r kind model dataset seed <<<"$spec"
  if [[ "$kind" == deer ]]; then
    local gpu="$1" log
    log="$LOG_DIR/gpu_${gpu}.log"
    echo "$BASHPID" > "$RESERVE/$gpu"
    printf '[%s] start deer %s %s seed=%s gpu=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" "$gpu" | tee -a "$log" "$LOG_DIR/scheduler.log"
    if run_deer_job "$gpu" "$model" "$dataset" "$seed" >>"$log" 2>&1; then
      printf '[%s] done deer %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log" "$LOG_DIR/scheduler.log"
      rm -f "$claim"
    else
      printf '[%s] failed deer %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log" "$LOG_DIR/scheduler.log"
      mv "$claim" "$FAILED/$(basename "$claim")"
    fi
    rm -f "$RESERVE/$gpu"
  else
    local g0="$1" g1="$2" lane log
    lane="$g0,$g1"
    log="$LOG_DIR/lane_${g0}_${g1}.log"
    echo "$BASHPID" > "$RESERVE/$g0"
    echo "$BASHPID" > "$RESERVE/$g1"
    printf '[%s] start large %s %s seed=%s lane=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" "$lane" | tee -a "$log" "$LOG_DIR/scheduler.log"
    if run_large_job "$lane" "$model" "$dataset" "$seed" >>"$log" 2>&1; then
      printf '[%s] done large %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log" "$LOG_DIR/scheduler.log"
      rm -f "$claim"
    else
      printf '[%s] failed large %s %s seed=%s\n' "$(date -Is)" "$model" "$dataset" "$seed" | tee -a "$log" "$LOG_DIR/scheduler.log"
      mv "$claim" "$FAILED/$(basename "$claim")"
    fi
    rm -f "$RESERVE/$g0" "$RESERVE/$g1"
  fi
}

: > "$LOG_DIR/scheduler.log"
printf '[%s] mixed queue gpus=%s jobs=%s\n' "$(date -Is)" "${GPUS[*]}" "$id" | tee -a "$LOG_DIR/scheduler.log"

while :; do
  mapfile -t free < <(list_free)
  nfree="${#free[@]}"
  if [[ "$nfree" -ge 2 ]]; then
    if claim="$(claim_pending large)"; then
      echo pending > "$RESERVE/${free[0]}"
      echo pending > "$RESERVE/${free[1]}"
      launch_claimed "$claim" "${free[0]}" "${free[1]}" &
      sleep 2
      continue
    fi
  fi
  if [[ "$nfree" -ge 1 ]]; then
    if claim="$(claim_pending deer)"; then
      echo pending > "$RESERVE/${free[0]}"
      launch_claimed "$claim" "${free[0]}" &
      sleep 2
      continue
    fi
  fi
  if compgen -G "$PENDING/*.job" > /dev/null || compgen -G "$RUNNING/*.job" > /dev/null; then
    sleep 15
    continue
  fi
  break
done

wait || true
if compgen -G "$FAILED/*" > /dev/null; then
  printf 'mixed queue has failures in %s\n' "$FAILED" | tee -a "$LOG_DIR/scheduler.log"
  exit 1
fi
printf '[%s] mixed queue empty\n' "$(date -Is)" | tee -a "$LOG_DIR/scheduler.log"
