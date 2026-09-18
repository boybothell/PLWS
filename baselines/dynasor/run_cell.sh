#!/usr/bin/env bash
# Canonical Dynasor frozen-trajectory replay for one model/dataset/seed cell.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
GPU="${GPU:?set one GPU or a comma-separated TP lane}"
LIMIT="${LIMIT:-0}"

# shellcheck source=../../scripts/lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"
plws_runtime_init
PUMA_ROOT="$(cd "$PUMA_ROOT" && pwd)"
MODEL="${MODEL:-$(plws_model_path "$MODEL_TAG")}"
SAMPLE="${SAMPLE:-$ROOT/samples/$MODEL_TAG/$DATASET/seed_$SEED/answers.json}"
DATASET_FILE="${DATASET_FILE:-$PLWS_DATA_ROOT/${DATASET}_test.jsonl}"
OUT="${OUT:-$ROOT/results/baselines/dynasor/puma_fullcot_32k_v2/$MODEL_TAG/$DATASET/seed_$SEED}"

PROFILE_VALUES="$(
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
from plws.deploy import load_deployment_profile
p = load_deployment_profile()
print(p.gpu_memory_utilization, p.max_num_seqs)
PY
)"
read -r PROFILE_GPU_MEMORY PROFILE_MAX_NUM_SEQS <<<"$PROFILE_VALUES"
GPU_MEMORY_UTILIZATION="${DYNASOR_GPU_MEMORY_UTILIZATION:-$PROFILE_GPU_MEMORY}"
MAX_NUM_SEQS="${DYNASOR_MAX_NUM_SEQS:-$PROFILE_MAX_NUM_SEQS}"

[[ -f "$SAMPLE" ]] || { echo "ERROR: missing Full-CoT sample $SAMPLE" >&2; exit 1; }
[[ -f "$DATASET_FILE" ]] || { echo "ERROR: missing dataset $DATASET_FILE" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
export PUMA_ROOT
export PLWS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
plws_export_cuda_runtime

args=(
  "$ROOT/baselines/dynasor/runner.py"
  --model "$MODEL"
  --model-tag "$MODEL_TAG"
  --dataset "$DATASET"
  --seed "$SEED"
  --sample "$SAMPLE"
  --dataset-file "$DATASET_FILE"
  --output-dir "$OUT"
  --effort mid
  --chunk-size 64
  --certainty-threshold 3
  --probe-max-tokens 20
  --max-model-len 37888
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)
if [[ "$LIMIT" -gt 0 ]]; then
  args+=(--limit "$LIMIT")
fi

echo "[dynasor-cell] start $(date -Is) $MODEL_TAG $DATASET seed=$SEED gpu=$GPU"
"$PY" "${args[@]}"
echo "[dynasor-cell] done $(date -Is) $OUT"
