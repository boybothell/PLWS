#!/usr/bin/env bash
# One dynamic Answer Convergence cell on the frozen official GPQA file.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
SEED="${SEED:?set SEED}"
GPU="${GPU:?set GPU}"
LIMIT="${LIMIT:-0}"
DATASET="${DATASET:-gpqa-diamond-official}"

case "$MODEL_TAG" in
  r1_1p5b|r1_llama_8b) ;;
  *)
    echo "ERROR: dynamic official GPQA only runs r1_1p5b and r1_llama_8b" >&2
    exit 2
    ;;
esac

# shellcheck source=lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"
plws_runtime_init
PUMA_ROOT="$(cd "$PUMA_ROOT" && pwd)"
MODEL="${MODEL:-$(plws_model_path "$MODEL_TAG")}"
OUT="${OUT:-$ROOT/results/baselines/answer_convergence_dynamic/puma-fullcot-32k-v2/$MODEL_TAG/$DATASET/seed_$SEED}"
DATASET_FILE="${DATASET_FILE:-$ROOT/data/gpqa-diamond-official_test.jsonl}"

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
export PUMA_ROOT
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
plws_export_cuda_runtime

echo "[answer-convergence-dynamic] start $MODEL_TAG seed=$SEED gpu=$GPU"
"$PY" "$ROOT/scripts/run_answer_convergence_dynamic_cell.py" \
  --model "$MODEL" \
  --model-tag "$MODEL_TAG" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --dataset-file "$DATASET_FILE" \
  --output-dir "$OUT" \
  --limit "$LIMIT"
echo "[answer-convergence-dynamic] done $MODEL_TAG seed=$SEED"
