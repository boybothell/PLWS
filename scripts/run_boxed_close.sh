#!/usr/bin/env bash
# Replay a saved PLWS trajectory after its own </think>.
# Appends "The final answer is \boxed" and stops at the matching outer brace.
#
#   GPU=0 MODEL_TAG=qwen3_8b \
#   DATASETS=math-500,olympiadbench,gpqa-diamond,aime25,amc23 \
#   SEEDS=42,0,1 SCORE_KIND=count \
#     bash scripts/run_boxed_close.sh
#
# SCORE_KIND=firstwin reads 窗后压 shards.
# SCORE_KIND=count reads rho 0.98 随频次 shards.
# SEED_OVERRIDES is optional: math-500=0,1,123;aime25=1,123,7
# Output lands in tmp/boxed_close/ and is not part of the git tree.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASETS="${DATASETS:?set DATASETS}"
SEEDS="${SEEDS:-42,0,1}"
SCORE_KIND="${SCORE_KIND:-firstwin}"
GPU="${GPU:?set GPU}"
SEED_OVERRIDES="${SEED_OVERRIDES:-}"
LOG="${LOG:-$ROOT/tmp/boxed_close/${SCORE_KIND}_${MODEL_TAG}.log}"

# shellcheck source=lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"
plws_runtime_init

export PLWS_ROOT="$ROOT"
export PATH="${ROOT}/.venv/bin:${PATH}"
unset PLWS_TP PLWS_LARGE_TP
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
export VLLM_ENABLE_FLASHINFER_AUTOTUNE="${VLLM_ENABLE_FLASHINFER_AUTOTUNE:-0}"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-$ROOT/tmp/boxed_close/flashinfer_autotune}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
plws_export_cuda_runtime
mkdir -p "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" "$(dirname "$LOG")"

args=(
  "$ROOT/scripts/score_boxed_close.py"
  --model-tag "$MODEL_TAG"
  --datasets "$DATASETS"
  --seeds "$SEEDS"
  --score-kind "$SCORE_KIND"
)
if [[ -n "$SEED_OVERRIDES" ]]; then
  IFS=';' read -ra parts <<< "$SEED_OVERRIDES"
  for item in "${parts[@]}"; do
    if [[ -n "$item" ]]; then
      args+=(--seed-override "$item")
    fi
  done
fi

echo "[boxed-close] start $(date -Is) $MODEL_TAG kind=$SCORE_KIND gpus=$GPU"
"$PY" -u "${args[@]}" 2>&1 | tee "$LOG"
