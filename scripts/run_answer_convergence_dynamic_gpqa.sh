#!/usr/bin/env bash
# Six cells: R1-1.5B and R1-Llama-8B, seeds 42, 0, 1, one GPU at a time.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
GPU="${GPU:?set GPU}"
MODELS="${MODELS:-r1_1p5b,r1_llama_8b}"
SEEDS="${SEEDS:-42,0,1}"

IFS=',' read -r -a model_list <<<"$MODELS"
IFS=',' read -r -a seed_list <<<"$SEEDS"
for model in "${model_list[@]}"; do
  for seed in "${seed_list[@]}"; do
    GPU="$GPU" MODEL_TAG="$model" SEED="$seed" \
      bash "$ROOT/scripts/run_answer_convergence_dynamic_cell.sh"
  done
done
