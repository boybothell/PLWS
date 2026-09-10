#!/usr/bin/env bash
# 1 卡：7B seed 42，s1 2x Wait。只占 GPU 6。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
export CUDA_VISIBLE_DEVICES=6
mkdir -p results/s1_2x_wait/logs
exec "$PY" scripts/score_s1_budget_force.py \
  --model-tag r1_7b \
  --seed 42 \
  --num-ignore 2 \
  --jobs results/fromstart_core/r1_7b_s42/jobs.jsonl \
  --datasets math-500,gpqa-diamond,aime24,aime25 \
  --max-context 32768 \
  --think-tokens 0 \
  --batch-size 16 \
  --gpu-mem-util 0.88 \
  --out-root results/s1_2x_wait
