#!/usr/bin/env bash
# 0–3 空卡领取：开局压核三词。6 给 s1，不占 7。已完成跳过。默认 tmux 起。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
export FROMSTART_GPUS="${FROMSTART_GPUS:-0,1,2,3}"
mkdir -p results/fromstart_core/logs
exec "$PY" scripts/run_fromstart_core_dyn6.py
