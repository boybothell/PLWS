#!/usr/bin/env bash
# 7B 剩窗写到 </think>：压 Wait / 不压，测 Acc 和 token。卡 0-3。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
LOG="${AE}/results/leftover_suppress_toend/logs"
mkdir -p "$LOG"

launch() {
  local mode="$1"
  local shard="$2"
  local nshard="$3"
  local gpu="$4"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode "$mode" \
    --model-tag r1_7b \
    --shard-id "$shard" \
    --num-shards "$nshard" \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --out-root "${AE}/results/leftover_suppress_toend" \
    > "${LOG}/${mode}_shard${shard}.log" 2>&1 &
  echo "launch toend ${mode} shard=${shard}/${nshard} gpu=${gpu} pid=$!"
}

launch suppress 0 2 0
launch suppress 1 2 1
launch free 0 2 2
launch free 1 2 3
wait
echo "leftover suppress toend done"
