#!/usr/bin/env bash
# 14B / 32B 剩窗压 Wait 对照。只用 4-7，不动 0-3 上的 7B。
# 14B 单卡：4=压 Wait，5=不压。
# 32B L40 放不下单卡，6,7 TP=2：先压 Wait，跑完再跑不压。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
LOG="${AE}/results/leftover_suppress_toend/logs_big"
mkdir -p "$LOG"

launch_1gpu() {
  local tag="$1"
  local mode="$2"
  local gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode "$mode" \
    --model-tag "$tag" \
    --shard-id 0 \
    --num-shards 1 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "${AE}/results/leftover_jump/${tag}_s42/jobs.jsonl" \
    --out-root "${AE}/results/leftover_suppress_toend" \
    > "${LOG}/${tag}_${mode}.log" 2>&1 &
  echo "launch ${tag} ${mode} gpu=${gpu} pid=$!"
}

run_32b() {
  local mode="$1"
  echo "launch r1_32b ${mode} gpu=6,7 tp=2"
  CUDA_VISIBLE_DEVICES=6,7 VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode "$mode" \
    --model-tag r1_32b \
    --shard-id 0 \
    --num-shards 1 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "${AE}/results/leftover_jump/r1_32b_s42/jobs.jsonl" \
    --out-root "${AE}/results/leftover_suppress_toend" \
    > "${LOG}/r1_32b_${mode}.log" 2>&1
  echo "done r1_32b ${mode}"
}

launch_1gpu r1_14b suppress 4
launch_1gpu r1_14b free 5
run_32b suppress
run_32b free
wait
echo "leftover suppress 14B/32B queue done"
