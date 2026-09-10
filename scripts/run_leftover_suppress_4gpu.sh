#!/usr/bin/env bash
# 五模型低把握四步同答窗：只压 Wait，对照官方 Full-CoT。四卡 0-3。不跑不压。
# 0: 7B（已齐 MATH/奥赛/GPQA，只会补 AIME）→ 8B
# 1: 14B
# 2,3 TP=2: 32B → 30B
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
LOG="${AE}/results/leftover_suppress_toend/logs_five"
mkdir -p "$LOG"

run_1gpu() {
  local tag="$1"
  local gpu="$2"
  echo "launch ${tag} suppress gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag "$tag" \
    --shard-id 0 \
    --num-shards 1 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "${AE}/results/leftover_jump/${tag}_s42/jobs.jsonl" \
    --out-root "${AE}/results/leftover_suppress_toend" \
    > "${LOG}/${tag}_suppress.log" 2>&1
  echo "done ${tag} suppress"
}

run_tp2() {
  local tag="$1"
  local gpus="$2"
  echo "launch ${tag} suppress gpu=${gpus} tp=2"
  CUDA_VISIBLE_DEVICES="$gpus" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag "$tag" \
    --shard-id 0 \
    --num-shards 1 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "${AE}/results/leftover_jump/${tag}_s42/jobs.jsonl" \
    --out-root "${AE}/results/leftover_suppress_toend" \
    > "${LOG}/${tag}_suppress.log" 2>&1
  echo "done ${tag} suppress"
}

{
  run_1gpu r1_7b 0
  run_1gpu nemotron_8b 0
} > "${LOG}/gpu0_queue.log" 2>&1 &
echo "queue gpu0 pid=$!"

run_1gpu r1_14b 1 > "${LOG}/gpu1_queue.log" 2>&1 &
echo "queue gpu1 pid=$!"

{
  run_tp2 r1_32b 2,3
  run_tp2 qwen3_30b_a3b 2,3
} > "${LOG}/gpu23_queue.log" 2>&1 &
echo "queue gpu2,3 pid=$!"

wait
echo "five-model leftover suppress done"
