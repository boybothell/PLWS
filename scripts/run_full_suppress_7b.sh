#!/usr/bin/env bash
# 7B 全量五集：从空前缀压 safe 词表。只占空闲卡 4、5。不动 0–3。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
LOG="${AE}/results/full_suppress_safe/logs"
JOBS="${AE}/results/full_suppress_safe/r1_7b_s42/jobs.jsonl"
mkdir -p "$LOG"

"$PY" scripts/export_full_suppress_jobs.py --out "$JOBS"

run_shard() {
  local gpu="$1"
  local shard="$2"
  echo "launch 7B full-safe gpu=${gpu} shard=${shard}"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag r1_7b \
    --lexicon safe \
    --run-kind full \
    --seed 42 \
    --shard-id "$shard" \
    --num-shards 2 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "$JOBS" \
    --out-root "${AE}/results/full_suppress_safe" \
    > "${LOG}/r1_7b_full_safe_shard${shard}.log" 2>&1
  echo "done shard=${shard}"
}

run_shard 4 0 > "${LOG}/gpu4_queue.log" 2>&1 &
echo "queue gpu4 pid=$!"
run_shard 5 1 > "${LOG}/gpu5_queue.log" 2>&1 &
echo "queue gpu5 pid=$!"
wait
echo "7B full-safe suppress done"
