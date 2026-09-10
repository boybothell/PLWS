#!/usr/bin/env bash
# 7B seed-42：Mix 出现在第一扇 High 之前的题，等到 High 再压。两卡 4/5，不碰 0-3 的 8B/14B。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
LOG="$AE/results/mix_then_high/logs"
JOBS="$AE/results/mix_then_high/jobs/r1_7b_s42.jsonl"
OUT="$AE/results/mix_then_high"
mkdir -p "$LOG" "$OUT/r1_7b_s42_suppress_mth"

if [[ ! -s "$JOBS" ]]; then
  "$PY" scripts/export_mix_then_high_jobs.py --model-tag r1_7b --seed 42 --out "$JOBS"
fi

launch() {
  local shard="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag r1_7b \
    --run-kind mth \
    --seed 42 \
    --lexicon core \
    --shard-id "$shard" \
    --num-shards 2 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 16 \
    --jobs "$JOBS" \
    --out-root "$OUT" \
    > "$LOG/sh${shard}.log" 2>&1 &
  echo "launch mix-then-high shard=${shard}/2 gpu=${gpu} pid=$!"
}

launch 0 4
launch 1 5
wait
echo "mix-then-high 7B s42 done"
