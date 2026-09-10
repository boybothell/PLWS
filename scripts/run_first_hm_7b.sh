#!/usr/bin/env bash
# 7B seed-42：等到第一扇 High 或 Mix 再压核三词。只用 0-3，不碰 6/7 上的 s1。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
LOG="$AE/results/first_hm_gate/logs"
JOBS="$AE/results/first_hm_gate/jobs/r1_7b_s42.jsonl"
OUT="$AE/results/first_hm_gate"
mkdir -p "$LOG" "$OUT/r1_7b_s42_suppress_hm"

if [[ ! -s "$JOBS" ]]; then
  "$PY" scripts/export_first_hm_jobs.py --model-tag r1_7b --seed 42 --out "$JOBS"
fi
"$PY" scripts/seed_first_hm_scores.py

launch() {
  local shard="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag r1_7b \
    --run-kind hm \
    --seed 42 \
    --lexicon core \
    --shard-id "$shard" \
    --num-shards 4 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 16 \
    --jobs "$JOBS" \
    --out-root "$OUT" \
    > "$LOG/sh${shard}.log" 2>&1 &
  echo "launch first-hm shard=${shard}/4 gpu=${gpu} pid=$!"
}

launch 0 0
launch 1 1
launch 2 2
launch 3 3
wait
echo "first-hm 7B s42 done"
