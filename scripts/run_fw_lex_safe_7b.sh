#!/usr/bin/env bash
# 7B 第一扇四步同答窗后续写，压扩词表（Wait/Alternatively/Hmm + However/Maybe/another way/double-check/Hold on）。
# 只占空闲卡 4、5。不动 0–3。不从空前缀开写。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
ROOT="${AE}/results/fw_lex_safe"
LOG="${ROOT}/logs"
JOBS="${ROOT}/r1_7b_s42/jobs.jsonl"
mkdir -p "$LOG" "${ROOT}/r1_7b_s42"

{
  cat "${AE}/results/leftover_jump/r1_7b_s42/jobs.jsonl"
  cat "${AE}/results/leftover_jump/r1_7b_s42/jobs_high.jsonl"
  cat "${AE}/results/leftover_jump/r1_7b_s42/jobs_mix.jsonl"
} > "$JOBS"
echo "jobs $(wc -l < "$JOBS") -> $JOBS"

run_shard() {
  local gpu="$1"
  local shard="$2"
  echo "launch 7B fw-lex-safe gpu=${gpu} shard=${shard}"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag r1_7b \
    --lexicon safe \
    --seed 42 \
    --shard-id "$shard" \
    --num-shards 2 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "$JOBS" \
    --out-root "$ROOT" \
    > "${LOG}/r1_7b_fw_lex_safe_shard${shard}.log" 2>&1
  echo "done shard=${shard}"
}

run_shard 4 0 > "${LOG}/gpu4_queue.log" 2>&1 &
echo "queue gpu4 pid=$!"
run_shard 5 1 > "${LOG}/gpu5_queue.log" 2>&1 &
echo "queue gpu5 pid=$!"
wait
echo "7B first-window lex-safe suppress done"
