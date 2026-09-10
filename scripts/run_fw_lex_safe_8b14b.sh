#!/usr/bin/env bash
# 8B / 14B 第一扇四步同答窗后续写，压扩词表。占空闲卡 4、5。不动 0–3。
# 不从空前缀开写。7B 已齐，不重跑。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
ROOT="${AE}/results/fw_lex_safe"
LOG="${ROOT}/logs"
mkdir -p "$LOG"

pack_jobs() {
  local tag="$1"
  local out="${ROOT}/${tag}_s42/jobs.jsonl"
  mkdir -p "${ROOT}/${tag}_s42"
  {
    cat "${AE}/results/leftover_jump/${tag}_s42/jobs.jsonl"
    cat "${AE}/results/leftover_jump/${tag}_s42/jobs_high.jsonl"
    cat "${AE}/results/leftover_jump/${tag}_s42/jobs_mix.jsonl"
  } > "$out"
  echo "jobs ${tag} $(wc -l < "$out") -> $out"
}

run_one() {
  local tag="$1"
  local gpu="$2"
  local jobs="${ROOT}/${tag}_s42/jobs.jsonl"
  echo "launch ${tag} fw-lex-safe gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag "$tag" \
    --lexicon safe \
    --seed 42 \
    --shard-id 0 \
    --num-shards 1 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "$jobs" \
    --out-root "$ROOT" \
    > "${LOG}/${tag}_fw_lex_safe.log" 2>&1
  echo "done ${tag}"
}

pack_jobs nemotron_8b
pack_jobs r1_14b

run_one nemotron_8b 4 > "${LOG}/gpu4_queue.log" 2>&1 &
echo "queue gpu4 8B pid=$!"
run_one r1_14b 5 > "${LOG}/gpu5_queue.log" 2>&1 &
echo "queue gpu5 14B pid=$!"
wait
echo "8B/14B first-window lex-safe suppress done"
