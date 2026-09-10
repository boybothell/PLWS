#!/usr/bin/env bash
# AIME seed 0/1/123：第一扇窗三档只压 Wait。占卡 0、1，不碰 2–3 上的 32B。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
LOG="${AE}/results/leftover_suppress_toend/logs_aime_seeds"
OUT="${AE}/results/leftover_suppress_toend"
mkdir -p "$LOG"

run_one() {
  local tag="$1"
  local seed="$2"
  local kind="$3"
  local gpu="$4"
  local jobs
  if [[ "$kind" == "low" ]]; then
    jobs="${AE}/results/leftover_jump/${tag}_s${seed}/jobs.jsonl"
  else
    jobs="${AE}/results/leftover_jump/${tag}_s${seed}/jobs_${kind}.jsonl"
  fi
  if [[ ! -s "$jobs" ]]; then
    echo "skip ${tag} s${seed} ${kind}: no jobs"
    return 0
  fi
  local n
  n="$(grep -c . "$jobs" || true)"
  echo "launch ${tag} s${seed} ${kind} n=${n} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag "$tag" \
    --run-kind "$kind" \
    --seed "$seed" \
    --shard-id 0 \
    --num-shards 1 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "$jobs" \
    --out-root "$OUT" \
    > "${LOG}/${tag}_s${seed}_${kind}_suppress.log" 2>&1
  echo "done ${tag} s${seed} ${kind}"
}

queue() {
  local gpu="$1"
  shift
  local tag
  for tag in "$@"; do
    local seed kind
    for seed in 0 1 123; do
      for kind in low high mix; do
        run_one "$tag" "$seed" "$kind" "$gpu"
      done
    done
  done
}

{
  queue 0 r1_7b nemotron_8b
} > "${LOG}/gpu0_queue.log" 2>&1 &
echo "queue gpu0 pid=$!"

{
  queue 1 r1_14b
} > "${LOG}/gpu1_queue.log" 2>&1 &
echo "queue gpu1 pid=$!"

wait
echo "aime extra-seed suppress done"
