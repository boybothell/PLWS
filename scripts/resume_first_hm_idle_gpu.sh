#!/usr/bin/env bash
# GPU5 已空：立刻接下 s1/s123 的 shard1。GPU4 仍在收 s0 shard0，收完再接 s1/s123 shard0。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
OUT="$AE/results/first_hm_gate"
LOG="$OUT/logs"

run_shard() {
  local seed="$1" shard="$2" gpu="$3"
  echo "launch s${seed} shard=${shard} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress --model-tag r1_7b --run-kind hm --seed "$seed" --lexicon core \
    --shard-id "$shard" --num-shards 2 --max-context 32768 --think-tokens 0 --batch-size 16 \
    --jobs "$OUT/jobs/r1_7b_s${seed}.jsonl" --out-root "$OUT" \
    >> "$LOG/s${seed}_sh${shard}.log" 2>&1
}

gpu5() {
  for seed in 1 123; do
    run_shard "$seed" 1 5
  done
  echo "gpu5 done"
}

gpu4() {
  while pgrep -f "score_leftover_suppress.py.*--seed 0 .*--shard-id 0 " >/dev/null; do
    echo "gpu4 wait s0 shard0"
    sleep 10
  done
  for seed in 1 123; do
    run_shard "$seed" 0 4
  done
  echo "gpu4 done"
}

gpu5 &
gpu4 &
wait
echo "resume first-hm idle gpu done"
