#!/usr/bin/env bash
# 7B：已有官方 / 窗后压轨迹上 2× Wait。默认 tmux。
# 两卡数据并行：S1_SHARDS=2，一张卡一个 shard，不要对 7B 开 TP=2。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
S1_SHARD="${S1_SHARD:-0}"
S1_SHARDS="${S1_SHARDS:-1}"
mkdir -p results/s1_on_traj/logs results/s1_on_traj/jobs
if [[ "$S1_SHARD" == "0" ]]; then
  "$PY" scripts/export_s1_on_traj_jobs.py --model-tag r1_7b --seeds 42,0,1,123
else
  while [[ ! -f results/s1_on_traj/jobs/r1_7b_s42_official.jsonl ]]; do sleep 1; done
fi
for seed in 42 0 1 123; do
  for src in official window; do
    echo "[s1-traj] r1_7b s${seed} ${src} shard=${S1_SHARD}/${S1_SHARDS}"
    "$PY" scripts/score_s1_budget_force.py \
      --model-tag r1_7b \
      --src "$src" \
      --seed "$seed" \
      --num-ignore 2 \
      --shard-id "$S1_SHARD" \
      --num-shards "$S1_SHARDS" \
      --jobs "results/s1_on_traj/jobs/r1_7b_s${seed}_${src}.jsonl" \
      --datasets math-500,olympiadbench,gpqa-diamond,aime24,aime25 \
      --max-context 32768 \
      --think-tokens 0 \
      --batch-size 8 \
      --gpu-mem-util 0.88 \
      --out-root results/s1_on_traj
  done
done
echo "[s1-traj] 7B shard ${S1_SHARD}/${S1_SHARDS} done"
