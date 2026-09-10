#!/usr/bin/env bash
# 7B other datasets, then 14B MATH. Max 4 GPUs.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-0,2,3,4}"
PY="${AE}/.venv/bin/python"
mkdir -p "${AE}/results/confcal_judge/v2/dense_lens/logs"
LOG="${AE}/results/confcal_judge/v2/dense_lens/logs/queue.log"

{
  echo "==== dump 14B MATH / 7B AIME $(date) ===="
  "$PY" scripts/dump_dense_candidates.py --dataset math-500 --model-tag r1_14b
  "$PY" scripts/dump_dense_candidates.py --dataset aime24 --model-tag r1_7b --seed 1
  "$PY" scripts/dump_dense_candidates.py --dataset aime25 --model-tag r1_7b --seed 1

  echo "==== 7B gpqa-diamond $(date) ===="
  bash scripts/run_dense_lens.sh gpqa-diamond

  echo "==== 7B olympiadbench $(date) ===="
  bash scripts/run_dense_lens.sh olympiadbench

  echo "==== 7B aime24 seed1 $(date) ===="
  SEED=1 CAND="${AE}/results/confcal_judge/v2/dense_candidates/r1_7b/aime24_s1.jsonl" \
    OUTD="${AE}/results/confcal_judge/v2/dense_lens/r1_7b/aime24_s1" \
    bash scripts/run_dense_lens.sh aime24

  echo "==== 7B aime25 seed1 $(date) ===="
  SEED=1 CAND="${AE}/results/confcal_judge/v2/dense_candidates/r1_7b/aime25_s1.jsonl" \
    OUTD="${AE}/results/confcal_judge/v2/dense_lens/r1_7b/aime25_s1" \
    bash scripts/run_dense_lens.sh aime25

  echo "==== 14B MATH $(date) ===="
  MODEL_TAG=r1_14b MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B \
    bash scripts/run_dense_lens.sh math-500

  echo "==== queue done $(date) ===="
} 2>&1 | tee -a "$LOG"
