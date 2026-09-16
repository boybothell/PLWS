#!/usr/bin/env bash
# Detached CPU leftover-job export. Keep quoting here so prereq can be edited
# while a GPU cell is still running.
set -euo pipefail

ROOT="${PLWS_ROOT:?set PLWS_ROOT}"
PY="${PLWS_PY:-/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
RUN_ROOT="${CONTEST_RUN_ROOT:-$ROOT/results/runs/contest_queue}"
EXPORT_LOG="$RUN_ROOT/logs/export__${MODEL_TAG}__${DATASET}__s${SEED}.log"

mkdir -p "$(dirname "$EXPORT_LOG")"
echo "[contest-prereq] detach CPU export $(date -Is) log=$EXPORT_LOG"

export PLWS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

nohup "$PY" "$ROOT/scripts/export_leftover_suppress_jobs.py" \
  --model-tag "$MODEL_TAG" \
  --seed "$SEED" \
  --datasets "$DATASET" \
  --kinds firstwin \
  --k 4 \
  --lexicon core \
  >>"$EXPORT_LOG" 2>&1 &
disown || true
