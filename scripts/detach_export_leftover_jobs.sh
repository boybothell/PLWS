#!/usr/bin/env bash
# CPU leftover-job export. Run in the foreground and wait for firstwin.jsonl.
# Dense already released the engine; leftover starts next and must see jobs.
# Keep this path so running prereq cells still pick up the wait.
set -euo pipefail

ROOT="${PLWS_ROOT:?set PLWS_ROOT}"
PY="${PLWS_PY:-/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
RUN_ROOT="${CONTEST_RUN_ROOT:-$ROOT/results/runs/contest_queue}"
EXPORT_LOG="$RUN_ROOT/logs/export__${MODEL_TAG}__${DATASET}__s${SEED}.log"
JOBS="$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$MODEL_TAG/$DATASET/seed_$SEED/jobs/firstwin.jsonl"

mkdir -p "$(dirname "$EXPORT_LOG")"
echo "[contest-prereq] CPU export $(date -Is) log=$EXPORT_LOG"

export PLWS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$ROOT/scripts/export_leftover_suppress_jobs.py" \
  --model-tag "$MODEL_TAG" \
  --seed "$SEED" \
  --datasets "$DATASET" \
  --kinds firstwin \
  --k 4 \
  --lexicon core \
  >>"$EXPORT_LOG" 2>&1

if [[ ! -f "$JOBS" ]]; then
  echo "ERROR: export finished but missing jobs $JOBS" >&2
  exit 1
fi
echo "[contest-prereq] CPU export ready $(date -Is) $JOBS"
