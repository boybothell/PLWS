#!/usr/bin/env bash
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-0,2,3,4}"
export MAXQ="${MAXQ:-80}"
LOG="${AE}/results/asag_r1_7b/math-500/queue.log"
mkdir -p "$(dirname "$LOG")"
{
  echo "==== asag $(date) ===="
  bash scripts/run_asag.sh asag
  echo "==== deer $(date) ===="
  bash scripts/run_asag.sh deer
  echo "==== analyze $(date) ===="
  "${AE}/.venv/bin/python" scripts/analyze_asag.py
  echo "==== done $(date) ===="
} 2>&1 | tee -a "$LOG"
