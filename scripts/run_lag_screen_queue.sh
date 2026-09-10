#!/usr/bin/env bash
# Dynamic 4-GPU lag-door screen queue. Cards 0/2/3/4, never idle until empty.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-0,2,3,4}"
LOG="${AE}/results/_logs/lag_screen_queue.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[lag-screen] start $(date -Iseconds) gpus=${GPUS}"
"${AE}/.venv/bin/python" -u scripts/run_lag_screen_queue.py
echo "[lag-screen] done $(date -Iseconds)"
