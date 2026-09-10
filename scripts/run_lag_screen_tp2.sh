#!/usr/bin/env bash
# 32B / 30B lag-door extract on a 2-GPU pair. Default 1,5. Do not use 0/2/3/4.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-1,5}"
LOG="${AE}/results/_logs/lag_screen_tp2.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[lag-screen-tp2] start $(date -Iseconds) gpus=${GPUS}"
"${AE}/.venv/bin/python" -u scripts/run_lag_screen_tp2.py
echo "[lag-screen-tp2] done $(date -Iseconds)"
