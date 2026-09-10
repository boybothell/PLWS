#!/usr/bin/env bash
# PUMA-header Wait re-score. Cards 0-5 only. Never uses 6/7.
# Writes dense_puma_wait. Does not chain half-depth.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-0,1,2,4,5}"
export WAIT_STATUS="${WAIT_STATUS:-wait_puma_status.json}"
export WAIT_MAIN_LOG="${WAIT_MAIN_LOG:-wait_puma_queue.log}"
export WAIT_JOB_DIR="${WAIT_JOB_DIR:-wait_puma_jobs}"
export WAIT_ROOT="${WAIT_ROOT:-$AE/results/confcal_judge/v2/dense_puma_wait}"
LOG="${AE}/results/_logs/${WAIT_MAIN_LOG}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[wait-puma] start $(date -Iseconds) gpus=${GPUS}"
"${AE}/.venv/bin/python" -u scripts/run_wait_vllm_queue.py
echo "[wait-puma] done $(date -Iseconds)"
