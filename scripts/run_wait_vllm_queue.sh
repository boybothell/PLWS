#!/usr/bin/env bash
# Dynamic Wait-only queue. Cards 0-5. Never uses 6/7.
# Does not chain half-depth; half waits for Wait/Qwen3 to go idle.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-0,1,2,3,4,5}"
LOG="${AE}/results/_logs/wait_vllm_queue.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[wait-vllm] start $(date -Iseconds) gpus=${GPUS}"
"${AE}/.venv/bin/python" -u scripts/run_wait_vllm_queue.py
echo "[wait-vllm] done $(date -Iseconds)"
