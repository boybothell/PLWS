#!/usr/bin/env bash
# CPU waiter: after the last 8B Olympiad PLWS shards exit and GPU 1/3 are
# free, start the 30B PUMA + PLWS + DEER matrix. Does not touch 8B workers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLL="${POLL:-30}"
MAX_USED_MIB="${MAX_USED_MIB:-800}"

eight_alive() {
  pgrep -f 'score_leftover_suppress.py.*qwen3_8b.*olympiadbench.*--seed 42' >/dev/null
}

gpu_used() {
  local idx="$1"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v idx="$idx" '$1+0==idx {gsub(/ /,"",$2); print $2+0}'
}

echo "[wait-thirty] $(date -Is) waiting for 8B Olympiad leftover on GPU 1/3"
while eight_alive; do
  echo "[wait-thirty] $(date -Is) 8B leftover still running"
  sleep "$POLL"
done

echo "[wait-thirty] $(date -Is) leftover gone; waiting GPU 1/3 below ${MAX_USED_MIB} MiB"
while true; do
  used1="$(gpu_used 1)"
  used3="$(gpu_used 3)"
  echo "[wait-thirty] $(date -Is) gpu1=${used1}MiB gpu3=${used3}MiB"
  if [[ "${used1:-99999}" -lt "$MAX_USED_MIB" && "${used3:-99999}" -lt "$MAX_USED_MIB" ]]; then
    break
  fi
  sleep "$POLL"
done

echo "[wait-thirty] $(date -Is) launching 30B matrix"
exec bash "$ROOT/scripts/launch_thirty.sh"
