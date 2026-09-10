#!/usr/bin/env bash
# Half-depth extract. Cards 0-5. Never uses 6/7.
# Wait until PUMA Wait / leftover Wait jobs are idle, then start.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-0,1,2,3,4,5}"
DONE="${AE}/results/_logs/half_depth_done"
LOG="${AE}/results/_logs/half_depth_queue.log"
mkdir -p "$(dirname "$LOG")"

wait_jobs_idle() {
  tmux has-session -t wait-puma-queue 2>/dev/null && return 1
  pgrep -f 'scripts/run_wait_vllm_queue.py' >/dev/null && return 1
  pgrep -f 'scripts/score_wait_vllm.py' >/dev/null && return 1
  pgrep -f 'scripts/run_qwen3_solver_cell.sh' >/dev/null && return 1
  return 0
}

while ! wait_jobs_idle; do
  echo "[half-depth] Wait/Qwen3 still running $(date -Is)" | tee -a "$LOG"
  sleep 120
done

if pgrep -f 'scripts/run_half_depth_queue.py' >/dev/null; then
  echo "[half-depth] already running, skip" | tee -a "$LOG"
  exit 0
fi
if [[ -f "$DONE" ]]; then
  echo "[half-depth] already finished ($(cat "$DONE")), skip" | tee -a "$LOG"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1
echo "[half-depth] start $(date -Iseconds) gpus=${GPUS}"
"${AE}/.venv/bin/python" -u scripts/run_half_depth_queue.py
echo "$$ $(date -Is)" > "$DONE"
echo "[half-depth] done $(date -Iseconds)"
