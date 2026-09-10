#!/usr/bin/env bash
# Wait until the current Wait/Qwen3 queues are idle, drain leftover PUMA Wait,
# then start half-depth. Does not take GPUs 6/7. Does not interrupt running jobs.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${AE}/results/_logs/half_after_wait.log"
mkdir -p "$(dirname "$LOG")"
echo "[half-after-wait] watching $(date -Is)" | tee -a "$LOG"

wait_idle() {
  tmux has-session -t wait-vllm-queue 2>/dev/null && return 1
  tmux has-session -t wait-puma-queue 2>/dev/null && return 1
  pgrep -f 'scripts/run_wait_vllm_queue.py' >/dev/null && return 1
  pgrep -f 'scripts/run_qwen3_solver_cell.sh' >/dev/null && return 1
  pgrep -f 'scripts/score_wait_vllm.py' >/dev/null && return 1
  return 0
}

while ! wait_idle; do
  echo "[half-after-wait] Wait/Qwen3 still running $(date -Is)" >> "$LOG"
  sleep 120
done

echo "[half-after-wait] Wait idle, drain leftover PUMA Wait $(date -Is)" | tee -a "$LOG"
cd "$AE"
export GPUS="${GPUS:-0,1,2,3,4,5}"
export WAIT_STATUS="${WAIT_STATUS:-wait_puma_status.json}"
export WAIT_MAIN_LOG="${WAIT_MAIN_LOG:-wait_puma_queue.log}"
export WAIT_JOB_DIR="${WAIT_JOB_DIR:-wait_puma_jobs}"
export WAIT_ROOT="${WAIT_ROOT:-$AE/results/confcal_judge/v2/dense_puma_wait}"
"${AE}/.venv/bin/python" -u scripts/run_wait_vllm_queue.py
echo "[half-after-wait] Wait drained, start half-depth $(date -Is)" | tee -a "$LOG"
bash scripts/run_half_depth_queue.sh
echo "[half-after-wait] done $(date -Is)" | tee -a "$LOG"
