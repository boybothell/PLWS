#!/usr/bin/env bash
# Main-table 随频次 for non-R1 small models (everything below 30B that is
# not already covered by the R1 count-bias queues).
#
# Locked host cell: RHOS=0.98, seeds 42/0/1, main five datasets.
# Does not start by itself — call with START=1 after GPUs are free.
#
# Preflight only (default):
#   bash scripts/run_count_bias_small.sh
#
# Start queue in tmux (when you explicitly want it):
#   START=1 bash scripts/run_count_bias_small.sh
# Default pool is GPUs 2,5 (TP=1 each).
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PLWS_ROOT="$ROOT"

MODELS="${MODELS:-nemotron_8b,qwen3_4b,qwen3_8b}"
SEEDS="${SEEDS:-42,0,1}"
DATASETS="${DATASETS:-amc23,aime25,gpqa-diamond,math-500,olympiadbench}"
RHOS="${RHOS:-0.98}"
GPUS="${GPUS:-2,5}"
TMUX_NAME="${TMUX_NAME:-plws-count-small}"
START="${START:-0}"
STATUS_JSON="${COUNT_BIAS_STATUS:-$ROOT/results/runs/plws/count_bias/queue_status_small.json}"

echo "count-bias small MODELS=$MODELS SEEDS=$SEEDS DATASETS=$DATASETS RHOS=$RHOS GPUS=$GPUS"
MODELS="$MODELS" SEEDS="$SEEDS" DATASETS="$DATASETS" RHOS="$RHOS" \
  bash "$ROOT/scripts/check_count_bias_prereq.sh"

if [[ "$START" != "1" ]]; then
  echo "preflight ok; not starting (set START=1 to launch tmux $TMUX_NAME on GPUS=$GPUS)"
  exit 0
fi

if [[ -z "${GPUS}" ]]; then
  echo "set GPUS, for example 2,5" >&2
  exit 1
fi

if tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
  echo "tmux session already exists: $TMUX_NAME" >&2
  exit 1
fi

mkdir -p "$(dirname "$STATUS_JSON")"
LOG="$ROOT/results/runs/plws/count_bias/queue_logs/${TMUX_NAME}.$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$TMUX_NAME" bash -lc "
  set -euo pipefail
  cd \"$ROOT\"
  export PLWS_ROOT=\"$ROOT\"
  export COUNT_BIAS_STATUS=\"$STATUS_JSON\"
  export COUNT_BIAS_ORDER=\"\${COUNT_BIAS_ORDER:-model}\"
  echo \"# \$(date -Is) start $TMUX_NAME GPUS=$GPUS MODELS=$MODELS RHOS=$RHOS\" | tee -a \"$LOG\"
  GPUS=\"$GPUS\" MODELS=\"$MODELS\" SEEDS=\"$SEEDS\" DATASETS=\"$DATASETS\" RHOS=\"$RHOS\" \
    bash scripts/run_count_bias_queue.sh 2>&1 | tee -a \"$LOG\"
  echo \"# \$(date -Is) exit=\$?\" | tee -a \"$LOG\"
"

echo "started tmux=$TMUX_NAME gpus=$GPUS status=$STATUS_JSON log=$LOG"
tmux ls
