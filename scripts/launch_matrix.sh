#!/usr/bin/env bash
# Start the first-window PLWS-core + unified-host DEER matrix in tmux.
# Waits for the current pre / AIME queues before claiming GPUs.
# Default is 1,3. Stop after Qwen3-8B PLWS; do not start 8B DEER or 30B.
# The last 8B Olympiad s42 half is a manual 4-way split; do not launch
# the matrix while those leftover-suppress workers are alive.
# Does not launch if plws-matrix exists.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
SESSION=plws-matrix
GPUS="${GPUS:-1,3}"
FLAGS="${FLAGS:---no-thirty --until-phase eight_plws}"

if pgrep -f 'score_leftover_suppress.py.*qwen3_8b.*olympiadbench.*--seed 42' >/dev/null; then
  echo "8B Olympiad s42 leftover-suppress still running; do not start matrix" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists" >&2
  tmux ls -F '#{session_name} #{session_attached}' | grep "^$SESSION " || true
  exit 1
fi

"$PY" "$ROOT/scripts/run_matrix_queue.py" --gpus "$GPUS" $FLAGS --dry-run

tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "PLWS_ROOT='$ROOT' '$PY' '$ROOT/scripts/run_matrix_queue.py' --gpus '$GPUS' $FLAGS; echo exit=\$?; read"
echo "started tmux $SESSION on GPUs $GPUS"
echo "status: $ROOT/results/runs/matrix_queue/status.json"
