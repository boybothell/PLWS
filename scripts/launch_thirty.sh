#!/usr/bin/env bash
# Dynamic 30B-only queue: PUMA (prereq) -> PLWS -> aligned DEER.
# TP=2 on GPU 1,3. Does not start 8B DEER or late_deer. Does not use 5/7.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
SESSION="${SESSION:-plws-thirty}"
GPUS="${GPUS:-1,3}"
FLAGS="${FLAGS:---models qwen3_30b_a3b --until-phase thirty_deer --no-wait-current}"

if [[ ",$GPUS," == *",5,"* || ",$GPUS," == *",7,"* ]]; then
  echo "GPU 5/7 stay unused" >&2
  exit 1
fi
if pgrep -f 'score_leftover_suppress.py.*qwen3_8b.*olympiadbench.*--seed 42' >/dev/null; then
  echo "8B Olympiad leftover-suppress still running; wait before 30B" >&2
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
