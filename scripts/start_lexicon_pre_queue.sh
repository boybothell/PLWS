#!/usr/bin/env bash
# Start the dynamic pre-experiment dispatcher. Workers stay in tmux.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SESSION="${1:-plws-lex-pre-dyn}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: session $SESSION already exists" >&2
  exit 2
fi

cd "$ROOT"
tmux new-session -d -s "$SESSION" -n dispatcher \
  "export PLWS_ROOT='$ROOT' PLWS_LEX_STAGE=pre PYTHONPATH='$ROOT/src'; '$PY' '$ROOT/scripts/run_lexicon_pre_queue.py' --gpus 0,1,2,3,4; exec bash"
echo "started tmux $SESSION"
tmux list-windows -t "$SESSION"
