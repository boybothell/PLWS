#!/usr/bin/env bash
# Create the confirm-matrix tmux session. Lanes stay inside tmux.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SESSION=plws-lex-confirm-v2
LANE="$ROOT/scripts/run_lexicon_confirm_lane.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: session $SESSION already exists" >&2
  exit 2
fi

cd "$ROOT"
tmux new-session -d -s "$SESSION" -n gpu01 \
  "bash '$LANE' 0,1 r1_14b:wait r1_14b:core_plus_let_me; exec bash"
tmux new-window -t "$SESSION" -n gpu2 \
  "bash '$LANE' 2 r1_7b:wait qwen3_4b:wait nemotron_8b:wait nemotron_8b:core_plus_let_me; exec bash"
tmux new-window -t "$SESSION" -n gpu3 \
  "bash '$LANE' 3 r1_7b:core_plus_let_me qwen3_4b:core_plus_let_me; exec bash"
tmux new-window -t "$SESSION" -n gpu4 \
  "bash '$LANE' 4 qwen3_4b:core qwen3_8b:core qwen3_8b:wait qwen3_8b:core_plus_let_me; exec bash"
tmux select-window -t "$SESSION:gpu01"
echo "started tmux $SESSION"
tmux list-windows -t "$SESSION"
