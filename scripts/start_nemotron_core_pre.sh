#!/usr/bin/env bash
# Nemotron pre-exp core, two 1-GPU shards on the cards freed from AIME 4-seed.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SESSION="${1:-plws-nemo-core}"
LANE="$ROOT/scripts/run_lexicon_confirm_lane.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: session $SESSION already exists" >&2
  exit 2
fi

cd "$ROOT"
tmux new-session -d -s "$SESSION" -n s0 \
  "export PLWS_ROOT='$ROOT' PLWS_LEX_STAGE=pre SHARD_ID=0 NUM_SHARDS=2 PYTHONPATH='$ROOT/src'; bash '$LANE' 2 nemotron_8b:core; exec bash"
tmux new-window -t "$SESSION" -n s1 \
  "export PLWS_ROOT='$ROOT' PLWS_LEX_STAGE=pre SHARD_ID=1 NUM_SHARDS=2 PYTHONPATH='$ROOT/src'; bash '$LANE' 6 nemotron_8b:core; exec bash"
echo "started tmux $SESSION on GPU 2 and 6"
tmux list-windows -t "$SESSION"
