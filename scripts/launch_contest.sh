#!/usr/bin/env bash
# Dynamic contest queue: PUMA (prereq) then PLWS. No DEER.
# Pool is five GPUs (0-4). 30B still wants TP=2; leftover single card
# runs 7B / Nemotron / 14B / 4B. Leaves leftover 8B workers alone.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
SESSION="${SESSION:-plws-contest}"
GPUS="${GPUS:-0,1,2,3,4}"
FLAGS="${FLAGS:-}"
MAX_USED_MIB="${MAX_USED_MIB:-800}"

contest_alive() {
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$ROOT" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import contest_queue_alive
raise SystemExit(0 if contest_queue_alive() else 1)
PY
}

idle_gpus() {
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$ROOT" "$GPUS" "$MAX_USED_MIB" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import idle_contest_gpus
pool = [item for item in sys.argv[2].split(",") if item]
idle = idle_contest_gpus(pool, max_used_mib=int(sys.argv[3]))
print(",".join(idle))
PY
}

if contest_alive; then
  echo "contest queue already running"
  exit 0
fi

live_contest_workers() {
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$ROOT" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import _iter_cmdlines
markers = (
    "run_contest_prereq_cell.sh",
    "run_contest_plws_cell.sh",
)
for cmd in _iter_cmdlines():
    if any(part.endswith(marker) for marker in markers for part in cmd):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

IDLE="$(idle_gpus)"
IFS=',' read -r -a IDLE_ARR <<< "${IDLE}"
if [[ -z "$IDLE" ]]; then
  if live_contest_workers; then
    echo "no idle GPU; adopting live contest workers on $GPUS"
  else
    echo "need an idle GPU in $GPUS; idle=${IDLE:-none}" >&2
    exit 1
  fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stale tmux session $SESSION has no contest queue; replacing it" >&2
  tmux kill-session -t "$SESSION"
fi

"$PY" "$ROOT/scripts/run_contest_queue.py" --gpus "$GPUS" $FLAGS --dry-run

tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "while true; do PLWS_ROOT='$ROOT' '$PY' '$ROOT/scripts/run_contest_queue.py' --gpus '$GPUS' $FLAGS; code=\$?; echo exit=\$code; if [ \$code -eq 0 ]; then break; fi; echo restarting in 15s; sleep 15; done; read"
sleep 2
if ! contest_alive; then
  echo "ERROR: tmux $SESSION started but run_contest_queue.py is not alive" >&2
  exit 1
fi
echo "started tmux $SESSION on pool $GPUS idle=$IDLE"
echo "status: $ROOT/results/runs/contest_queue/status.json"
