#!/usr/bin/env bash
# CPU waiter: start contest as soon as one TP=2 pair is idle.
# Leaves leftover 8B workers on busy GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python}"
POLL="${POLL:-30}"
MAX_USED_MIB="${MAX_USED_MIB:-800}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SESSION="${SESSION:-plws-contest}"
LOG="${LOG:-$ROOT/results/runs/contest_queue/waiter.log}"

mkdir -p "$(dirname "$LOG")"

say() {
  local line="[wait-contest] $(date -Is) $*"
  echo "$line" | tee -a "$LOG"
}

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
print(",".join(idle_contest_gpus(pool, max_used_mib=int(sys.argv[3]))))
PY
}

say "contest starts on the first idle TP=2 pair in $GPUS"

while true; do
  if contest_alive; then
    say "contest queue already running"
    exit 0
  fi
  IDLE="$(idle_gpus)"
  IFS=',' read -r -a IDLE_ARR <<< "${IDLE}"
  if [[ -z "$IDLE" || "${#IDLE_ARR[@]}" -lt 2 ]]; then
    say "waiting for an idle pair; idle=${IDLE:-none}"
    sleep "$POLL"
    continue
  fi
  say "idle $IDLE; launching contest lane"
  if GPUS="$GPUS" SESSION="$SESSION" bash "$ROOT/scripts/launch_contest.sh"; then
    sleep 2
    if contest_alive; then
      say "contest queue is up"
      exit 0
    fi
    say "launch returned but contest queue is not up; retry"
  else
    say "launch failed; retry in ${POLL}s"
  fi
  sleep "$POLL"
done
