#!/usr/bin/env bash
# Wait for the legacy in-memory queue, then reconcile artifacts and retry only
# incomplete cells with the current scheduler and wrappers.
set -uo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
QUEUE="$ROOT/scripts/run_dynamic_experiment_queue.py"
OLD_PID="${OLD_QUEUE_PID:-4071503}"
GPUS="${DYNAMIC_GPUS:-0,1,2,3,4,7}"
MAX_ROUNDS="${RECOVERY_MAX_ROUNDS:-3}"
RUN_ROOT="$ROOT/results/runs/dynamic_experiment_queue"
LOG="$RUN_ROOT/recovery_supervisor.log"
STATE="$RUN_ROOT/recovery_supervisor.json"

write_state() {
  local state="$1" round="${2:-0}" code="${3:-}"
  "$PY" - "$STATE" "$state" "$round" "$code" "$OLD_PID" "$GPUS" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "state": sys.argv[2],
    "round": int(sys.argv[3]),
    "last_returncode": int(sys.argv[4]) if sys.argv[4] else None,
    "old_queue_pid": int(sys.argv[5]),
    "gpus": sys.argv[6].split(","),
}
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
tmp.replace(path)
PY
}

mkdir -p "$RUN_ROOT"
touch "$LOG"
write_state waiting
echo "[$(date -Is)] waiting for queue pid=$OLD_PID" >> "$LOG"

while [[ -r "/proc/$OLD_PID/cmdline" ]]; do
  cmdline="$(tr '\0' ' ' < "/proc/$OLD_PID/cmdline" 2>/dev/null || true)"
  [[ "$cmdline" == *"run_dynamic_experiment_queue.py"* ]] || break
  sleep 30
done

echo "[$(date -Is)] legacy queue ended; starting artifact reconciliation" >> "$LOG"
for round in $(seq 1 "$MAX_ROUNDS"); do
  write_state running "$round"
  echo "[$(date -Is)] recovery round=$round gpus=$GPUS" >> "$LOG"
  "$PY" "$QUEUE" --gpus "$GPUS" --max-attempts 2 >> "$LOG" 2>&1
  code=$?
  if [[ "$code" -eq 0 ]]; then
    write_state succeeded "$round" "$code"
    echo "[$(date -Is)] recovery completed round=$round" >> "$LOG"
    exit 0
  fi
  write_state retrying "$round" "$code"
  echo "[$(date -Is)] recovery round=$round failed code=$code" >> "$LOG"
  sleep 30
done

write_state failed "$MAX_ROUNDS" 1
echo "[$(date -Is)] recovery exhausted rounds=$MAX_ROUNDS" >> "$LOG"
exit 1
