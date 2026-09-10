#!/usr/bin/env bash
# Wait for the current recovery queue, run two runtime pilots, then launch all
# verified Answer Convergence cells. No existing workload is interrupted.
set -uo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python}"
GPUS="${ANSWER_CONVERGENCE_GPUS:-0,1,2,3,4,7}"
CURRENT_STATE="$ROOT/results/runs/dynamic_experiment_queue/recovery_supervisor.json"
RUN_ROOT="$ROOT/results/runs/answer_convergence_queue"
STATE="$RUN_ROOT/supervisor.json"
LOG="$RUN_ROOT/supervisor.log"
QUEUE="$ROOT/scripts/run_answer_convergence_queue.py"

mkdir -p "$RUN_ROOT"
touch "$LOG"

write_state() {
  local state="$1" detail="${2:-}"
  "$PY" - "$STATE" "$state" "$detail" "$GPUS" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "gpus": sys.argv[4].split(","),
}
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
tmp.replace(path)
PY
}

read_current_state() {
  "$PY" - "$CURRENT_STATE" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print("missing")
else:
    try:
        print(json.loads(path.read_text()).get("state", "unknown"))
    except Exception:
        print("invalid")
PY
}

write_state waiting_current_queue
echo "[$(date -Is)] waiting for current recovery queue" >> "$LOG"
while true; do
  current="$(read_current_state)"
  if [[ "$current" == "succeeded" ]]; then
    break
  fi
  if [[ "$current" == "failed" ]]; then
    write_state blocked "current recovery queue failed"
    echo "[$(date -Is)] blocked: current recovery queue failed" >> "$LOG"
    exit 1
  fi
  sleep 30
done

IFS=',' read -r gpu0 gpu1 _ <<<"$GPUS"
gpu1="${gpu1:-$gpu0}"
write_state running_pilots
echo "[$(date -Is)] starting 7B MATH/AIME24 ten-question pilots" >> "$LOG"

MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B \
MODEL_TAG=r1_7b DATASET=math-500 SEED=0 GPU="$gpu0" LIMIT=10 \
OUT="$ROOT/results/baselines/answer_convergence/pilot/r1_7b/math-500/seed_0" \
PLWS_ROOT="$ROOT" PY="$PY" \
bash "$ROOT/scripts/run_answer_convergence_cell.sh" >> "$LOG" 2>&1 &
pilot_math_pid=$!

MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B \
MODEL_TAG=r1_7b DATASET=aime24 SEED=0 GPU="$gpu1" LIMIT=10 \
OUT="$ROOT/results/baselines/answer_convergence/pilot/r1_7b/aime24/seed_0" \
PLWS_ROOT="$ROOT" PY="$PY" \
bash "$ROOT/scripts/run_answer_convergence_cell.sh" >> "$LOG" 2>&1 &
pilot_aime_pid=$!

pilot_failed=0
wait "$pilot_math_pid" || pilot_failed=1
wait "$pilot_aime_pid" || pilot_failed=1
if [[ "$pilot_failed" -ne 0 ]]; then
  write_state pilot_failed "full queue not started"
  echo "[$(date -Is)] pilot failed; full queue not started" >> "$LOG"
  exit 1
fi

write_state running_full_queue
echo "[$(date -Is)] pilots passed; starting full queue" >> "$LOG"
for round in 1 2 3; do
  "$PY" "$QUEUE" --gpus "$GPUS" --max-attempts 2 >> "$LOG" 2>&1
  code=$?
  if [[ "$code" -eq 0 ]]; then
    write_state succeeded "full queue complete in round $round"
    echo "[$(date -Is)] full queue complete" >> "$LOG"
    exit 0
  fi
  echo "[$(date -Is)] full queue round=$round failed code=$code; reconciling" >> "$LOG"
done

write_state failed "full queue exhausted retries"
exit 1
