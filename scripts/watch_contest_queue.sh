#!/usr/bin/env bash
# Relaunch the contest queue if the scheduler dies while work remains.
# Does not touch live cells.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
export GPUS="${GPUS:-0,1,2,3,4}"

contest_alive() {
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$ROOT" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import contest_queue_alive
raise SystemExit(0 if contest_queue_alive() else 1)
PY
}

work_remains() {
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
from plws.contest import _iter_cmdlines
status_path = root / "results" / "runs" / "contest_queue" / "status.json"
markers = ("run_contest_prereq_cell.sh", "run_contest_plws_cell.sh")
if any(
    any(part.endswith(marker) for marker in markers for part in cmd)
    for cmd in _iter_cmdlines()
):
    raise SystemExit(0)
if not status_path.is_file():
    raise SystemExit(0)
status = json.loads(status_path.read_text(encoding="utf-8"))
counts = status.get("counts") or {}
if int(counts.get("pending") or 0) or int(counts.get("running") or 0):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

while true; do
  sleep 20
  if contest_alive; then
    continue
  fi
  if ! work_remains; then
    echo "contest queue idle and finished; watcher stopping"
    break
  fi
  echo "contest queue dead with work left; relaunching on $GPUS"
  bash "$ROOT/scripts/launch_contest.sh" || true
done
