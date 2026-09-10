#!/usr/bin/env bash
# 7B High/Mix 门控跑完后，接着 8B/14B 多种子窗后压，再续 14B 开局压。
# 只占 0–3，不碰 6/7 上的 s1。必须 tmux 起。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
export WIN4_GPUS="${WIN4_GPUS:-0,1,2,3}"
export FROMSTART_GPUS="${FROMSTART_GPUS:-0,1,2,3}"
LOG="$AE/results/window4_mid/logs/hm_then_win.log"
WINLOG="$AE/results/window4_mid/logs/dyn4.log"
FSLOG="$AE/results/fromstart_core/logs/dyn4.log"
DONE_RE='window4 8b+14b done fail_d=0 fail_s=0'
mkdir -p "$(dirname "$LOG")" "$(dirname "$FSLOG")"

say() { echo "[resume4] $* $(date -Is)" | tee -a "$LOG"; }

hm_busy() {
  pgrep -f 'scripts/run_first_hm_7b.sh' >/dev/null && return 0
  pgrep -f 'score_leftover_suppress.py --mode suppress --model-tag r1_7b --run-kind hm' >/dev/null && return 0
  return 1
}

hm_pending() {
  "$PY" "$AE/scripts/count_first_hm_pending.py"
}

say "wait first-hm pending=$(hm_pending)"
while hm_busy; do
  sleep 20
done
say "first-hm processes gone pending=$(hm_pending)"

if [[ "$(hm_pending)" -gt 0 ]]; then
  say "retry first-hm leftover"
  bash "$AE/scripts/run_first_hm_7b.sh" 2>&1 | tee -a "$AE/results/first_hm_gate/logs/run.log"
  say "retry finished pending=$(hm_pending)"
fi

if [[ "$(hm_pending)" -gt 0 ]]; then
  say "first-hm still pending=$(hm_pending); continue multi-seed anyway"
fi

if pgrep -f 'scripts/run_window4_mid_dyn.py' >/dev/null; then
  say "absorb gpu0: restart window mixer on 0-3"
  pkill -f 'scripts/run_window4_mid_dyn.py' || true
  pkill -f 'score_leftover_suppress.py --mode suppress --model-tag nemotron_8b' || true
  pkill -f 'score_leftover_suppress.py --mode suppress --model-tag r1_14b' || true
  sleep 4
fi

export WIN4_GPUS=0,1,2,3
say "start window4 8b+14b gpus=$WIN4_GPUS"
bash "$AE/scripts/run_window4_mid.sh" 2>&1 | tee -a "$WINLOG"
say "window4 mixer exited"

if ! grep -q "$DONE_RE" "$WINLOG"; then
  say "window4 did not finish clean; not starting fromstart"
  tail -n 40 "$WINLOG" | tee -a "$LOG"
  exit 1
fi

say "start fromstart leftover gpus=$FROMSTART_GPUS"
bash "$AE/scripts/run_fromstart_core_6gpu.sh" 2>&1 | tee -a "$FSLOG"
say "all 0-3 handoff done"
