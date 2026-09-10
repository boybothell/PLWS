#!/usr/bin/env bash
# 等 8B/14B 窗后压正常结束，再续 14B 没跑完的开局压。只用 0–3，不抢 6 上的 s1，不碰 7。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
LOG="$AE/results/fromstart_core/logs/handoff.log"
DONE_RE='window4 8b+14b done fail_d=0 fail_s=0'
mkdir -p "$(dirname "$LOG")"
echo "[handoff] wait window4 mixer to start $(date -Is)" | tee -a "$LOG"
while ! pgrep -f 'scripts/run_window4_mid_dyn.py' >/dev/null; do
  sleep 20
done
echo "[handoff] window mixer up $(date -Is)" | tee -a "$LOG"
while pgrep -f 'scripts/run_window4_mid_dyn.py' >/dev/null; do
  sleep 20
done
echo "[handoff] window mixer gone $(date -Is)" | tee -a "$LOG"
WINLOG=""
for f in "$AE/results/window4_mid/logs/dyn4.log" "$AE/results/window4_mid/logs/dyn6.log"; do
  if [[ -f "$f" ]] && grep -q "$DONE_RE" "$f"; then
    WINLOG="$f"
    break
  fi
done
if [[ -z "$WINLOG" ]]; then
  echo "[handoff] window4 mid did not finish clean; not starting fromstart" | tee -a "$LOG"
  tail -n 30 "$AE/results/window4_mid/logs/dyn4.log" "$AE/results/window4_mid/logs/dyn6.log" 2>/dev/null | tee -a "$LOG"
  exit 1
fi
echo "[handoff] clean done in $WINLOG" | tee -a "$LOG"
echo "[handoff] start fromstart $(date -Is)" | tee -a "$LOG"
exec bash "$AE/scripts/run_fromstart_core_6gpu.sh"
