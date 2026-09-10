#!/usr/bin/env bash
# Fill the internals needed to transfer the lag door across sets.
# GPQA DoLA (8/14/20/27) → Olympiad DoLA → Olympiad lens → layer-pair replay.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-0,2,3,4}"
LOG="${AE}/results/_logs/lag_internal_fill.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[lag-internal] start $(date -Iseconds) gpus=${GPUS}"

echo "[lag-internal] GPQA DoLA"
GPUS="$GPUS" bash scripts/run_dense_internal_metrics.sh gpqa-diamond

echo "[lag-internal] analyze MATH × GPQA after DoLA"
"${AE}/.venv/bin/python" scripts/analyze_lag_layers.py

echo "[lag-internal] Olympiad DoLA"
GPUS="$GPUS" bash scripts/run_dense_internal_metrics.sh olympiadbench

echo "[lag-internal] Olympiad lens"
GPUS="$GPUS" bash scripts/run_dense_lens.sh olympiadbench

echo "[lag-internal] analyze three sets"
"${AE}/.venv/bin/python" scripts/analyze_lag_layers.py

echo "[lag-internal] done $(date -Iseconds)"
