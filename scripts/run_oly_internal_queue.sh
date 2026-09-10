#!/usr/bin/env bash
# Extract the same 7B internals on OlympiadBench as MATH, then re-run rescue-R.
# Order: stop-margin / DoLA / lookback → logit-lens → PMI / margin.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
export GPUS="${GPUS:-0,2,3,4}"
LOG="${AE}/results/confcal_judge/v2/oly_internal_queue.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[oly-internal] start $(date -Iseconds) gpus=${GPUS}"

echo "[oly-internal] dense_internal"
GPUS="$GPUS" bash scripts/run_dense_internal_metrics.sh olympiadbench

echo "[oly-internal] dense_lens"
GPUS="$GPUS" bash scripts/run_dense_lens.sh olympiadbench

echo "[oly-internal] dense_solver_probes"
/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python \
  scripts/run_dense_solver_probes.py --dataset olympiadbench --gpus "$GPUS"

echo "[oly-internal] replay rescue-R"
"${AE}/.venv/bin/python" scripts/replay_rescue_R_gate.py

echo "[oly-internal] done $(date -Iseconds)"
