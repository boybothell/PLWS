#!/usr/bin/env bash
# Sequential dense re-runs on GPUs 2,3,4,5. Long job; run inside tmux.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY="${AE}/.venv/bin/python"
GPUS="${GPUS:-2,3,4,5}"
LOG="${AE}/results/confcal_judge/v2/dense_reprobe.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date -Is) dense reprobe start gpus=${GPUS} ====="

run_analyze() {
  local ds="$1"
  shift
  "$PY" scripts/analyze_dense_probes.py --dataset "$ds" --out "tables/probe_dense_${ds}.md" --scores "$@" || true
}

echo "===== $(date -Is) 7B solver probes: math-500 ====="
"$PY" scripts/run_dense_solver_probes.py --dataset math-500 --gpus "$GPUS"
run_analyze math-500 results/confcal_judge/v2/dense_solver_probes/math-500

echo "===== $(date -Is) 4B forecast: math-500 ====="
"$PY" scripts/run_dense_forecast.py --dataset math-500 --gpus "$GPUS"
run_analyze math-500 \
  results/confcal_judge/v2/dense_solver_probes/math-500 \
  results/confcal_judge/v2/dense_forecast/math-500

echo "===== $(date -Is) 7B solver probes: gpqa-diamond ====="
"$PY" scripts/run_dense_solver_probes.py --dataset gpqa-diamond --gpus "$GPUS"
run_analyze gpqa-diamond results/confcal_judge/v2/dense_solver_probes/gpqa-diamond

echo "===== $(date -Is) 4B forecast: gpqa-diamond ====="
"$PY" scripts/run_dense_forecast.py --dataset gpqa-diamond --gpus "$GPUS"
run_analyze gpqa-diamond \
  results/confcal_judge/v2/dense_solver_probes/gpqa-diamond \
  results/confcal_judge/v2/dense_forecast/gpqa-diamond

echo "===== $(date -Is) 7B solver probes: olympiadbench ====="
"$PY" scripts/run_dense_solver_probes.py --dataset olympiadbench --gpus "$GPUS"
run_analyze olympiadbench results/confcal_judge/v2/dense_solver_probes/olympiadbench

echo "===== $(date -Is) 4B forecast: olympiadbench ====="
"$PY" scripts/run_dense_forecast.py --dataset olympiadbench --gpus "$GPUS"
run_analyze olympiadbench \
  results/confcal_judge/v2/dense_solver_probes/olympiadbench \
  results/confcal_judge/v2/dense_forecast/olympiadbench

echo "===== $(date -Is) 7B challenge: math-500 ====="
"$PY" scripts/run_dense_challenge.py --dataset math-500 --gpus "$GPUS"
run_analyze math-500 \
  results/confcal_judge/v2/dense_solver_probes/math-500 \
  results/confcal_judge/v2/dense_forecast/math-500 \
  results/confcal_judge/v2/dense_challenge/math-500

echo "===== $(date -Is) 7B challenge: gpqa-diamond ====="
"$PY" scripts/run_dense_challenge.py --dataset gpqa-diamond --gpus "$GPUS"
run_analyze gpqa-diamond \
  results/confcal_judge/v2/dense_solver_probes/gpqa-diamond \
  results/confcal_judge/v2/dense_forecast/gpqa-diamond \
  results/confcal_judge/v2/dense_challenge/gpqa-diamond

echo "===== $(date -Is) dense reprobe complete ====="
