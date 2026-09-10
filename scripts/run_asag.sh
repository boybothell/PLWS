#!/usr/bin/env bash
# Reproduce ASAG vs DEER on MATH-500 subset, 4 GPUs.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY="${AE}/.venv/bin/python"
GPUS="${GPUS:-0,2,3,4}"
MODE="${1:-asag}"
MAXQ="${MAXQ:-80}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N="${#GPU_ARR[@]}"
OUTD="${AE}/results/asag_r1_7b/math-500/${MODE}"
mkdir -p "${OUTD}/logs"
echo "asag ${MODE} maxq=${MAXQ} gpus=${GPUS}"
pids=()
i=0
for gpu in "${GPU_ARR[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_asag.py \
    --mode "$MODE" --max-questions "$MAXQ" \
    --shard-id "$i" --num-shards "$N" \
    --out "${OUTD}/shard${i}.jsonl" \
    > "${OUTD}/logs/shard${i}.log" 2>&1 &
  pids+=("$!")
  echo "launch ${MODE} shard=${i} gpu=${gpu} pid=${pids[-1]}"
  i=$((i + 1))
done
ec=0
for pid in "${pids[@]}"; do
  wait "$pid" || ec=1
done
exit "$ec"
