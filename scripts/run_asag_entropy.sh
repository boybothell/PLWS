#!/usr/bin/env bash
# Extract ASAG attention entropy on PUMA steps. GPUs 0-5 only.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY="${AE}/.venv/bin/python"
"$PY" scripts/export_asag_entropy_cands.py
run_cell() {
  local stem="$1"
  local gpus="$2"
  local cand="${AE}/results/asag_entropy/${stem}.jsonl"
  local outd="${AE}/results/asag_entropy"
  mkdir -p "${outd}/logs"
  IFS=',' read -r -a arr <<< "$gpus"
  local n="${#arr[@]}"
  local i=0
  local pids=()
  for gpu in "${arr[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/score_asag_entropy.py \
      --candidates "$cand" \
      --out "${outd}/${stem}_shard${i}.jsonl" \
      --shard-id "$i" --num-shards "$n" \
      > "${outd}/logs/${stem}_shard${i}.log" 2>&1 &
    pids+=("$!")
    echo "launch ${stem} shard=${i} gpu=${gpu} pid=${pids[-1]}"
    i=$((i + 1))
  done
  local ec=0
  for pid in "${pids[@]}"; do
    wait "$pid" || ec=1
  done
  return "$ec"
}
run_cell "r1_7b_math-500_s42" "0,1,2,3" &
pid_math=$!
run_cell "r1_7b_gpqa-diamond_s42" "4,5" &
pid_gpqa=$!
wait "$pid_math"
wait "$pid_gpqa"
"$PY" scripts/report_asag_entropy_check.py
echo "asag entropy check done"
