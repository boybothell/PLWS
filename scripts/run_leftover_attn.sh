#!/usr/bin/env bash
# 7B leftover first-window: question attention + Wait layer contrast. GPUs 0-5.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY="${AE}/.venv/bin/python"
"$PY" scripts/export_leftover_attn_cands.py
run_cell() {
  local stem="$1"
  local gpus="$2"
  local cand="${AE}/results/leftover_attn/${stem}.jsonl"
  local outd="${AE}/results/leftover_attn"
  mkdir -p "${outd}/logs"
  IFS=',' read -r -a arr <<< "$gpus"
  local n="${#arr[@]}"
  local i=0
  local pids=()
  for gpu in "${arr[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/score_leftover_attn.py \
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
run_cell "r1_7b_math-500" "0,1,2" &
pid_math=$!
run_cell "r1_7b_gpqa-diamond" "3,4,5" &
pid_gpqa=$!
wait "$pid_math"
wait "$pid_gpqa"
"$PY" scripts/report_leftover_attn.py
echo "leftover attn done"
