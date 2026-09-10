#!/usr/bin/env bash
# 写完终答隐状态。卡 0-5。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY="${AE}/.venv/bin/python"
OUT="${AE}/results/leftover_final_h"
mkdir -p "${OUT}/logs"

"$PY" scripts/export_leftover_final_h.py

score_model() {
  local tag="$1"
  local gpus="$2"
  local cand="${OUT}/${tag}.jsonl"
  if [[ ! -s "$cand" ]]; then
    echo "skip ${tag}: no candidates"
    return 0
  fi
  IFS=',' read -r -a arr <<< "$gpus"
  local n="${#arr[@]}"
  local i=0
  local pids=()
  for gpu in "${arr[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/score_leftover_waithelp.py \
      --candidates "$cand" \
      --out "${OUT}/${tag}_shard${i}.jsonl" \
      --model-tag "$tag" \
      --shard-id "$i" --num-shards "$n" \
      > "${OUT}/logs/${tag}_shard${i}.log" 2>&1 &
    pids+=("$!")
    echo "launch ${tag} shard=${i} gpu=${gpu} pid=${pids[-1]}"
    i=$((i + 1))
  done
  local ec=0
  for pid in "${pids[@]}"; do
    wait "$pid" || ec=1
  done
  return "$ec"
}

echo "[$(date -Is)] 7B + 8B + Qwen3-4B"
score_model r1_7b 0,1 &
score_model nemotron_8b 2,3 &
score_model qwen3_4b 4,5 &
wait

echo "[$(date -Is)] 14B + Qwen3-8B"
score_model r1_14b 0,1,2,3 &
score_model qwen3_8b 4,5 &
wait

echo "[$(date -Is)] leftover final-h scoring done"
