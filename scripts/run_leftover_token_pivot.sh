#!/usr/bin/env bash
# 剩窗后推理 token 熵。只用卡 0-5。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY="${AE}/.venv/bin/python"
OUT="${AE}/results/leftover_token_pivot"
mkdir -p "${OUT}/logs"

"$PY" scripts/export_leftover_token_pivot.py

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
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/score_leftover_token_pivot.py \
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

echo "[$(date -Is)] 7B + 8B"
score_model r1_7b 0,1,2 &
pid7=$!
score_model nemotron_8b 3,4,5 &
pid8=$!
wait "$pid7"
wait "$pid8"

echo "[$(date -Is)] Qwen3-4B + 8B"
score_model qwen3_4b 0,1,2 &
pid4=$!
score_model qwen3_8b 3,4,5 &
pidq8=$!
wait "$pid4"
wait "$pidq8"

echo "[$(date -Is)] leftover token pivot scoring done"
"$PY" scripts/report_leftover_token_pivot.py
