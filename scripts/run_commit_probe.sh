#!/usr/bin/env bash
# Leftover commit-probe on GPUs 0-5. 7B MATH + 7B GPQA only.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
AE_PY="${AE}/.venv/bin/python"
VLLM_PY="/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python"
export VLLM_LENS_DISABLE=1
export LD_LIBRARY_PATH="$("$AE_PY" - <<'PY'
from pathlib import Path
root = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
print(":".join(sorted({str(p) for p in (root / ".venv" / "lib").glob("**/nvidia/*/lib") if p.is_dir()})))
PY
):${LD_LIBRARY_PATH:-}"
"$AE_PY" scripts/export_leftover_commit.py
run_cell() {
  local stem="$1"
  local dataset="$2"
  local gpus="$3"
  local cand="${AE}/results/commit_probe/${stem}.jsonl"
  mkdir -p "${AE}/results/commit_probe/logs"
  IFS=',' read -r -a arr <<< "$gpus"
  local n="${#arr[@]}"
  local i=0
  local pids=()
  for gpu in "${arr[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_PY" scripts/score_commit_probe.py \
      --candidates "$cand" \
      --out "${AE}/results/commit_probe/${stem}_shard${i}.jsonl" \
      --dataset "$dataset" \
      --shard-id "$i" --num-shards "$n" \
      > "${AE}/results/commit_probe/logs/${stem}_shard${i}.log" 2>&1 &
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
run_cell "r1_7b_math-500_s42" "math-500" "0,1" &
pid_math=$!
run_cell "r1_7b_gpqa-diamond_s42" "gpqa-diamond" "2,3,4,5" &
pid_gpqa=$!
wait "$pid_math"
wait "$pid_gpqa"
"$AE_PY" scripts/report_commit_probe.py
echo "commit probe done"
