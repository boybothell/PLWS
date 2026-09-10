#!/usr/bin/env bash
# Full-dense Qwen3-4B answer confidence + logit lens. Default MATH, 4 GPUs, HF.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
DS="${1:-math-500}"
PY="${AE}/.venv/bin/python"
CAND="${AE}/results/confcal_judge/v2/dense_candidates/${DS}.jsonl"
OUTD="${AE}/results/confcal_judge/v2/dense_4b_lens/${DS}"
GPUS="${GPUS:-2,3,4,5}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N="${#GPU_ARR[@]}"
mkdir -p "${OUTD}/logs"
echo "4b-lens ${DS} n_cand=$(wc -l < "$CAND") gpus=${GPUS} backend=hf"
pids=()
i=0
for gpu in "${GPU_ARR[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/score_dense_4b_lens.py \
    --candidates "$CAND" \
    --out "${OUTD}/scores_shard${i}.jsonl" \
    --shard-id "$i" --num-shards "$N" \
    > "${OUTD}/logs/shard${i}.log" 2>&1 &
  pids+=("$!")
  echo "launch shard=${i} gpu=${gpu} pid=${pids[-1]}"
  i=$((i + 1))
done
ec=0
for pid in "${pids[@]}"; do
  wait "$pid" || ec=1
done
if [[ "$ec" -ne 0 ]]; then
  echo "a shard failed" >&2
  exit 1
fi
"$PY" scripts/analyze_dense_4b_lens.py --dataset "$DS" --scores "$OUTD" \
  --out "${AE}/tables/probe_dense_4b_lens_${DS}.md"
echo "4b-lens ${DS} done"
