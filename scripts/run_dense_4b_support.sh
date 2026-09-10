#!/usr/bin/env bash
# 4B PMI / margin / evidence-gain. Default MATH, vLLM.
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
DS="${1:-math-500}"
VLLM_ROOT="/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm"
PY="${VLLM_ROOT}/.venv/bin/python"
CAND="${AE}/results/confcal_judge/v2/dense_candidates/${DS}.jsonl"
OUTD="${AE}/results/confcal_judge/v2/dense_4b_support/${DS}"
GPUS="${GPUS:-0,3,5}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N="${#GPU_ARR[@]}"
mkdir -p "${OUTD}/logs"
EXTRA=""
for lib in $(find "${VLLM_ROOT}/.venv/lib" -type d -path '*/nvidia/*/lib' | sort); do
  EXTRA="${EXTRA:+$EXTRA:}${lib}"
done
export VLLM_LENS_DISABLE=1
export LD_LIBRARY_PATH="${EXTRA}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "4b-support ${DS} n_cand=$(wc -l < "$CAND") gpus=${GPUS} backend=vllm"
pids=()
i=0
for gpu in "${GPU_ARR[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    "$PY" scripts/score_dense_4b_support.py \
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
"${AE}/.venv/bin/python" scripts/analyze_dense_4b_support.py --dataset "$DS" --scores "$OUTD" \
  --out "${AE}/tables/probe_dense_4b_support_${DS}.md"
echo "4b-support ${DS} done"
