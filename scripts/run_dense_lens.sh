#!/usr/bin/env bash
# Full-dense logit-lens. Default: r1_7b MATH, 4 GPUs.
#   GPUS=0,2,3,4 bash scripts/run_dense_lens.sh gpqa-diamond
#   MODEL_TAG=r1_14b MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B \
#     GPUS=0,2,3,4 bash scripts/run_dense_lens.sh math-500
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
DS="${1:-math-500}"
TAG="${MODEL_TAG:-r1_7b}"
PY="${AE}/.venv/bin/python"
if [[ "$TAG" == "r1_7b" && "$DS" != aime24 && "$DS" != aime25 ]]; then
  CAND="${CAND:-${AE}/results/confcal_judge/v2/dense_candidates/${DS}.jsonl}"
  OUTD="${OUTD:-${AE}/results/confcal_judge/v2/dense_lens/${DS}}"
else
  CAND="${CAND:-${AE}/results/confcal_judge/v2/dense_candidates/${TAG}/${DS}.jsonl}"
  OUTD="${OUTD:-${AE}/results/confcal_judge/v2/dense_lens/${TAG}/${DS}}"
fi
MODEL_ARG=()
if [[ -n "${MODEL:-}" ]]; then
  MODEL_ARG=(--model "$MODEL")
fi
GPUS="${GPUS:-0,2,3,4}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N="${#GPU_ARR[@]}"
mkdir -p "${OUTD}/logs"
echo "dense-lens tag=${TAG} ${DS} n_cand=$(wc -l < "$CAND") gpus=${GPUS}"
pids=()
i=0
for gpu in "${GPU_ARR[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/score_first_af_lens.py \
    --candidates "$CAND" \
    --out "${OUTD}/scores_shard${i}.jsonl" \
    --shard-id "$i" --num-shards "$N" \
    "${MODEL_ARG[@]}" \
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
ANALYZE_SEED=()
if [[ -n "${SEED:-}" ]]; then
  ANALYZE_SEED=(--seed "$SEED")
fi
"$PY" scripts/analyze_dense_layer_curve.py \
  --scores "$OUTD" --dataset "$DS" --model-tag "$TAG" \
  "${ANALYZE_SEED[@]}" \
  --out "${AE}/tables/probe_layer_curve_${TAG}_${DS}.md"
echo "dense-lens ${TAG} ${DS} done"
