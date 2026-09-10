#!/usr/bin/env bash
# One Qwen3-4B / Qwen3-8B cell: sample → official PUMA → dense trials → Wait.
# Thinking is on (PUMA qwen3 chat template). Do not use Thinking-2507.
#
#   MODEL_TAG=qwen3_4b DATASET=math-500 SEED=42 GPU=0 bash scripts/run_qwen3_solver_cell.sh
set -euo pipefail

AE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python}"
AE_PY="${AE_PY:-$AE/.venv/bin/python}"
MODEL_TAG="${MODEL_TAG:?qwen3_4b|qwen3_8b}"
DATASET="${DATASET:?}"
SEED="${SEED:-42}"
GPU="${GPU:?}"

case "$MODEL_TAG" in
  qwen3_4b) MODEL=/mnt/d/lsj/models/Qwen3-4B ;;
  qwen3_8b) MODEL=/mnt/d/lsj/models/Qwen3-8B ;;
  *) echo "MODEL_TAG must be qwen3_4b or qwen3_8b" >&2; exit 1 ;;
esac

ALIGN_CONF=DS-7B.conf
export VLLM_LENS_DISABLE=1
export PY AE_PY

echo "[qwen3-cell] start $(date -Is) $MODEL_TAG $DATASET seed=$SEED gpu=$GPU"

MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" ALIGN_CONF="$ALIGN_CONF" \
  GPU="$GPU" DATASET="$DATASET" SEED="$SEED" \
  bash "$AE/scripts/run_puma_aligned_sample.sh"

MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" ALIGN_CONF="$ALIGN_CONF" \
  GPU="$GPU" DATASET="$DATASET" SEED="$SEED" \
  bash "$AE/scripts/run_puma_official.sh"

MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" GPUS="$GPU" TP=1 SEED="$SEED" \
  bash "$AE/scripts/run_dense_trials_model.sh"

DUMP_SEED=""
if [[ "$DATASET" == aime24 || "$DATASET" == aime25 ]]; then
  DUMP_SEED="$SEED"
fi
if [[ -n "$DUMP_SEED" ]]; then
  "$AE_PY" "$AE/scripts/dump_dense_candidates.py" \
    --dataset "$DATASET" --model-tag "$MODEL_TAG" --seed "$DUMP_SEED"
  CAND="$AE/results/confcal_judge/v2/dense_candidates/$MODEL_TAG/${DATASET}_s${DUMP_SEED}.jsonl"
  WAIT_OUT="$AE/results/confcal_judge/v2/dense_puma_wait/$MODEL_TAG/${DATASET}_s${DUMP_SEED}/scores_shard0.jsonl"
else
  "$AE_PY" "$AE/scripts/dump_dense_candidates.py" \
    --dataset "$DATASET" --model-tag "$MODEL_TAG"
  CAND="$AE/results/confcal_judge/v2/dense_candidates/$MODEL_TAG/${DATASET}.jsonl"
  WAIT_OUT="$AE/results/confcal_judge/v2/dense_puma_wait/$MODEL_TAG/${DATASET}/scores_shard0.jsonl"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="$(
python3 - <<'PY'
from pathlib import Path
root = Path('/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm')
print(':'.join(sorted({str(p) for p in (root/'.venv'/'lib').glob('**/nvidia/*/lib') if p.is_dir()})))
PY
):${LD_LIBRARY_PATH:-}"

"$PY" -u "$AE/scripts/score_wait_vllm.py" \
  --candidates "$CAND" \
  --out "$WAIT_OUT" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prompt-mode puma \
  --shard-id 0 \
  --num-shards 1 \
  --max-context 16384 \
  --batch-size 16 \
  --gpu-mem-util 0.90 \
  --tp 1

echo "[qwen3-cell] done $(date -Is) $MODEL_TAG $DATASET seed=$SEED"
