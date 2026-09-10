#!/usr/bin/env bash
# One PLWS first-window shard under puma-fullcot-32k-v2.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set dataset}"
SEED="${SEED:?set seed}"
KIND="${KIND:-firstwin}"
GPU="${GPU:?set one GPU or a TP lane}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"

export PLWS_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(
  "$PY" - <<'PY'
from pathlib import Path
root = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
print(":".join(sorted({
    str(path)
    for path in (root / ".venv" / "lib").glob("**/nvidia/*/lib")
    if path.is_dir()
})))
PY
):${LD_LIBRARY_PATH:-}"

jobs="$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$MODEL_TAG/$DATASET/seed_$SEED/jobs/firstwin.jsonl"
if [[ ! -f "$jobs" ]]; then
  echo "ERROR: missing jobs $jobs" >&2
  exit 1
fi

echo "[matrix-plws] start $(date -Is) $MODEL_TAG $DATASET seed=$SEED kind=$KIND shard=$SHARD_ID/$NUM_SHARDS gpu=$GPU"
"$PY" "$ROOT/scripts/score_leftover_suppress.py" \
  --mode suppress \
  --model-tag "$MODEL_TAG" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --run-kind "$KIND" \
  --jobs "$jobs" \
  --lexicon core \
  --k 4 \
  --max-context 37888 \
  --think-tokens 0 \
  --answer-tokens 2048 \
  --batch-size "$BATCH_SIZE" \
  --sampling-seed 20260904 \
  --isolated-output \
  --shard-id "$SHARD_ID" \
  --num-shards "$NUM_SHARDS"
echo "[matrix-plws] done $(date -Is) $MODEL_TAG $DATASET seed=$SEED kind=$KIND"
