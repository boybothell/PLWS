#!/usr/bin/env bash
# One canonical Qwen cell:
# Full-CoT sample -> PUMA -> dense upstream -> PLWS jobs -> PLWS suppress scores.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
MODEL_TAG="${MODEL_TAG:?qwen3_4b|qwen3_8b|qwen3_30b_a3b}"
DATASET="${DATASET:?set dataset}"
SEED="${SEED:?set seed}"
GPU="${GPU:?set one GPU or a comma-separated TP lane}"

case "$MODEL_TAG" in
  qwen3_4b)
    MODEL=/mnt/d/lsj/models/Qwen3-4B
    ALIGN_CONF=DS-7B.conf
    EXPECTED_TP=1
    ;;
  qwen3_8b)
    MODEL=/mnt/d/lsj/models/Qwen3-8B
    ALIGN_CONF=DS-7B.conf
    EXPECTED_TP=1
    ;;
  qwen3_30b_a3b)
    MODEL=/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507
    ALIGN_CONF=Q30B-T.conf
    EXPECTED_TP=2
    ;;
  *)
    echo "ERROR: unsupported MODEL_TAG=$MODEL_TAG" >&2
    exit 1
    ;;
esac

TP="$(awk -F',' '{print NF}' <<<"$GPU")"
if [[ "$TP" -ne "$EXPECTED_TP" ]]; then
  echo "ERROR: $MODEL_TAG requires TP=$EXPECTED_TP, got GPU=$GPU" >&2
  exit 1
fi

# The sampling/PUMA/dense helpers set this in child shells, but those exports
# do not propagate back here.  The final PLWS scorer imports vLLM directly,
# so establish and verify the CUDA runtime path at the cell boundary.
export VLLM_LENS_DISABLE=1
export CUDA_VISIBLE_DEVICES="$GPU"
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

echo "[qwen-plws] preflight vLLM/CUDA runtime"
"$PY" - <<'PY'
import vllm
print(f"[qwen-plws] vLLM import ok version={vllm.__version__}")
PY

if [[ "$SEED" == 42 ]]; then
  PUMA_DIR="$ROOT/results/baselines/puma/puma_offline_${MODEL_TAG}/$DATASET"
else
  PUMA_DIR="$ROOT/results/baselines/puma/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET"
fi

echo "[qwen-plws] start $(date -Is) $MODEL_TAG $DATASET seed=$SEED gpu=$GPU"

PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
  ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
  bash "$ROOT/scripts/run_puma_aligned_sample.sh"

PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
  ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
  PUMA_DIR="$PUMA_DIR" \
  bash "$ROOT/scripts/run_puma_official.sh"

PLWS_ROOT="$ROOT" MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" SEED="$SEED" \
  GPUS="$GPU" TP="$TP" SEED_LAYOUT=1 PUMA_DIR="$PUMA_DIR" \
  bash "$ROOT/scripts/run_dense_trials_model.sh"

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
  "$ROOT/scripts/export_leftover_suppress_jobs.py" \
  --model-tag "$MODEL_TAG" --seed "$SEED" --datasets "$DATASET" \
  --kinds all --k 4 --lexicon core

for kind in low mix high; do
  jobs="$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$MODEL_TAG/$DATASET/seed_$SEED/jobs/$kind.jsonl"
  CUDA_VISIBLE_DEVICES="$GPU" PLWS_ROOT="$ROOT" \
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
    "$ROOT/scripts/score_leftover_suppress.py" \
    --mode suppress --model-tag "$MODEL_TAG" --dataset "$DATASET" \
    --seed "$SEED" --run-kind "$kind" --jobs "$jobs" \
    --lexicon core --k 4 \
    --max-context 37888 --think-tokens 0 --answer-tokens 2048
done

echo "[qwen-plws] done $(date -Is) $MODEL_TAG $DATASET seed=$SEED"

