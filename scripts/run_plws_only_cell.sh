#!/usr/bin/env bash
# Finish one PLWS cell without generating Full-CoT or running PUMA.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
GPU="${GPU:?set GPU}"

if [[ "$SEED" == 42 ]]; then
  PUMA_DIR="$ROOT/results/baselines/puma/puma_offline_${MODEL_TAG}/$DATASET"
else
  PUMA_DIR="$ROOT/results/baselines/puma/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET"
fi
for required in statistics.json prefixed_answers.json filtered_steps.json answers.json; do
  [[ -f "$PUMA_DIR/$required" ]] || {
    echo "ERROR: PLWS-only cell lacks existing PUMA prerequisite: $PUMA_DIR/$required" >&2
    exit 1
  }
done

TP="$(awk -F',' '{print NF}' <<<"$GPU")"
if [[ "$MODEL_TAG" == qwen3_30b_a3b && "$TP" -ne 2 ]]; then
  echo "ERROR: qwen3_30b_a3b PLWS requires two GPUs, got $GPU" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
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

echo "[plws-only] start $(date -Is) $MODEL_TAG $DATASET seed=$SEED gpu=$GPU"
PLWS_ROOT="$ROOT" MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" SEED="$SEED" \
  GPUS="$GPU" TP="$TP" SEED_LAYOUT=1 PUMA_DIR="$PUMA_DIR" \
  bash "$ROOT/scripts/run_dense_trials_model.sh"

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
  "$ROOT/scripts/export_leftover_suppress_jobs.py" \
  --model-tag "$MODEL_TAG" --seed "$SEED" --datasets "$DATASET" \
  --kinds all --k 4 --lexicon core

for kind in low mix high; do
  jobs="$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$MODEL_TAG/$DATASET/seed_$SEED/jobs/$kind.jsonl"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
    "$ROOT/scripts/score_leftover_suppress.py" \
    --mode suppress --model-tag "$MODEL_TAG" --dataset "$DATASET" \
    --seed "$SEED" --run-kind "$kind" --jobs "$jobs" \
    --lexicon core --k 4 \
    --max-context 37888 --think-tokens 0 --answer-tokens 2048
done
echo "[plws-only] done $(date -Is) $MODEL_TAG $DATASET seed=$SEED"
