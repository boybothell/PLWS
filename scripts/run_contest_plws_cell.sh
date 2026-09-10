#!/usr/bin/env bash
# One contest PLWS first-window cell. Same 32k host as the main table.
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
BATCH_SIZE="${BATCH_SIZE:-8}"

eval "$("$PY" - "$ROOT" "$MODEL_TAG" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import protocol_for
p = protocol_for(sys.argv[2])
print(f"PROTOCOL_ID={p.protocol_id}")
print(f"GENERATION_TOKENS={p.generation_tokens}")
print(f"ANSWER_TOKENS={p.answer_fix_tokens}")
print(f"MAX_CONTEXT={p.max_model_len}")
PY
)"

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

echo "[contest-plws] start $(date -Is) $MODEL_TAG $DATASET seed=$SEED $PROTOCOL_ID gen=$GENERATION_TOKENS gpu=$GPU"
"$PY" "$ROOT/scripts/score_leftover_suppress.py" \
  --mode suppress \
  --model-tag "$MODEL_TAG" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --run-kind "$KIND" \
  --jobs "$jobs" \
  --lexicon core \
  --k 4 \
  --protocol-id "$PROTOCOL_ID" \
  --generation-tokens "$GENERATION_TOKENS" \
  --max-context "$MAX_CONTEXT" \
  --think-tokens 0 \
  --answer-tokens "$ANSWER_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --sampling-seed 20260904 \
  --isolated-output \
  --shard-id "$SHARD_ID" \
  --num-shards "$NUM_SHARDS"
echo "[contest-plws] done $(date -Is) $MODEL_TAG $DATASET seed=$SEED $PROTOCOL_ID"
