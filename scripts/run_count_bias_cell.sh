#!/usr/bin/env bash
# Count-adaptive post-lock CORE bias. One cell. Does not write 窗后压 scores.
#
# The comparison arm is the existing firstwin CORE ban: after the lock,
# every CORE token has logit -inf. This script does not regenerate it.
#
#   N_pre  = CORE sequences completed in the lock prefix
#   N_post = CORE sequences completed in the continuation so far
#   beta   = -PEAK * N_post / (N_pre + N_post + 1)
#
# beta is added only to a token that would complete a CORE sequence.
# The first post-lock CORE token sees N_post = 0 and is not penalized.
#
#   GPU=4 DATASET=math-500 SEED=42 \
#     bash scripts/run_count_bias_cell.sh
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_TAG="${MODEL_TAG:-r1_7b}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
GPU="${GPU:?set GPU}"
PEAK="${PEAK:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-0}"

# shellcheck source=lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"
plws_runtime_init

eval "$("$PY" - "$ROOT" "$MODEL_TAG" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import protocol_for
protocol = protocol_for(sys.argv[2])
print(f"PROTOCOL_ID={protocol.protocol_id}")
print(f"GENERATION_TOKENS={protocol.generation_tokens}")
print(f"ANSWER_TOKENS={protocol.answer_fix_tokens}")
print(f"MAX_CONTEXT={protocol.max_model_len}")
PY
)"

export PLWS_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
plws_export_cuda_runtime

jobs="$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$MODEL_TAG/$DATASET/seed_$SEED/jobs/firstwin.jsonl"
out="${OUT:-$ROOT/results/runs/plws/count_bias/$MODEL_TAG/$DATASET/seed_$SEED/shard_0.jsonl}"
limit_args=()
if [[ "$LIMIT" != "0" ]]; then
  limit_args=(--limit "$LIMIT")
fi
if [[ ! -f "$jobs" ]]; then
  echo "ERROR: missing jobs $jobs" >&2
  exit 1
fi

echo "start gpus=$GPU schedule=count $MODEL_TAG $DATASET seed=$SEED peak=$PEAK limit=$LIMIT $(date -Is)"
"$PY" "$ROOT/scripts/score_leftover_suppress.py" \
  --mode suppress \
  --model-tag "$MODEL_TAG" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --run-kind firstwin \
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
  --bias-schedule count \
  --bias-peak "$PEAK" \
  "${limit_args[@]}" \
  --out "$out" \
  --shard-id 0 \
  --num-shards 1
echo "done gpus=$GPU schedule=count $MODEL_TAG $DATASET seed=$SEED $(date -Is)"
