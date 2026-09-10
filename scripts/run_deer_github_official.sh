#!/usr/bin/env bash
# Direct runner for the unmodified iie-ycx/DEER GitHub implementation.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DEER_ROOT=/mnt/d/lsj/visual-latent-tts/repos/DEER
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
MODEL="${MODEL:?set MODEL}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET=math|olympiadbench|gpqa|aime|aime25}"
GPU="${GPU:?set GPU}"
FAMILY="${FAMILY:-standard}"
OUT="${OUT:-$ROOT/results/baselines/deer/github_official_greedy_16k/$MODEL_TAG/$DATASET}"
BATCH_SIZE="${DEER_BATCH_SIZE:-32}"

[[ -f "$DEER_ROOT/.git/HEAD" ]] || {
  echo "ERROR: official DEER checkout missing: $DEER_ROOT" >&2
  exit 1
}
[[ -f "$DEER_ROOT/data/$DATASET/test.jsonl" ]] || {
  echo "ERROR: official DEER dataset missing: $DEER_ROOT/data/$DATASET/test.jsonl" >&2
  exit 1
}
[[ -f "$MODEL/config.json" ]] || {
  echo "ERROR: model missing: $MODEL" >&2
  exit 1
}

if [[ -n "$(git -C "$DEER_ROOT" status --short)" ]]; then
  echo "ERROR: refusing to run a modified DEER checkout" >&2
  exit 1
fi

case "$FAMILY" in
  standard)
    ENTRY="$DEER_ROOT/vllm-deer.py"
    THINK_RATIO=0.6
    POLICY=avg1
    ;;
  qwen3)
    ENTRY="$DEER_ROOT/vllm-deer-qwen3.py"
    THINK_RATIO=0.8
    POLICY=avg2
    ;;
  *)
    echo "ERROR: unknown FAMILY=$FAMILY" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
export LD_LIBRARY_PATH="$(
  "$PY" - <<'PY'
from pathlib import Path
root = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
print(":".join(sorted({str(p) for p in (root / ".venv" / "lib").glob("**/nvidia/*/lib") if p.is_dir()})))
PY
):${LD_LIBRARY_PATH:-}"

cd "$DEER_ROOT"
"$PY" "$ENTRY" \
  --model_name_or_path "$MODEL" \
  --dataset_dir "$DEER_ROOT/data" \
  --output_path "$OUT" \
  --dataset "$DATASET" \
  --threshold 0.95 \
  --max_generated_tokens 16000 \
  --think_ratio "$THINK_RATIO" \
  --policy "$POLICY" \
  --temperature 0.0 \
  --top_p 1.0 \
  --batch_size "$BATCH_SIZE" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code

cat > "$OUT/manifest.json" <<EOF
{
  "method": "DEER",
  "source": "https://github.com/iie-ycx/DEER",
  "source_commit": "$(git -C "$DEER_ROOT" rev-parse HEAD)",
  "entrypoint": "$(basename "$ENTRY")",
  "model_tag": "$MODEL_TAG",
  "dataset": "$DATASET",
  "decoding": {"temperature": 0.0, "top_p": 1.0, "max_generated_tokens": 16000},
  "deer": {"threshold": 0.95, "think_ratio": $THINK_RATIO, "policy": "$POLICY"},
  "batch_size": $BATCH_SIZE,
  "cuda_visible_devices": "$GPU"
}
EOF
