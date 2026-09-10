#!/usr/bin/env bash
# Full-CoT sample with PUMA Step1a params → samples/<model_tag>/<dataset>/seed_<seed>/
#
# Aligns with repos/PUMA/configs/*.conf + puma/run_vllm.py:
#   max_tokens=32768, answer_fix=2048,
#   temperature/top_p/top_k from model generation_config
#   (top_k=None → -1), prompt-version=default
#   TP = number of GPUs in CUDA_VISIBLE_DEVICES
#
# Usage:
#   GPU=1 DATASET=aime24 SEED=0 bash scripts/run_puma_aligned_sample.sh
#   GPU=0,1 MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-32B \
#     MODEL_TAG=r1_32b DATASET=math-500 SEED=123 bash scripts/run_puma_aligned_sample.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUMA_ROOT="$(cd "$ROOT/../PUMA" && pwd)"
PY="${PY:-/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python}"
MODEL="${MODEL:-/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_TAG="${MODEL_TAG:-r1_7b}"
ALIGN_CONF="${ALIGN_CONF:-DS-7B.conf}"
DATASET="${DATASET:?set DATASET}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
ANSWER_FIX="${ANSWER_FIX:-2048}"
PROMPT_RESERVE="${PROMPT_RESERVE:-3072}"
PROTOCOL_ID="${PROTOCOL_ID:-puma-fullcot-32k-v2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_TOKENS + PROMPT_RESERVE + ANSWER_FIX))}"
PROMPT_VERSION="${PROMPT_VERSION:-default}"

# New layout (canonical)
OUT="${OUT:-$ROOT/samples/$MODEL_TAG/$DATASET/seed_${SEED}}"
# Legacy flat tag (compat symlink target name)
LEGACY_TAG="sample_${MODEL_TAG}_${DATASET//-/_}"
if [[ "$SEED" != "42" ]]; then
  LEGACY_TAG="${LEGACY_TAG}_s${SEED}"
fi
LEGACY_LINK="$ROOT/samples/$LEGACY_TAG"
BENCH="$PUMA_ROOT/data/${DATASET}_test.jsonl"

if [[ ! -f "$BENCH" ]]; then
  echo "ERROR: missing benchmark $BENCH"
  exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "ERROR: model not found: $MODEL"
  exit 1
fi

export VLLM_LENS_DISABLE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="$(
python3 - <<'PY'
from pathlib import Path
root = Path('/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm')
print(':'.join(sorted({str(p) for p in (root/'.venv'/'lib').glob('**/nvidia/*/lib') if p.is_dir()})))
PY
):${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
META="$OUT/sample_meta.json"
LOG="$OUT/sample.log"
TP=$(awk -F',' '{print NF}' <<<"$GPU")

if [[ -f "$OUT/answers.json" ]]; then
  if "$PY" - "$META" "$PROTOCOL_ID" "$MAX_TOKENS" "$ANSWER_FIX" "$PROMPT_RESERVE" "$MAX_MODEL_LEN" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
meta = json.loads(path.read_text())
valid = (
    meta.get("protocol_id") == sys.argv[2]
    and int(meta.get("max_tokens") or 0) == int(sys.argv[3])
    and int(meta.get("answer_fix_max_tokens") or 0) == int(sys.argv[4])
    and int(meta.get("prompt_reserve_tokens") or 0) == int(sys.argv[5])
    and int(meta.get("max_model_len") or 0) == int(sys.argv[6])
    and meta.get("prompt_version") == "default"
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "[skip] canonical $OUT/answers.json exists"
    exit 0
  fi
  echo "ERROR: existing sample is not $PROTOCOL_ID: $OUT" >&2
  exit 1
fi

cat > "$META" <<EOF
{
  "protocol_id": "$PROTOCOL_ID",
  "tag": "$LEGACY_TAG",
  "path": "$MODEL_TAG/$DATASET/seed_${SEED}",
  "layout": "samples/<model_tag>/<dataset>/seed_<seed>",
  "dataset": "$DATASET",
  "benchmark": "$BENCH",
  "model": "$MODEL",
  "model_tag": "$MODEL_TAG",
  "align": "PUMA $ALIGN_CONF Step1a",
  "seed": $SEED,
  "max_tokens": $MAX_TOKENS,
  "answer_fix_max_tokens": $ANSWER_FIX,
  "prompt_reserve_tokens": $PROMPT_RESERVE,
  "max_model_len": $MAX_MODEL_LEN,
  "prompt_version": "$PROMPT_VERSION",
  "temperature": "from generation_config",
  "top_p": "from generation_config",
  "tensor_parallel": $TP,
  "cuda_visible_devices": "$GPU",
  "n_questions": $(wc -l < "$BENCH")
}
EOF

# compat symlink at old flat path
if [[ -e "$LEGACY_LINK" || -L "$LEGACY_LINK" ]]; then
  if [[ ! -L "$LEGACY_LINK" ]]; then
    echo "WARNING: legacy path exists and is not a symlink: $LEGACY_LINK"
  fi
else
  ln -s "$OUT" "$LEGACY_LINK"
fi

echo "[sample] GPU=$GPU TP=$TP model_tag=$MODEL_TAG dataset=$DATASET seed=$SEED -> $OUT"
echo "[sample] start $(date -Is)" | tee "$LOG"

cd "$PUMA_ROOT"
"$PY" puma/run_vllm.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --dataset_path "$BENCH" \
  --output_path "$OUT/answers.json" \
  --max_tokens "$MAX_TOKENS" \
  --answer_max_tokens "$ANSWER_FIX" \
  --max_model_len "$MAX_MODEL_LEN" \
  --prompt-version "$PROMPT_VERSION" \
  --seed "$SEED" \
  2>&1 | tee -a "$LOG"

echo "[sample] done $(date -Is)" | tee -a "$LOG"
echo "[sample] wrote $OUT/answers.json"
