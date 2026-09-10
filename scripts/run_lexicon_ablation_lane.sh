#!/usr/bin/env bash
# Run one sequential GPU lane of the paired R1-7B lexicon ablation.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
GPU="${1:?GPU index required}"
STAGE="${2:?smoke, screen, screen_hard, or full required}"
shift 2

case "$STAGE" in
  smoke|screen|screen_hard|full)
    THINK_TOKENS=0
    ;;
  *)
    echo "ERROR: stage must be smoke, screen, screen_hard, or full" >&2
    exit 2
    ;;
esac

JOBS="$ROOT/results/experiments/lexicon_ablation/jobs/$STAGE.jsonl"
OUT_ROOT="$ROOT/results/experiments/lexicon_ablation/$STAGE"
LOG_ROOT="$OUT_ROOT/logs"
mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
export PLWS_ROOT="$ROOT"
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

for config in "$@"; do
  mode=suppress
  lexicon="$config"
  if [[ "$config" == free ]]; then
    mode=free
    lexicon=core
  fi
  out_dir="$OUT_ROOT/$config"
  mkdir -p "$out_dir"
  echo "[$(date -Is)] start gpu=$GPU stage=$STAGE config=$config"
  "$PY" "$ROOT/scripts/score_leftover_suppress.py" \
    --mode "$mode" \
    --model-tag r1_7b \
    --lexicon "$lexicon" \
    --seed 42 \
    --run-kind pilot \
    --max-context 37888 \
    --think-tokens "$THINK_TOKENS" \
    --answer-tokens 2048 \
    --batch-size 64 \
    --sampling-seed 20260904 \
    --isolated-output \
    --jobs "$JOBS" \
    --out "$out_dir/scores.jsonl" \
    > "$LOG_ROOT/${config}.log" 2>&1
  echo "[$(date -Is)] done gpu=$GPU stage=$STAGE config=$config"
done
