#!/usr/bin/env bash
# Official MATH-500 + GPQA-Diamond seed-42 firstwin lexicon ablation, one GPU.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
GPU="${1:?GPU index}"
MODEL_TAG="${2:?r1_7b or qwen3_4b}"
shift 2

case "$MODEL_TAG" in
  r1_7b|qwen3_4b) ;;
  *)
    echo "ERROR: paper lexicon lane only runs r1_7b or qwen3_4b" >&2
    exit 2
    ;;
esac

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

LOG_ROOT="$ROOT/results/experiments/lexicon_ablation/paper_mg42/logs"
mkdir -p "$LOG_ROOT"

for lexicon in "$@"; do
  for dataset in math-500 gpqa-diamond; do
    jobs="$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$MODEL_TAG/$dataset/seed_42/jobs/firstwin.jsonl"
    out_dir="$ROOT/results/runs/plws/window_first/k_4/lexicon_${lexicon}/$MODEL_TAG/$dataset/seed_42/scores/firstwin"
    out="$out_dir/shard_0.jsonl"
    mkdir -p "$out_dir"
    log="$LOG_ROOT/${MODEL_TAG}__${lexicon}__${dataset}.log"
    echo "[$(date -Is)] start gpu=$GPU $MODEL_TAG $lexicon $dataset"
    "$PY" "$ROOT/scripts/score_leftover_suppress.py" \
      --mode suppress \
      --model-tag "$MODEL_TAG" \
      --dataset "$dataset" \
      --seed 42 \
      --run-kind firstwin \
      --jobs "$jobs" \
      --lexicon "$lexicon" \
      --k 4 \
      --max-context 37888 \
      --think-tokens 0 \
      --answer-tokens 2048 \
      --batch-size 64 \
      --sampling-seed 20260904 \
      --isolated-output \
      --out "$out" \
      > "$log" 2>&1
    echo "[$(date -Is)] done gpu=$GPU $MODEL_TAG $lexicon $dataset"
  done
done
