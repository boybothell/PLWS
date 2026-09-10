#!/usr/bin/env bash
# GPU 6: Qwen3-4B AIME24/25 seeds 0/1/123 × core/wait/core+. Seed 42 reused.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
GPU="${1:-6}"
SEEDS="${2:-0,1,123}"
CONFIGS=(core wait core_plus_let_me)

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

IFS=',' read -r -a seed_list <<< "$SEEDS"
for seed in "${seed_list[@]}"; do
  jobs="$ROOT/results/experiments/lexicon_ablation/jobs/aime4s_qwen3_4b_s${seed}.jsonl"
  if [[ ! -f "$jobs" ]]; then
    echo "ERROR: missing jobs $jobs" >&2
    exit 2
  fi
  for config in "${CONFIGS[@]}"; do
    out_dir="$ROOT/results/experiments/lexicon_ablation/aime4s/qwen3_4b/s${seed}/${config}"
    log_dir="$out_dir/logs"
    mkdir -p "$log_dir"
    echo "[$(date -Is)] start gpu=$GPU seed=$seed config=$config"
    "$PY" "$ROOT/scripts/score_leftover_suppress.py" \
      --mode suppress \
      --model-tag qwen3_4b \
      --lexicon "$config" \
      --seed "$seed" \
      --run-kind pilot \
      --max-context 37888 \
      --think-tokens 0 \
      --answer-tokens 2048 \
      --batch-size 64 \
      --sampling-seed 20260904 \
      --isolated-output \
      --jobs "$jobs" \
      --out "$out_dir/scores.jsonl" \
      > "$log_dir/${config}.log" 2>&1
    echo "[$(date -Is)] done gpu=$GPU seed=$seed config=$config"
  done
done
