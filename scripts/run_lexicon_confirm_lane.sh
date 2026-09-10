#!/usr/bin/env bash
# One GPU/TP lane of the five-dataset lexicon confirm matrix.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
STAGE="${PLWS_LEX_STAGE:-confirm}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
GPU="${1:?GPU index or comma-separated TP lane required}"
shift

if [[ "$#" -lt 1 ]]; then
  echo "ERROR: need at least one model:config spec" >&2
  exit 2
fi

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

for spec in "$@"; do
  model="${spec%%:*}"
  config="${spec#*:}"
  if [[ -z "$model" || -z "$config" || "$model" == "$spec" ]]; then
    echo "ERROR: expected model:config, got $spec" >&2
    exit 2
  fi
  jobs="$ROOT/results/experiments/lexicon_ablation/jobs/${STAGE}_${model}.jsonl"
  if [[ ! -f "$jobs" ]]; then
    echo "ERROR: missing jobs $jobs" >&2
    exit 2
  fi
  batch=64
  if [[ "$model" == r1_14b ]]; then
    batch=4
  fi
  out_dir="$ROOT/results/experiments/lexicon_ablation/$STAGE/$model/$config"
  log_dir="$out_dir/logs"
  mkdir -p "$log_dir"
  shard_args=()
  shard_id="${SHARD_ID:-}"
  num_shards="${NUM_SHARDS:-1}"
  out="$out_dir/scores.jsonl"
  log_name="$config"
  if [[ -n "$shard_id" && "$num_shards" -gt 1 ]]; then
    shard_args=(--shard-id "$shard_id" --num-shards "$num_shards")
    out="$out_dir/scores_shard${shard_id}.jsonl"
    log_name="${config}_s${shard_id}of${num_shards}"
  fi
  echo "[$(date -Is)] start gpu=$GPU model=$model config=$config shards=${shard_id:-0}/${num_shards}"
  "$PY" "$ROOT/scripts/score_leftover_suppress.py" \
    --mode suppress \
    --model-tag "$model" \
    --lexicon "$config" \
    --seed 42 \
    --run-kind pilot \
    --max-context 37888 \
    --think-tokens 0 \
    --answer-tokens 2048 \
    --batch-size "$batch" \
    --sampling-seed 20260904 \
    --isolated-output \
    --jobs "$jobs" \
    --out "$out" \
    "${shard_args[@]}" \
    > "$log_dir/${log_name}.log" 2>&1
  echo "[$(date -Is)] done gpu=$GPU model=$model config=$config"
done
