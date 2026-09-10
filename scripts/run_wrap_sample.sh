#!/usr/bin/env bash
# Two single-GPU shards of one model, then the next model.
# GPU 4 = shard 0, GPU 6 = shard 1. Does not touch 0-3/5/7.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
LEXICON=core_plus_but_so_therefore
STAGE=wrap_sample
GPU0="${WRAP_GPU0:-4}"
GPU1="${WRAP_GPU1:-6}"
if [[ -n "${WRAP_MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=($WRAP_MODELS)
else
  MODELS=(qwen3_4b nemotron_8b)
fi

export PLWS_ROOT="$ROOT"
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

run_shard() {
  local gpu="$1"
  local model="$2"
  local shard_id="$3"
  local jobs="$4"
  local out_dir="$ROOT/results/experiments/lexicon_ablation/$STAGE/$model/$LEXICON"
  local log_dir="$out_dir/logs"
  mkdir -p "$log_dir"
  local out="$out_dir/scores_shard${shard_id}.jsonl"
  local log="$log_dir/${LEXICON}_s${shard_id}of2.log"
  echo "[$(date -Is)] start gpu=$gpu model=$model shard=$shard_id/2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/score_leftover_suppress.py" \
    --mode suppress \
    --model-tag "$model" \
    --lexicon "$LEXICON" \
    --run-kind firstwin \
    --k 4 \
    --max-context 37888 \
    --think-tokens 0 \
    --answer-tokens 2048 \
    --batch-size 64 \
    --sampling-seed 20260904 \
    --isolated-output \
    --jobs "$jobs" \
    --out "$out" \
    --shard-id "$shard_id" \
    --num-shards 2 \
    > "$log" 2>&1
  echo "[$(date -Is)] done gpu=$gpu model=$model shard=$shard_id/2"
}

for model in "${MODELS[@]}"; do
  jobs="$ROOT/results/experiments/lexicon_ablation/jobs/wrap_sample_${model}.jsonl"
  if [[ ! -f "$jobs" ]]; then
    echo "ERROR: missing jobs $jobs" >&2
    exit 2
  fi
  echo "[$(date -Is)] model=$model both shards on GPU $GPU0/$GPU1"
  run_shard "$GPU0" "$model" 0 "$jobs" &
  pid0=$!
  run_shard "$GPU1" "$model" 1 "$jobs" &
  pid1=$!
  status=0
  wait "$pid0" || status=1
  wait "$pid1" || status=1
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: $model shard failed; not starting the next model" >&2
    exit 1
  fi
  echo "[$(date -Is)] model=$model both shards finished"
done
echo "[$(date -Is)] wrap sample queue finished"
