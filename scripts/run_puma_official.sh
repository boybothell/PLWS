#!/usr/bin/env bash
# Official PUMA compress+regen from an already-sampled Full-CoT answers.json.
#
#   MODEL=/mnt/d/lsj/models/Qwen3-4B MODEL_TAG=qwen3_4b DATASET=math-500 \
#     SEED=42 GPU=0 bash scripts/run_puma_official.sh
set -euo pipefail

AE="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PLWS_ROOT="$AE"
# shellcheck source=lib/runtime.sh
ROOT="$AE"
source "$AE/scripts/lib/runtime.sh"
plws_runtime_init
PUMA_ROOT="$(cd "$PUMA_ROOT" && pwd)"
export PYTHONPATH="$AE/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${MODEL:?set MODEL}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
ALIGN_CONF="${ALIGN_CONF:-DS-7B.conf}"

if [[ "$SEED" != "42" ]]; then
  PUMA_DIR="${PUMA_DIR:-$AE/results/baselines/puma/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET}"
else
  PUMA_DIR="${PUMA_DIR:-$AE/results/baselines/puma/puma_offline_${MODEL_TAG}/$DATASET}"
fi
SAMPLE="${SAMPLE:-$AE/samples/$MODEL_TAG/$DATASET/seed_${SEED}}"
BENCH="${BENCH:-$PUMA_ROOT/data/${DATASET}_test.jsonl}"

mkdir -p "$PUMA_DIR"
RUN_FINISHED=0
write_status() {
  local state="$1" message="${2:-}"
  "$PY" - "$PUMA_DIR/status.json" "$state" "$message" <<'PY'
import sys
from plws.artifacts import atomic_write_json, utc_now
atomic_write_json(sys.argv[1], {
    "schema_version": 1,
    "state": sys.argv[2],
    "message": sys.argv[3] or None,
    "updated_at": utc_now(),
})
PY
}
on_exit() {
  local code=$?
  if [[ "$RUN_FINISHED" != "1" ]]; then
    write_status failed "run_puma_official.sh exited with code $code" || true
  fi
}
trap on_exit EXIT
"$PY" - "$PUMA_DIR/manifest.json" "$MODEL_TAG" "$MODEL" "$DATASET" "$SEED" "$SAMPLE" "$BENCH" <<'PY'
import sys
from plws.artifacts import atomic_write_json, utc_now
atomic_write_json(sys.argv[1], {
    "schema_version": 1,
    "method": "puma",
    "experiment": "official",
    "model_tag": sys.argv[2],
    "model": sys.argv[3],
    "dataset": sys.argv[4],
    "seed": int(sys.argv[5]),
    "sample": sys.argv[6],
    "benchmark": sys.argv[7],
    "created_at": utc_now(),
})
PY
write_status running

if [[ -f "$PUMA_DIR/statistics.json" && -f "$PUMA_DIR/prefixed_answers.json" ]]; then
  write_status succeeded "existing complete output reused"
  RUN_FINISHED=1
  echo "[puma-official] skip complete $MODEL_TAG $DATASET seed=$SEED"
  exit 0
fi
if [[ ! -f "$SAMPLE/answers.json" ]]; then
  echo "ERROR: missing sample $SAMPLE/answers.json" >&2
  exit 1
fi

if [[ ! -f "$PUMA_DIR/answers.json" ]]; then
  SAMPLE_TMP="$PUMA_DIR/.answers.json.tmp.$$"
  cp -f "$SAMPLE/answers.json" "$SAMPLE_TMP"
  mv -f "$SAMPLE_TMP" "$PUMA_DIR/answers.json"
fi

# Copy this model+dataset's official seed-42 knobs; only override SEED.
conf_usable() {
  [[ -f "$1" ]] && grep -q '^SIMILARITY_THRESHOLD=' "$1"
}

pick_official_conf() {
  local roots=()
  if [[ "$MODEL_TAG" == r1_7b && "$DATASET" == math-500 ]]; then
    roots+=("$AE/results/baselines/official/math500_official/puma_ds7b")
  fi
  roots+=("$AE/results/baselines/puma/puma_offline_${MODEL_TAG}/${DATASET}")
  roots+=("$AE/results/baselines/puma/puma_offline_${MODEL_TAG}/amc23")
  roots+=("$AE/results/baselines/puma/puma_offline_${MODEL_TAG}/gpqa-diamond")
  local root name hit
  for root in "${roots[@]}"; do
    for name in _DS-7B.local.conf _DS-14B.local.conf _Nemotron.local.conf _Q30B-T.local.conf _local.conf; do
      if conf_usable "$root/$name"; then
        echo "$root/$name"
        return
      fi
    done
    hit="$(ls "$root"/_*.conf 2>/dev/null | head -n 1 || true)"
    if [[ -n "$hit" ]] && conf_usable "$hit"; then
      echo "$hit"
      return
    fi
  done
  echo "$AE/results/baselines/puma/puma_offline_r1_7b/gpqa-diamond/_DS-7B.local.conf"
}
CONF_SRC="$(pick_official_conf)"
LOCAL="$PUMA_DIR/_local.conf"
if [[ "$CONF_SRC" == "$LOCAL" ]] || ! conf_usable "$CONF_SRC"; then
  if conf_usable "$AE/results/baselines/puma/puma_offline_r1_7b/amc23/_local.conf"; then
    CONF_SRC="$AE/results/baselines/puma/puma_offline_r1_7b/amc23/_local.conf"
  else
    CONF_SRC="$AE/results/baselines/puma/puma_offline_r1_7b/gpqa-diamond/_DS-7B.local.conf"
  fi
fi
TMP="$LOCAL.tmp.$$"
{
  cat "$CONF_SRC"
  echo "SEED=$SEED"
} > "$TMP"
mv -f "$TMP" "$LOCAL"
echo "[puma-official] conf $CONF_SRC -> $LOCAL"

export VLLM_LENS_DISABLE=1
export CUDA_VISIBLE_DEVICES="$GPU"
# The PUMA generation stages require a 38k-token KV cache.  The upstream
# default (0.78) leaves too little cache space for some 14B continuations.
# Keep this scoped to the backfill wrapper and allow an explicit override.
export VLLM_GPU_MEMORY_UTILIZATION="${PUMA_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export PYTHON="$PY"
plws_export_cuda_runtime

echo "[puma-official] GPU=$GPU $MODEL_TAG $DATASET seed=$SEED -> $PUMA_DIR"
cd "$PUMA_ROOT"
bash run_pipeline.sh "$LOCAL" "$PUMA_DIR" "$MODEL" "$DATASET" "$BENCH" \
  2>&1 | tee -a "$PUMA_DIR/run.log"
write_status succeeded
RUN_FINISHED=1
echo "[puma-official] done $PUMA_DIR"
