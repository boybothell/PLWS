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
ALIGN_CONF="${ALIGN_CONF:-$(plws_align_conf "$MODEL_TAG")}"

if [[ "$SEED" != "42" ]]; then
  PUMA_DIR="${PUMA_DIR:-$AE/results/baselines/puma/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET}"
else
  PUMA_DIR="${PUMA_DIR:-$AE/results/baselines/puma/puma_offline_${MODEL_TAG}/$DATASET}"
fi
SAMPLE="${SAMPLE:-$AE/samples/$MODEL_TAG/$DATASET/seed_${SEED}}"
BENCH="${BENCH:-$PLWS_DATA_ROOT/${DATASET}_test.jsonl}"

verify_statistics() {
  "$PY" -m plws.puma_grading \
    --statistics "$PUMA_DIR/statistics.json" \
    --fix
}

# A queue restart can adopt a Full-CoT sampler that did not start with
# FULLCOT_ONLY.  A one-shot marker lets that sampler stop at this phase
# boundary without interrupting generation; the queue then re-launches the
# cell under its current scheduling policy.  Markers live in the machine-local
# CONTEST_RUN_ROOT and are consumed exactly once.
if [[ -n "${CONTEST_RUN_ROOT:-}" ]]; then
  FULLCOT_BARRIER="$CONTEST_RUN_ROOT/fullcot_barrier/${MODEL_TAG}__${DATASET}__s${SEED}"
  if [[ -f "$FULLCOT_BARRIER" ]]; then
    rm -f "$FULLCOT_BARRIER"
    echo "[puma-official] fullcot barrier released $MODEL_TAG $DATASET seed=$SEED"
    exit 75
  fi
fi

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
  verify_statistics
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

# Prefer a prior same-model _local.conf; otherwise use official ALIGN_CONF.
# Never cat a missing r1_7b leftover — that path is local-only and not shipped.
LOCAL="$PUMA_DIR/_local.conf"
CONF_SRC="$("$PY" -m plws.puma_official_conf pick \
  --ae "$AE" --puma-root "$PUMA_ROOT" \
  --model-tag "$MODEL_TAG" --dataset "$DATASET" --align-conf "$ALIGN_CONF")"
"$PY" -m plws.puma_official_conf write \
  --src "$CONF_SRC" --dst "$LOCAL" --seed "$SEED"
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
verify_statistics
write_status succeeded
RUN_FINISHED=1
echo "[puma-official] done $PUMA_DIR"
