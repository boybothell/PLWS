#!/usr/bin/env bash
# Canonical rental-server entry point for one model/dataset/seed cell.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_TAG="${MODEL_TAG:?qwen3_30b_a3b|r1_32b|qwen3_32b|qwq_32b}"
DATASET="${DATASET:?math-500|olympiadbench|gpqa-diamond|aime24|aime25|aime26|brumo25|hmmt25|amc23}"
SEED="${SEED:?42|0|1|123|7}"
GPU="${GPU:?one GPU or a comma-separated TP lane}"
STAGES="${STAGES:-puma,plws,deer}"

case "$MODEL_TAG" in
  qwen3_30b_a3b|r1_32b|qwen3_32b|qwq_32b) ;;
  *)
    echo "ERROR: unsupported large MODEL_TAG=$MODEL_TAG" >&2
    exit 2
    ;;
esac

case "$DATASET" in
  math-500|olympiadbench|gpqa-diamond|aime24|aime25|aime26|brumo25|hmmt25|amc23) ;;
  *)
    echo "ERROR: unsupported DATASET=$DATASET" >&2
    exit 2
    ;;
esac

case "$SEED" in
  42|0|1|123|7) ;;
  *)
    echo "ERROR: unsupported SEED=$SEED" >&2
    exit 2
    ;;
esac

export PLWS_ROOT="$ROOT" MODEL_TAG DATASET SEED GPU
export PLWS_TP="${PLWS_TP:-$(awk -F',' '{print NF}' <<<"$GPU")}"

contains_stage() {
  [[ ",$STAGES," == *",$1,"* ]]
}

for stage in ${STAGES//,/ }; do
  case "$stage" in
    puma|plws|deer) ;;
    *)
      echo "ERROR: unknown stage '$stage' in STAGES=$STAGES" >&2
      exit 2
      ;;
  esac
done

# PLWS requires the PUMA/dense/jobs prerequisite. Each underlying runner is
# idempotent and rescans canonical artifacts before doing expensive work.
if contains_stage puma || contains_stage plws; then
  bash "$ROOT/scripts/run_contest_prereq_cell.sh"
fi

if contains_stage plws; then
  bash "$ROOT/scripts/run_contest_plws_cell.sh"
fi

if contains_stage deer; then
  bash "$ROOT/scripts/run_deer_official.sh"
fi
