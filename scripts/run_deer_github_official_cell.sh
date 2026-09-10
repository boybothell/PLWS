#!/usr/bin/env bash
# Verified single-cell wrapper for the unmodified iie-ycx/DEER implementation.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:?set MODEL}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET}"
GPU="${GPU:?set GPU}"
FAMILY="${FAMILY:?set FAMILY=standard|qwen3}"
OUT="${OUT:?set canonical cell output directory}"
DEER_DATA=/mnt/d/lsj/visual-latent-tts/repos/DEER/data

count_jsonl() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip()))
PY
}

expected="$(count_jsonl "$DEER_DATA/$DATASET/test.jsonl")"

complete_output() {
  [[ -f "$OUT/manifest.json" ]] || return 1
  local file count
  file="$(rg --files "$OUT" -g '*.jsonl' | sort | tail -n 1)"
  [[ -n "$file" && -f "$file" ]] || return 1
  count="$(count_jsonl "$file")"
  [[ "$count" -eq "$expected" ]]
}

if complete_output; then
  echo "[deer-cell] skip verified complete $MODEL_TAG $DATASET n=$expected"
  exit 0
fi

echo "[deer-cell] start $(date -Is) $MODEL_TAG $DATASET gpu=$GPU family=$FAMILY"
PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" \
  GPU="$GPU" FAMILY="$FAMILY" OUT="$OUT" \
  bash "$ROOT/scripts/run_deer_github_official.sh"

if ! complete_output; then
  echo "ERROR: DEER output failed completeness check: $MODEL_TAG $DATASET expected=$expected" >&2
  exit 1
fi
echo "[deer-cell] done $(date -Is) $MODEL_TAG $DATASET n=$expected"

