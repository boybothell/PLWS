#!/usr/bin/env bash
# Fill missing Qwen Full-CoT / PUMA / dense / window jobs for one cell.
# Never regenerates 7B / Nemotron / 14B Full-CoT.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
MODEL_TAG="${MODEL_TAG:?qwen3_4b|qwen3_8b|qwen3_30b_a3b}"
DATASET="${DATASET:?set dataset}"
SEED="${SEED:?set seed}"
GPU="${GPU:?set one GPU or a TP lane}"

case "$MODEL_TAG" in
  qwen3_4b)
    MODEL=/mnt/d/lsj/models/Qwen3-4B
    ALIGN_CONF=DS-7B.conf
    EXPECTED_TP=1
    ;;
  qwen3_8b)
    MODEL=/mnt/d/lsj/models/Qwen3-8B
    ALIGN_CONF=DS-7B.conf
    EXPECTED_TP=1
    ;;
  qwen3_30b_a3b)
    MODEL=/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507
    ALIGN_CONF=Q30B-T.conf
    EXPECTED_TP=2
    ;;
  r1_7b|nemotron_8b|r1_14b)
    echo "ERROR: refused to regenerate Full-CoT for $MODEL_TAG" >&2
    exit 2
    ;;
  *)
    echo "ERROR: unsupported MODEL_TAG=$MODEL_TAG" >&2
    exit 2
    ;;
esac

TP="$(awk -F',' '{print NF}' <<<"$GPU")"
if [[ "$TP" -ne "$EXPECTED_TP" ]]; then
  echo "ERROR: $MODEL_TAG requires TP=$EXPECTED_TP, got GPU=$GPU" >&2
  exit 1
fi

export PLWS_ROOT="$ROOT"
export VLLM_LENS_DISABLE=1
export CUDA_VISIBLE_DEVICES="$GPU"
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

if [[ "$SEED" == 42 ]]; then
  PUMA_DIR="$ROOT/results/baselines/puma/puma_offline_${MODEL_TAG}/$DATASET"
else
  PUMA_DIR="$ROOT/results/baselines/puma/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET"
fi
SAMPLE="$ROOT/samples/$MODEL_TAG/$DATASET/seed_${SEED}"

jobs_ready() {
  "$PY" - "$ROOT" "$MODEL_TAG" "$DATASET" "$SEED" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.matrix import jobs_present
from plws.paths import PLWSPaths
raise SystemExit(0 if jobs_present(PLWSPaths(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])) else 1)
PY
}

sample_is_v2() {
  [[ -f "$SAMPLE/answers.json" && -f "$SAMPLE/sample_meta.json" ]] || return 1
  "$PY" - "$SAMPLE/sample_meta.json" <<'PY'
import json
import sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text())
valid = (
    meta.get("protocol_id") == "puma-fullcot-32k-v2"
    and meta.get("max_tokens") == 32768
    and meta.get("answer_fix_max_tokens") == 2048
    and meta.get("prompt_reserve_tokens") == 3072
    and meta.get("max_model_len") == 37888
    and meta.get("prompt_version") == "default"
)
raise SystemExit(0 if valid else 1)
PY
}

stash() {
  local src="$1" dest="$2"
  [[ -e "$src" ]] || return 0
  mkdir -p "$(dirname "$dest")"
  mv "$src" "$dest"
  echo "[matrix-prereq] stashed $src -> $dest"
}

puma_ready() {
  [[ -f "$PUMA_DIR/statistics.json" && -f "$PUMA_DIR/prefixed_answers.json" ]]
}

if puma_ready && jobs_ready; then
  echo "[matrix-prereq] skip complete $MODEL_TAG $DATASET seed=$SEED"
  exit 0
fi

echo "[matrix-prereq] start $(date -Is) $MODEL_TAG $DATASET seed=$SEED gpu=$GPU"

if puma_ready; then
  echo "[matrix-prereq] reuse complete PUMA $PUMA_DIR"
elif [[ -f "$PUMA_DIR/answers.json" ]]; then
  echo "[matrix-prereq] finish incomplete PUMA from existing answers $PUMA_DIR"
  PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
    ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    PUMA_DIR="$PUMA_DIR" \
    bash "$ROOT/scripts/run_puma_official.sh"
else
  if [[ -f "$SAMPLE/answers.json" ]] && ! sample_is_v2; then
    stamp="$(date +%Y%m%dT%H%M%S)"
    stash "$SAMPLE" "$ROOT/samples/_stale_non_v2/$MODEL_TAG/$DATASET/seed_${SEED}_$stamp"
  fi
  PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
    ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    bash "$ROOT/scripts/run_puma_aligned_sample.sh"
  PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
    ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    PUMA_DIR="$PUMA_DIR" \
    bash "$ROOT/scripts/run_puma_official.sh"
fi

DENSE_GPU_ONLY=1 PLWS_ROOT="$ROOT" MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" \
  SEED="$SEED" GPUS="$GPU" TP="$TP" SEED_LAYOUT=1 PUMA_DIR="$PUMA_DIR" \
  bash "$ROOT/scripts/run_dense_trials_model.sh"

EXPORT_LOG="$ROOT/results/runs/matrix_queue/logs/export__${MODEL_TAG}__${DATASET}__s${SEED}.log"
mkdir -p "$(dirname "$EXPORT_LOG")"
echo "[matrix-prereq] detach CPU export $(date -Is) log=$EXPORT_LOG"
# Release the GPU lease; jobs are written off-GPU. PLWS waits on jobs_present.
nohup env \
  PLWS_ROOT="$ROOT" \
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$ROOT/scripts/export_leftover_suppress_jobs.py" \
  --model-tag "$MODEL_TAG" --seed "$SEED" --datasets "$DATASET" \
  --kinds firstwin --k 4 --lexicon core \
  >>"$EXPORT_LOG" 2>&1 &
disown || true

echo "[matrix-prereq] gpu released $(date -Is) $MODEL_TAG $DATASET seed=$SEED"
