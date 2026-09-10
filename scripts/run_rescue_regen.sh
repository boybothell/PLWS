#!/usr/bin/env bash
# Rescue-R / 置信度门 → PUMA Step 4.
#
#   DATASET=math-500 POLICY=lag GPU=2 bash scripts/run_rescue_regen.sh
#   DATASET=gpqa-diamond POLICY=conf GPU=3 bash scripts/run_rescue_regen.sh
set -euo pipefail

AE=/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit
PUMA_ROOT=/mnt/d/lsj/visual-latent-tts/repos/PUMA
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python

DATASET="${DATASET:?}"
POLICY="${POLICY:?}"
GPU="${GPU:?}"
SEED="${SEED:-42}"
TP="${TP:-1}"
MODEL_TAG="${MODEL_TAG:-r1_7b}"
MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B
CONF=DS-7B.conf

if [[ "$DATASET" == math-500 ]]; then
  PUMA_DIR="${PUMA_DIR:-$AE/results/math500_official/puma_ds7b}"
else
  PUMA_DIR="${PUMA_DIR:-$AE/results/puma_offline_${MODEL_TAG}/$DATASET}"
fi

OUT="${OUT:-$AE/results/rescue_R_gate/$DATASET/$POLICY}"
CANDS="$OUT/final_candidates.json"
PREFIXED="$OUT/prefixed_answers.json"
STATS_TXT="$OUT/statistics.txt"
ANSWERS="$PUMA_DIR/answers.json"
QUESTIONS="$PUMA_DIR/filtered_steps.json"
TRIAL="$AE/results/dense_G_${MODEL_TAG}/$DATASET/dense_puma/trial_answers.json"

[[ -f "$CANDS" ]] || { echo "missing $CANDS; run scripts/replay_rescue_R_gate.py first"; exit 1; }
[[ -f "$ANSWERS" && -f "$QUESTIONS" && -f "$TRIAL" ]] || { echo "missing puma/trial files"; exit 1; }

export VLLM_LENS_DISABLE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="$(
python3 - <<'PY'
from pathlib import Path
root = Path('/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm')
print(':'.join(sorted({str(p) for p in (root / '.venv' / 'lib').glob('**/nvidia/*/lib') if p.is_dir()})))
PY
):${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
cd "$PUMA_ROOT"
if [[ -f "$PUMA_DIR/_local.conf" ]]; then
  # shellcheck source=/dev/null
  source "$PUMA_DIR/_local.conf"
elif [[ -f "$PUMA_DIR/_${CONF%.conf}.local.conf" ]]; then
  # shellcheck source=/dev/null
  source "$PUMA_DIR/_${CONF%.conf}.local.conf"
else
  # shellcheck source=/dev/null
  source "configs/$CONF"
fi

echo "[rescue-regen] start $(date -Is) GPU=$GPU $DATASET $POLICY" | tee "$OUT/run.log"
echo "[rescue-regen] cands=$CANDS puma_dir=$PUMA_DIR" | tee -a "$OUT/run.log"

if [[ ! -f "$PREFIXED" ]]; then
  echo ">>> prefixed answers (PUMA Step 4)" | tee -a "$OUT/run.log"
  "$PY" puma/gen_prefixed_answers.py \
    --final-candidates "$CANDS" \
    --questions-file "$QUESTIONS" \
    --output-file "$PREFIXED" \
    --model "$MODEL" \
    --max-tokens "${MAX_ANSWER_TOKENS:-4096}" \
    --tensor-parallel-size "$TP" \
    --dataset "$DATASET" \
    --prompt-version "${PROMPT_VERSION:-default}" \
    --confidence-aggregation "${CONFIDENCE_AGGREGATION:-geometric}" \
    --seed "${SEED}" \
    2>&1 | tee -a "$OUT/run.log"
else
  echo ">>> prefixed exists, skip" | tee -a "$OUT/run.log"
fi

echo ">>> statistics (grade regenerated answers)" | tee -a "$OUT/run.log"
"$PY" puma/statistics_puma.py \
  --original "$ANSWERS" \
  --compressed "$PREFIXED" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output "$STATS_TXT" \
  --workers 0 \
  --trial-answers "$TRIAL" \
  --questions-with-steps "$QUESTIONS" \
  2>&1 | tee -a "$OUT/run.log"

echo "[rescue-regen] done $(date -Is) $OUT" | tee -a "$OUT/run.log"
