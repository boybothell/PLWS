#!/usr/bin/env bash
# HISTORICAL DIAGNOSTIC — NOT PLWS.
# DEFAULT dense gate → PUMA Step 4: regenerate the final answer from the
# truncated prefix. Canonical PLWS instead continues generation while
# suppressing Wait-class tokens via score_leftover_suppress.py.
#
#   MODEL_TAG=r1_7b DATASET=math-500 GPU=2 bash scripts/run_default_regen.sh
#   MODEL_TAG=r1_14b DATASET=math-500 GPU=3,4 TP=2 bash scripts/run_default_regen.sh
set -euo pipefail

AE=/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit
PUMA_ROOT=/mnt/d/lsj/visual-latent-tts/repos/PUMA
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python

MODEL_TAG="${MODEL_TAG:?}"
DATASET="${DATASET:?}"
SEED="${SEED:-42}"
GPU="${GPU:?}"
TP="${TP:-1}"

case "$MODEL_TAG" in
  r1_7b)
    MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B
    CONF=DS-7B.conf
    ;;
  nemotron_8b)
    MODEL=/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1
    CONF=Nemotron.conf
    ;;
  r1_14b)
    MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B
    CONF=DS-14B.conf
    TP="${TP:-2}"
    ;;
  r1_32b)
    MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-32B
    CONF=DS-32B.conf
    TP="${TP:-2}"
    ;;
  qwen3_30b_a3b)
    MODEL=/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507
    CONF=Q30B-T.conf
    TP="${TP:-2}"
    ;;
  qwen3_4b)
    MODEL=/mnt/d/lsj/models/Qwen3-4B
    CONF=Q30B-T.conf
    ;;
  qwen3_8b)
    MODEL=/mnt/d/lsj/models/Qwen3-8B
    CONF=Q30B-T.conf
    ;;
  *)
    echo "unknown MODEL_TAG=$MODEL_TAG" >&2
    exit 1
    ;;
esac

if [[ "$MODEL_TAG" == r1_7b && "$DATASET" == math-500 && "$SEED" == 42 ]]; then
  PUMA_DIR="${PUMA_DIR:-$AE/results/math500_official/puma_ds7b}"
elif [[ "$SEED" != 42 ]]; then
  PUMA_DIR="${PUMA_DIR:-$AE/results/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET}"
else
  PUMA_DIR="${PUMA_DIR:-$AE/results/puma_offline_${MODEL_TAG}/$DATASET}"
fi

OUT="${OUT:-$AE/results/default_dense_gate/$MODEL_TAG/$DATASET/s$SEED}"
CANDS="$OUT/final_candidates.json"
PREFIXED="$OUT/prefixed_answers.json"
STATS_TXT="$OUT/statistics.txt"
ANSWERS="$PUMA_DIR/answers.json"
QUESTIONS="$PUMA_DIR/filtered_steps.json"
SEEDED_TRIAL="$AE/results/dense_G_${MODEL_TAG}/$DATASET/seed_${SEED}/dense_puma/trial_answers.json"
FLAT_TRIAL="$AE/results/dense_G_${MODEL_TAG}/$DATASET/dense_puma/trial_answers.json"
if [[ -z "${TRIAL:-}" ]]; then
  if [[ -f "$SEEDED_TRIAL" ]]; then
    TRIAL="$SEEDED_TRIAL"
  elif [[ -f "$FLAT_TRIAL" ]]; then
    TRIAL="$FLAT_TRIAL"
  elif [[ "$DATASET" == aime24 || "$DATASET" == aime25 || "$SEED" != 42 ]]; then
    TRIAL="$SEEDED_TRIAL"
  else
    TRIAL="$FLAT_TRIAL"
  fi
fi
if [[ ! -f "$TRIAL" ]]; then
  if [[ "$SEED" != 42 ]]; then
    TRIAL="$AE/results/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET/trial_answers.json"
  else
    TRIAL="$AE/results/puma_offline_${MODEL_TAG}/$DATASET/trial_answers.json"
  fi
fi

[[ -f "$CANDS" ]] || { echo "missing $CANDS; run scripts/replay_default_dense_gate.py first"; exit 1; }
[[ -f "$ANSWERS" ]] || { echo "missing $ANSWERS"; exit 1; }
[[ -f "$QUESTIONS" ]] || { echo "missing $QUESTIONS"; exit 1; }

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

echo "[default-regen] start $(date -Is) GPU=$GPU TP=$TP $MODEL_TAG $DATASET seed=$SEED" | tee "$OUT/run.log"
echo "[default-regen] cands=$CANDS puma_dir=$PUMA_DIR" | tee -a "$OUT/run.log"

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

echo "[default-regen] done $(date -Is) $OUT" | tee -a "$OUT/run.log"
