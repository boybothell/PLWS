#!/usr/bin/env bash
# 第一扇剩窗切点 → PUMA Step 4 重写。只生成 fire 的题，再和密探k4 prefixed 合并判分。
#
#   MODEL_TAG=r1_7b DATASET=math-500 GPU=0 bash scripts/run_leftover_regen.sh
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

OUT="${OUT:-$AE/results/leftover_regen/$MODEL_TAG/$DATASET/s$SEED}"
CANDS="$OUT/final_candidates.json"
PREFIXED_FIRE="$OUT/prefixed_answers_fire.json"
PREFIXED="$OUT/prefixed_answers.json"
STATS_TXT="$OUT/statistics.txt"
ANSWERS="$PUMA_DIR/answers.json"
QUESTIONS="$PUMA_DIR/filtered_steps.json"
if [[ "$DATASET" == aime24 || "$DATASET" == aime25 ]]; then
  TRIAL="${TRIAL:-$AE/results/dense_G_${MODEL_TAG}/$DATASET/seed_${SEED}/dense_puma/trial_answers.json}"
else
  TRIAL="${TRIAL:-$AE/results/dense_G_${MODEL_TAG}/$DATASET/dense_puma/trial_answers.json}"
fi

[[ -f "$CANDS" ]] || { echo "missing $CANDS; run export_leftover_regen_cands.py"; exit 1; }
[[ -f "$ANSWERS" && -f "$QUESTIONS" && -f "$TRIAL" ]] || { echo "missing puma/trial files"; exit 1; }

N_FIRE="$("$PY" -c "import json; print(len(json.load(open('$CANDS'))))")"
if [[ "$N_FIRE" == 0 ]]; then
  echo "[leftover-regen] no fire $MODEL_TAG $DATASET s$SEED, merge only"
  "$PY" "$AE/scripts/merge_leftover_regen.py" --model "$MODEL_TAG" --dataset "$DATASET" --seed "$SEED"
  exit 0
fi

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

echo "[leftover-regen] start $(date -Is) GPU=$GPU TP=$TP $MODEL_TAG $DATASET seed=$SEED fire=$N_FIRE" | tee "$OUT/run.log"

if [[ ! -f "$PREFIXED_FIRE" ]]; then
  echo ">>> prefixed leftover cuts" | tee -a "$OUT/run.log"
  "$PY" puma/gen_prefixed_answers.py \
    --final-candidates "$CANDS" \
    --questions-file "$QUESTIONS" \
    --output-file "$PREFIXED_FIRE" \
    --model "$MODEL" \
    --max-tokens "${MAX_ANSWER_TOKENS:-4096}" \
    --tensor-parallel-size "$TP" \
    --dataset "$DATASET" \
    --prompt-version "${PROMPT_VERSION:-default}" \
    --confidence-aggregation "${CONFIDENCE_AGGREGATION:-geometric}" \
    --seed "${SEED}" \
    2>&1 | tee -a "$OUT/run.log"
else
  echo ">>> prefixed_fire exists, skip gen" | tee -a "$OUT/run.log"
fi

"$PY" "$AE/scripts/merge_leftover_regen.py" --model "$MODEL_TAG" --dataset "$DATASET" --seed "$SEED"

echo ">>> statistics" | tee -a "$OUT/run.log"
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

echo "[leftover-regen] done $(date -Is) $OUT" | tee -a "$OUT/run.log"
