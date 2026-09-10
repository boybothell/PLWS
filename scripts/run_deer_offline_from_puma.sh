#!/usr/bin/env bash
# Offline DEER Acc/Tok/CR on the same Full-CoT as PUMA, with Wait ATP aligned to online.
#
# Online DEER probes only when generation hits stop-token "Wait", ≤ max_judge_steps=10.
# This offline path freezes Full-CoT and probes at the first 10 \bWait\b ends only
# (NOT every PUMA separate_steps paragraph).
#
# Usage:
#   GPU=0 DATASET=aime24 bash scripts/run_deer_offline_from_puma.sh
#   GPU=1 DATASET=aime25 bash scripts/run_deer_offline_from_puma.sh
set -euo pipefail

PUMA_ROOT=/mnt/d/lsj/visual-latent-tts/repos/PUMA
AE_ROOT=/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit
MODEL="${MODEL:-/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B}"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
GPU="${GPU:-0}"
DATASET="${DATASET:?set DATASET=aime24|aime25|math-500|gpqa-diamond|olympiadbench}"
MAX_JUDGE="${MAX_JUDGE_STEPS:-10}"
DEER_THRESHOLD="${DEER_THRESHOLD:-0.95}"

PUMA_DIR="${PUMA_DIR:-$AE_ROOT/results/puma_offline_r1_7b/${DATASET}}"
if [[ "$DATASET" == "math-500" && ! -f "$PUMA_DIR/answers.json" ]]; then
  PUMA_DIR="$AE_ROOT/results/math500_official/puma_ds7b"
fi

# Wait-ATP offline (do not reuse old every-step deer_offline_r1_7b trials)
OUT="${OUT:-$AE_ROOT/results/deer_offline_wait_r1_7b/${DATASET}}"
ANSWERS=$PUMA_DIR/answers.json
WAIT_Q=$OUT/wait_probe_questions.json
TRIAL=$OUT/trial_answers.json
CANDS=$OUT/final_candidates.json
PREFIXED=$OUT/prefixed_answers.json
STATS=$OUT/statistics.txt
COMPARE=$OUT/compare_puma_vs_deer_offline.txt

[[ -f "$ANSWERS" ]] || { echo "missing $ANSWERS"; exit 1; }

export VLLM_LENS_DISABLE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="$(
python3 - <<'PY'
from pathlib import Path
root = Path('/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm')
print(':'.join(sorted({str(p) for p in (root/'.venv'/'lib').glob('**/nvidia/*/lib') if p.is_dir()})))
PY
):${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
cd "$PUMA_ROOT"
# shellcheck source=/dev/null
if [[ -f "$PUMA_DIR/_DS-7B.local.conf" ]]; then
  source "$PUMA_DIR/_DS-7B.local.conf"
else
  source configs/DS-7B.conf
fi

echo "[deer-off-wait] start $(date -Is) GPU=$GPU dataset=$DATASET max_judge=$MAX_JUDGE" | tee "$OUT/run.log"
echo "[deer-off-wait] PUMA_DIR=$PUMA_DIR OUT=$OUT" | tee -a "$OUT/run.log"

echo ">>> build Wait-ATP probe questions (≤$MAX_JUDGE Wait ends)" | tee -a "$OUT/run.log"
"$PY" "$AE_ROOT/scripts/build_deer_wait_probe_questions.py" \
  --answers "$ANSWERS" \
  --output "$WAIT_Q" \
  --max-judge-steps "$MAX_JUDGE" \
  --dataset "$DATASET" \
  2>&1 | tee -a "$OUT/run.log"

if [[ ! -f "$TRIAL" ]]; then
  echo ">>> DEER offline trials (Wait ATP only, --respect-embedding-filter)" | tee -a "$OUT/run.log"
  "$PY" puma/gen_trial_answers.py \
    --questions-file "$WAIT_Q" \
    --output-file "$TRIAL" \
    --model "$MODEL" \
    --max-tokens "${MAX_TRIAL_TOKENS:-30}" \
    --temperature "${TEMPERATURE:-0.6}" \
    --top_p "${TOP_P:-0.95}" \
    --tensor-parallel-size 1 \
    --dataset "$DATASET" \
    --trial-decoding "${TRIAL_DECODING:-sampling}" \
    --confidence-mode "${CONFIDENCE_MODE:-token_in_boxed}" \
    --confidence-aggregation "${CONFIDENCE_AGGREGATION:-geometric}" \
    --prompt-version "${PROMPT_VERSION:-default}" \
    --seed "${SEED:-42}" \
    --respect-embedding-filter \
    2>&1 | tee -a "$OUT/run.log"
else
  echo ">>> trials exist, skip" | tee -a "$OUT/run.log"
fi

echo ">>> extract DEER candidates λ=$DEER_THRESHOLD (conf > λ, Wait probes)" | tee -a "$OUT/run.log"
"$PY" "$AE_ROOT/scripts/extract_deer_offline_candidates.py" \
  --trial-answers "$TRIAL" \
  --output "$CANDS" \
  --threshold "$DEER_THRESHOLD" \
  2>&1 | tee -a "$OUT/run.log"

if [[ ! -f "$PREFIXED" ]]; then
  echo ">>> prefixed answers (same as PUMA Step4)" | tee -a "$OUT/run.log"
  "$PY" puma/gen_prefixed_answers.py \
    --final-candidates "$CANDS" \
    --questions-file "$WAIT_Q" \
    --output-file "$PREFIXED" \
    --model "$MODEL" \
    --max-tokens "${MAX_ANSWER_TOKENS:-4096}" \
    --tensor-parallel-size 1 \
    --dataset "$DATASET" \
    --prompt-version "${PROMPT_VERSION:-default}" \
    --confidence-aggregation "${CONFIDENCE_AGGREGATION:-geometric}" \
    --seed "${SEED:-42}" \
    2>&1 | tee -a "$OUT/run.log"
else
  echo ">>> prefixed exists, skip" | tee -a "$OUT/run.log"
fi

echo ">>> statistics (same grader/token path as PUMA)" | tee -a "$OUT/run.log"
"$PY" puma/statistics_puma.py \
  --original "$ANSWERS" \
  --compressed "$PREFIXED" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --confidence-threshold "$DEER_THRESHOLD" \
  --epsilon 0.0 \
  --consecutive 1 \
  --confidence-mode "${CONFIDENCE_MODE:-token_in_boxed}" \
  --confidence-aggregation "${CONFIDENCE_AGGREGATION:-geometric}" \
  --min-stop-step 0 \
  --trial-decoding "${TRIAL_DECODING:-sampling}" \
  --output "$STATS" \
  --workers 0 \
  --trial-answers "$TRIAL" \
  --questions-with-steps "$WAIT_Q" \
  2>&1 | tee -a "$OUT/run.log"

export PUMA_STATS="$PUMA_DIR/statistics.txt"
export DEER_STATS="$STATS"
export DATASET_NAME="$DATASET"
"$PY" - <<'PY' | tee "$COMPARE"
import os, re
from pathlib import Path

def parse_stats2(path: Path):
    lines = path.read_text().splitlines() if path.exists() else []
    out = {}
    mode = None
    for ln in lines:
        if "Average tokens (original):" in ln:
            out["tok_vanilla"] = float(ln.split(":")[-1].strip())
        elif "Average tokens (compressed):" in ln and "trial" not in ln:
            out["tok_method"] = float(ln.split(":")[-1].strip())
        elif "Average trial answer tokens:" in ln:
            out["tok_trial"] = float(ln.split(":")[-1].strip())
        elif "Compression (excluding trial answers):" in ln:
            mode = "ex"
        elif "Compression (including trial answers):" in ln:
            mode = "in"
        elif "Online Simulation" in ln:
            mode = None
        elif "Compression rate:" in ln and mode == "ex":
            out["cr"] = float(ln.split(":")[-1].strip().rstrip("%"))
        elif "Reduction:" in ln and mode == "ex":
            out["tr"] = float(ln.split(":")[-1].strip().rstrip("%"))
        elif "Compression rate:" in ln and mode == "in":
            out["crt"] = float(ln.split(":")[-1].strip().rstrip("%"))
        elif "Reduction:" in ln and mode == "in":
            out["trt"] = float(ln.split(":")[-1].strip().rstrip("%"))
    for i, ln in enumerate(lines):
        if ln.strip().endswith("Accuracy:"):
            for j in range(i + 1, min(i + 6, len(lines))):
                if "Original:" in lines[j]:
                    out["acc_van"] = float(re.search(r"([\d.]+)%", lines[j]).group(1))
                if "Compressed:" in lines[j]:
                    out["acc"] = float(re.search(r"([\d.]+)%", lines[j]).group(1))
            break
    return out

ds = os.environ["DATASET_NAME"]
puma = parse_stats2(Path(os.environ["PUMA_STATS"]))
deer = parse_stats2(Path(os.environ["DEER_STATS"]))
nan = float("nan")

print("=" * 72)
print(f"{ds} / DS-7B — offline Wait-ATP Acc/Tok on shared Full-CoT")
print("=" * 72)
print("""
Probe: first ≤10 \bWait\b ends on frozen Full-CoT (aligned to online ATP schedule).
Decision: first trial conf > 0.95 (same as vllm_deer). Prefix regen + statistics_puma.
""")
print(f"{'Method':<10} {'Acc':>8} {'Tok':>10} {'TR_ex':>8} {'TR_in':>8} {'CRT':>8}")
print("-" * 60)
print(f"{'Vanilla':<10} {puma.get('acc_van', nan):7.2f}% {puma.get('tok_vanilla', nan):10.1f} {'0.00%':>8} {'0.00%':>8} {'—':>8}")
print(f"{'PUMA':<10} {puma.get('acc', nan):7.2f}% {puma.get('tok_method', nan):10.1f} {puma.get('tr', nan):7.2f}% {puma.get('trt', nan):7.2f}% {puma.get('crt', nan):7.2f}%")
print(f"{'DEER-Wait':<10} {deer.get('acc', nan):7.2f}% {deer.get('tok_method', nan):10.1f} {deer.get('tr', nan):7.2f}% {deer.get('trt', nan):7.2f}% {deer.get('crt', nan):7.2f}%")
print("=" * 72)
print("PUMA:", os.environ["PUMA_STATS"])
print("DEER:", os.environ["DEER_STATS"])
PY

echo "[deer-off-wait] ALL done $(date -Is)" | tee -a "$OUT/run.log"
echo "compare → $COMPARE"
