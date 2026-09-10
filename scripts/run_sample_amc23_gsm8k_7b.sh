#!/usr/bin/env bash
# 7B Full-CoT 采样 AMC23 + GSM8K，四种子，两卡 4/5。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
SCRIPT="$AE/scripts/run_puma_aligned_sample.sh"
LOG="$AE/samples/_logs"
mkdir -p "$LOG"
SEEDS=(42 0 1 123)
DATASETS=(amc23 gsm8k)
GPUS=(4 5)
MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B
TAG=r1_7b

need=()
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ -f "$AE/samples/$TAG/$ds/seed_${seed}/answers.json" ]]; then
      echo "HAVE $ds s${seed}"
    else
      need+=("$ds|$seed")
      echo "NEED $ds s${seed}"
    fi
  done
done
echo "missing ${#need[@]}"
if ((${#need[@]} == 0)); then
  echo "all samples exist"
  exit 0
fi

run_gpu() {
  local gpu="$1"
  shift
  local jobs=("$@")
  local log="$LOG/sample_amc_gsm_gpu${gpu}.log"
  {
    echo "[gpu${gpu}] start $(date -Is) n=${#jobs[@]}"
    for spec in "${jobs[@]}"; do
      IFS='|' read -r ds seed <<<"$spec"
      echo "[gpu${gpu}] === $ds seed_${seed} ==="
      MODEL="$MODEL" MODEL_TAG="$TAG" ALIGN_CONF=DS-7B.conf \
        GPU="$gpu" DATASET="$ds" SEED="$seed" bash "$SCRIPT"
    done
    echo "[gpu${gpu}] ALL done $(date -Is)"
  } > "$log" 2>&1
}

Q0=(); Q1=()
i=0
for spec in "${need[@]}"; do
  if (( i % 2 == 0 )); then Q0+=("$spec"); else Q1+=("$spec"); fi
  i=$((i + 1))
done

if ((${#Q0[@]})); then run_gpu 4 "${Q0[@]}" & fi
if ((${#Q1[@]})); then run_gpu 5 "${Q1[@]}" & fi
wait
echo "sample amc23+gsm8k 7B 4seed done"
