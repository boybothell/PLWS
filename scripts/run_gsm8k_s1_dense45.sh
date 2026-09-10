#!/usr/bin/env bash
set +e
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
export LD_LIBRARY_PATH="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
echo "[d45] gsm8k s1 dense GPUS=4,5 $(date -Is)"
DENSE_GPU_ONLY=1 SEED_LAYOUT=1 MODEL_TAG=r1_7b DATASET=gsm8k SEED=1 GPUS=4,5 TP=1 \
  bash scripts/run_dense_trials_model.sh
echo "[d45] dense exit=$? $(date -Is)"
if [[ -s results/dense_G_r1_7b/gsm8k/seed_1/dense_puma/trial_answers.json && ! -s results/dense_G_r1_7b/gsm8k/seed_1/per_sample.json ]]; then
  nohup "$AE/.venv/bin/python" "$AE/scripts/compute_dense_G.py" \
    --dataset gsm8k \
    --dense-root "$AE/results/dense_G_r1_7b/gsm8k/seed_1/dense_puma" \
    --out "$AE/results/dense_G_r1_7b/gsm8k/seed_1/per_sample.json" \
    --workers 8 >> samples/_logs/g_gsm8k_s1.log 2>&1 &
  echo "[d45] G background pid=$!"
fi
echo "[d45] done $(date -Is)"
