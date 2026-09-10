#!/usr/bin/env bash
# 7B AMC23/GSM8K：官方 PUMA → 密探 → 第一扇窗后压。4/5 各跑各的，一格失败不连坐。
set +e
set -u
AE="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PLWS_ROOT="$AE"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
RUN_ROOT="$AE/results/runs/plws/window_first/k_4/lexicon_core"
LOG="$RUN_ROOT/logs"
mkdir -p "$LOG"
MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B
TAG=r1_7b

puma_dir() {
  local ds="$1" seed="$2"
  if [[ "$seed" != "42" ]]; then
    echo "$AE/results/puma_offline_${TAG}_s${seed}/$ds"
  else
    echo "$AE/results/puma_offline_${TAG}/$ds"
  fi
}

puma_done() {
  local d
  d="$(puma_dir "$1" "$2")"
  [[ -s "$d/statistics.json" && -s "$d/prefixed_answers.json" ]]
}

dense_done() {
  [[ -s "$AE/results/dense_G_${TAG}/$1/seed_$2/dense_puma/trial_answers.json" ]]
}

puma_one() {
  local ds="$1" seed="$2" gpu="$3"
  if puma_done "$ds" "$seed"; then
    echo "[plws] skip puma $ds s${seed}"
    return 0
  fi
  echo "[plws] puma $ds s${seed} gpu${gpu} $(date -Is)"
  MODEL="$MODEL" MODEL_TAG="$TAG" DATASET="$ds" SEED="$seed" GPU="$gpu" \
    bash scripts/run_puma_official.sh
}

dense_one() {
  local ds="$1" seed="$2" gpu="$3"
  if dense_done "$ds" "$seed"; then
    echo "[plws] skip dense $ds s${seed}"
    return 0
  fi
  if ! puma_done "$ds" "$seed"; then
    echo "[plws] skip dense $ds s${seed}: puma incomplete"
    return 1
  fi
  echo "[plws] dense $ds s${seed} gpu${gpu} $(date -Is)"
  SEED_LAYOUT=1 MODEL_TAG="$TAG" DATASET="$ds" SEED="$seed" GPUS="$gpu" TP=1 \
    bash scripts/run_dense_trials_model.sh
}

run_cell() {
  local ds="$1" seed="$2" gpu="$3"
  local try
  for try in 1 2; do
    if puma_one "$ds" "$seed" "$gpu"; then
      break
    fi
    echo "[plws] puma FAIL $ds s${seed} try=${try}"
    sleep 5
  done
  if ! puma_done "$ds" "$seed"; then
    echo "[plws] puma GIVE UP $ds s${seed}"
    return 1
  fi
  for try in 1 2; do
    if dense_one "$ds" "$seed" "$gpu"; then
      return 0
    fi
    echo "[plws] dense FAIL $ds s${seed} try=${try}"
    sleep 5
  done
  echo "[plws] dense GIVE UP $ds s${seed}"
  return 1
}

gpu_loop() {
  local gpu="$1"
  shift
  echo "[plws] gpu${gpu} queue: $*"
  local spec ds seed
  for spec in "$@"; do
    ds="${spec%%:*}"; seed="${spec##*:}"
    run_cell "$ds" "$seed" "$gpu"
  done
  echo "[plws] gpu${gpu} queue done $(date -Is)"
}

echo "[plws] samples already ready, start leftover chain $(date -Is)"
gpu_loop 4 amc23:0 gsm8k:42 gsm8k:1 > "$LOG/plws_gpu4.log" 2>&1 &
pid4=$!
gpu_loop 5 gsm8k:0 gsm8k:123 > "$LOG/plws_gpu5.log" 2>&1 &
pid5=$!
wait "$pid4"
wait "$pid5"
while pgrep -f 'plws4fill|run_puma_official.sh|run_dense_trials_model.sh' >/dev/null; do
  echo "[plws] wait sidecar puma/dense $(date -Is)"
  sleep 30
done

echo "[plws] export leftover first-window jobs (restore 5集 + amc23/gsm8k)"
"$PY" scripts/export_leftover_suppress_jobs.py \
  --model-tag "$TAG" --kinds all --seeds 42,0,1,123 \
  --datasets math-500,olympiadbench,gpqa-diamond,aime24,aime25,amc23,gsm8k

suppress_kind() {
  local dataset="$1" seed="$2" kind="$3"
  local cell="$RUN_ROOT/$TAG/$dataset/seed_${seed}"
  local jobs="$cell/jobs/${kind}.jsonl"
  local old_jobs="$AE/results/leftover_jump/${TAG}_s${seed}/jobs.jsonl"
  [[ "$kind" == "low" ]] || old_jobs="$AE/results/leftover_jump/${TAG}_s${seed}/jobs_${kind}.jsonl"
  if [[ ! -f "$jobs" && -f "$old_jobs" ]]; then
    jobs="$RUN_ROOT/work/legacy_jobs/${TAG}/${dataset}/seed_${seed}/${kind}.jsonl"
    mkdir -p "$(dirname "$jobs")"
    grep -F "\"dataset\": \"$dataset\"" "$old_jobs" > "$jobs" || true
  fi
  if [[ ! -s "$jobs" ]]; then
    echo "[plws] skip empty $dataset $kind s${seed}"
    return 0
  fi
  echo "[plws] suppress $dataset s${seed} $kind"
  local i gpu outfile
  local pids=()
  for i in 0 1; do
    gpu=$((4 + i))
    outfile="$cell/scores/$kind/shard_${i}.jsonl"
    CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
      --mode suppress --model-tag "$TAG" --dataset "$dataset" \
      --run-kind "$kind" --seed "$seed" --lexicon core \
      --shard-id "$i" --num-shards 2 --max-context 32768 --think-tokens 0 --batch-size 16 \
      --jobs "$jobs" --out "$outfile" \
      >> "$LOG/plws_${TAG}_${dataset}_s${seed}_${kind}_sh${i}.log" 2>&1 &
    pids+=("$!")
  done
  local pid rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  return "$rc"
}

for seed in 42 0 1 123; do
  for dataset in amc23 gsm8k; do
    for kind in low high mix; do
      suppress_kind "$dataset" "$seed" "$kind"
    done
  done
done
echo "[plws] AMC23+GSM8K first-window suppress done $(date -Is)"
