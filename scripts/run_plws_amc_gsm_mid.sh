#!/usr/bin/env bash
# 8B + 14B AMC23/GSM8K：采样 → 官方 PUMA → 密探 → 第一扇窗后压。
# 只占 4/5。0–3 的 8B/14B leftover 和 6–7 的 s1 不动。
# 8B 写回 leftover_jump（五集已齐，mixer 不会再 export）。
# 14B jobs 写 jobs_amcgsm*.jsonl，避免 0–3 mixer 盖掉 leftover_jump。
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

SEVEN="math-500,olympiadbench,gpqa-diamond,aime24,aime25,amc23,gsm8k"

model_path() {
  case "$1" in
    nemotron_8b) echo /mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1 ;;
    r1_14b) echo /mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B ;;
    *) echo "unknown tag $1" >&2; return 1 ;;
  esac
}

align_conf() {
  case "$1" in
    nemotron_8b) echo Nemotron.conf ;;
    r1_14b) echo DS-14B.conf ;;
  esac
}

puma_dir() {
  local tag="$1" ds="$2" seed="$3"
  if [[ "$seed" != "42" ]]; then
    echo "$AE/results/puma_offline_${tag}_s${seed}/$ds"
  else
    echo "$AE/results/puma_offline_${tag}/$ds"
  fi
}

sample_done() { [[ -s "$AE/samples/$1/$2/seed_$3/answers.json" ]]; }
puma_done() {
  local d
  d="$(puma_dir "$1" "$2" "$3")"
  [[ -s "$d/statistics.json" && -s "$d/prefixed_answers.json" ]]
}
dense_done() { [[ -s "$AE/results/dense_G_$1/$2/seed_$3/dense_puma/trial_answers.json" ]]; }

sample_one() {
  local tag="$1" ds="$2" seed="$3" gpu="$4"
  if sample_done "$tag" "$ds" "$seed"; then
    echo "[plws-mid] skip sample $tag $ds s${seed}"
    return 0
  fi
  echo "[plws-mid] sample $tag $ds s${seed} gpu=${gpu} $(date -Is)"
  MODEL="$(model_path "$tag")" MODEL_TAG="$tag" ALIGN_CONF="$(align_conf "$tag")" \
    GPU="$gpu" DATASET="$ds" SEED="$seed" bash scripts/run_puma_aligned_sample.sh
}

puma_one() {
  local tag="$1" ds="$2" seed="$3" gpu="$4"
  if puma_done "$tag" "$ds" "$seed"; then
    echo "[plws-mid] skip puma $tag $ds s${seed}"
    return 0
  fi
  echo "[plws-mid] puma $tag $ds s${seed} gpu=${gpu} $(date -Is)"
  MODEL="$(model_path "$tag")" MODEL_TAG="$tag" DATASET="$ds" SEED="$seed" GPU="$gpu" \
    bash scripts/run_puma_official.sh
}

dense_one() {
  local tag="$1" ds="$2" seed="$3" gpus="$4" tp="$5"
  if dense_done "$tag" "$ds" "$seed"; then
    echo "[plws-mid] skip dense $tag $ds s${seed}"
    return 0
  fi
  if ! puma_done "$tag" "$ds" "$seed"; then
    echo "[plws-mid] skip dense $tag $ds s${seed}: puma incomplete"
    return 1
  fi
  echo "[plws-mid] dense $tag $ds s${seed} gpus=${gpus} tp=${tp} $(date -Is)"
  DENSE_GPU_ONLY=1 SEED_LAYOUT=1 MODEL_TAG="$tag" DATASET="$ds" SEED="$seed" \
    GPUS="$gpus" TP="$tp" bash scripts/run_dense_trials_model.sh
}

kick_g() {
  local tag="$1" ds="$2" seed="$3"
  local root="$AE/results/dense_G_${tag}/${ds}/seed_${seed}"
  if [[ -s "$root/per_sample.json" ]]; then
    return 0
  fi
  if [[ ! -s "$root/dense_puma/trial_answers.json" ]]; then
    return 0
  fi
  echo "[plws-mid] G background $tag $ds s${seed}"
  "$PY" scripts/compute_dense_G.py \
    --dataset "$ds" \
    --dense-root "$root/dense_puma" \
    --out "$root/per_sample.json" \
    --workers 8 >> "$LOG/g_${tag}_${ds}_s${seed}.log" 2>&1 &
}

run_cell() {
  local tag="$1" ds="$2" seed="$3" gpu="$4" tp="$5"
  local try
  for try in 1 2; do
    if sample_one "$tag" "$ds" "$seed" "$gpu"; then
      break
    fi
    echo "[plws-mid] sample FAIL $tag $ds s${seed} try=${try}"
    sleep 5
  done
  if ! sample_done "$tag" "$ds" "$seed"; then
    echo "[plws-mid] sample GIVE UP $tag $ds s${seed}"
    return 1
  fi
  for try in 1 2; do
    if puma_one "$tag" "$ds" "$seed" "$gpu"; then
      break
    fi
    echo "[plws-mid] puma FAIL $tag $ds s${seed} try=${try}"
    sleep 5
  done
  if ! puma_done "$tag" "$ds" "$seed"; then
    echo "[plws-mid] puma GIVE UP $tag $ds s${seed}"
    return 1
  fi
  for try in 1 2; do
    if dense_one "$tag" "$ds" "$seed" "$gpu" "$tp"; then
      return 0
    fi
    echo "[plws-mid] dense FAIL $tag $ds s${seed} try=${try}"
    sleep 5
  done
  echo "[plws-mid] dense GIVE UP $tag $ds s${seed}"
  return 1
}

gpu_loop() {
  local tag="$1" gpu="$2" tp="$3"
  shift 3
  echo "[plws-mid] gpu${gpu} $tag queue: $*"
  local spec ds seed
  for spec in "$@"; do
    ds="${spec%%:*}"; seed="${spec##*:}"
    run_cell "$tag" "$ds" "$seed" "$gpu" "$tp"
  done
  echo "[plws-mid] gpu${gpu} $tag queue done $(date -Is)"
}

export_jobs() {
  local tag="$1" stem="$2" datasets="$3"
  echo "[plws-mid] export $tag stem=${stem:-default} $(date -Is)"
  if [[ -n "$stem" ]]; then
    "$PY" scripts/export_leftover_suppress_jobs.py \
      --model-tag "$tag" --kinds all --seeds 42,0,1,123 \
      --datasets "$datasets" --out-stem "$stem"
  else
    "$PY" scripts/export_leftover_suppress_jobs.py \
      --model-tag "$tag" --kinds all --seeds 42,0,1,123 \
      --datasets "$datasets"
  fi
}

jobs_file() {
  local tag="$1" dataset="$2" seed="$3" kind="$4" stem="$5"
  local path="$RUN_ROOT/$tag/$dataset/seed_${seed}/jobs/${kind}.jsonl"
  local legacy="$AE/results/leftover_jump/${tag}_s${seed}"
  local old_path work
  if [[ -n "$stem" ]]; then
    if [[ "$kind" == "low" ]]; then
      old_path="$legacy/${stem}.jsonl"
    else
      old_path="$legacy/${stem}_${kind}.jsonl"
    fi
  elif [[ "$kind" == "low" ]]; then
    old_path="$legacy/jobs.jsonl"
  else
    old_path="$legacy/jobs_${kind}.jsonl"
  fi
  if [[ ! -f "$path" && -f "$old_path" ]]; then
    work="$RUN_ROOT/work/legacy_jobs/$tag/$dataset/seed_${seed}/${kind}.jsonl"
    mkdir -p "$(dirname "$work")"
    grep -F "\"dataset\": \"$dataset\"" "$old_path" > "$work" || true
    path="$work"
  fi
  echo "$path"
}

suppress_kind() {
  local tag="$1" dataset="$2" seed="$3" kind="$4" stem="$5"
  local jobs
  jobs="$(jobs_file "$tag" "$dataset" "$seed" "$kind" "$stem")"
  if [[ ! -s "$jobs" ]]; then
    echo "[plws-mid] skip empty $tag $dataset $kind s${seed}"
    return 0
  fi
  echo "[plws-mid] suppress $tag $dataset s${seed} $kind"
  local i gpu outdir outfile
  local pids=()
  outdir="$RUN_ROOT/$tag/$dataset/seed_${seed}/scores/$kind"
  mkdir -p "$outdir"
  # 分片仍是 0/1；写到 shard90/91，避开 0–3 mixer 正在写的 shard0–5
  for i in 0 1; do
    gpu=$((4 + i))
    outfile="$outdir/shard_$((90 + i)).jsonl"
    CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
      --mode suppress --model-tag "$tag" --dataset "$dataset" \
      --run-kind "$kind" --seed "$seed" --lexicon core \
      --shard-id "$i" --num-shards 2 --max-context 32768 --think-tokens 0 --batch-size 16 \
      --jobs "$jobs" --out "$outfile" \
      >> "$LOG/plws_${tag}_${dataset}_s${seed}_${kind}_sh${i}.log" 2>&1 &
    pids+=("$!")
  done
  local pid rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  return "$rc"
}

suppress_model() {
  local tag="$1" stem="$2"
  local dataset seed kind
  for seed in 42 0 1 123; do
    for dataset in amc23 gsm8k; do
      for kind in low high mix; do
        suppress_kind "$tag" "$dataset" "$seed" "$kind" "$stem"
      done
    done
  done
}

kick_all_g() {
  local tag="$1"
  local ds seed
  for ds in amc23 gsm8k; do
    for seed in 42 0 1 123; do
      kick_g "$tag" "$ds" "$seed"
    done
  done
}

echo "[plws-mid] start 8B+14B AMC23/GSM8K $(date -Is)"

echo "[plws-mid] ===== 8B TP=1 双卡并行 ====="
gpu_loop nemotron_8b 4 1 amc23:42 amc23:0 gsm8k:42 gsm8k:1 > "$LOG/plws_mid_8b_gpu4.log" 2>&1 &
pid4=$!
gpu_loop nemotron_8b 5 1 amc23:1 amc23:123 gsm8k:0 gsm8k:123 > "$LOG/plws_mid_8b_gpu5.log" 2>&1 &
pid5=$!
wait "$pid4"
wait "$pid5"
kick_all_g nemotron_8b
export_jobs nemotron_8b "" "$SEVEN"
suppress_model nemotron_8b ""
echo "[plws-mid] 8B AMC23+GSM8K suppress done $(date -Is)"

echo "[plws-mid] ===== 14B TP=2 两卡一起 ====="
for spec in amc23:42 amc23:0 amc23:1 amc23:123 gsm8k:42 gsm8k:0 gsm8k:1 gsm8k:123; do
  ds="${spec%%:*}"; seed="${spec##*:}"
  run_cell r1_14b "$ds" "$seed" "4,5" 2
done
kick_all_g r1_14b
export_jobs r1_14b jobs_amcgsm "amc23,gsm8k"
suppress_model r1_14b jobs_amcgsm
echo "[plws-mid] 14B AMC23+GSM8K suppress done $(date -Is)"
echo "[plws-mid] ALL done $(date -Is)"
