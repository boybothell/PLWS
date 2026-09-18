#!/usr/bin/env bash
# Dense every-step trials for soft-Pareto multimodel.
#
#   MODEL_TAG=nemotron_8b DATASET=math-500 GPUS=3,4 bash scripts/run_dense_trials_model.sh
#   MODEL_TAG=r1_32b DATASET=olympiadbench GPUS=1,2 TP=2 bash scripts/run_dense_trials_model.sh
#
# Dense trials are frozen upstream inputs shared by PLWS and diagnostics.
# They are not PLWS outputs.
#
# Writes:
#   results/upstream/dense_trials/dense_G_<MODEL_TAG>/<DATASET>/dense_puma/{trial_answers,answers}.json
#   results/upstream/dense_trials/dense_G_<MODEL_TAG>/<DATASET>/per_sample.json
set -euo pipefail

AE="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PLWS_ROOT="$AE"
# shellcheck source=lib/runtime.sh
ROOT="$AE"
source "$AE/scripts/lib/runtime.sh"
plws_runtime_init
PUMA_ROOT="$(cd "$PUMA_ROOT" && pwd)"
AE_PY="${AE_PY:-$PY}"
export PYTHONPATH="$AE/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_TAG="${MODEL_TAG:?}"
DATASET="${DATASET:?}"
GPUS="${GPUS:?}"
TP="${TP:-${PLWS_TP:-1}}"
SEED="${SEED:-42}"

MODEL="$(plws_model_path "$MODEL_TAG")"
CONF="$(plws_align_conf "$MODEL_TAG")"

if [[ "$SEED" != "42" ]]; then
  PUMA_DIR="${PUMA_DIR:-$AE/results/baselines/puma/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET}"
else
  PUMA_DIR="${PUMA_DIR:-$AE/results/baselines/puma/puma_offline_${MODEL_TAG}/$DATASET}"
fi
if [[ "$MODEL_TAG" == r1_7b && "$DATASET" == math-500 && "$SEED" == "42" ]]; then
  PUMA_DIR="${PUMA_DIR_OVERRIDE:-$AE/results/baselines/official/math500_official/puma_ds7b}"
fi
# AIME / 非 42 的 seed 走 seed_*/，避免盖掉 seed42 平铺
if [[ "${SEED_LAYOUT:-}" == "1" || "$SEED" != "42" || "$DATASET" == "aime24" || "$DATASET" == "aime25" || "$DATASET" == "aime26" || "$DATASET" == "brumo25" || "$DATASET" == "hmmt25" || "$DATASET" == "amc23" || "$DATASET" == "gsm8k" ]]; then
  OUT="${OUT:-$AE/results/upstream/dense_trials/dense_G_${MODEL_TAG}/$DATASET/seed_${SEED}/dense_puma}"
  G_OUT="${G_OUT:-$AE/results/upstream/dense_trials/dense_G_${MODEL_TAG}/$DATASET/seed_${SEED}/per_sample.json}"
else
  OUT="${OUT:-$AE/results/upstream/dense_trials/dense_G_${MODEL_TAG}/$DATASET/dense_puma}"
  G_OUT="${G_OUT:-$AE/results/upstream/dense_trials/dense_G_${MODEL_TAG}/$DATASET/per_sample.json}"
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPU=${#GPU_ARR[@]}
if (( N_GPU % TP != 0 )); then
  echo "GPUS length ($N_GPU) must be divisible by TP=$TP" >&2
  exit 1
fi
N=$((N_GPU / TP))

FILTERED="$PUMA_DIR/filtered_steps.json"
ANSWERS="$PUMA_DIR/answers.json"
PUMA_TRIALS="$PUMA_DIR/trial_answers.json"

mkdir -p "$OUT"
RUN_FINISHED=0
write_status() {
  local state="$1" message="${2:-}"
  "$AE_PY" - "$OUT/status.json" "$state" "$message" <<'PY'
import sys
from plws.artifacts import atomic_write_json, utc_now
atomic_write_json(sys.argv[1], {
    "schema_version": 1,
    "state": sys.argv[2],
    "message": sys.argv[3] or None,
    "updated_at": utc_now(),
})
PY
}
on_exit() {
  local code=$?
  if [[ "$RUN_FINISHED" != "1" ]]; then
    write_status failed "run_dense_trials_model.sh exited with code $code" || true
  fi
}
trap on_exit EXIT
"$AE_PY" - "$OUT/manifest.json" "$MODEL_TAG" "$MODEL" "$DATASET" "$SEED" "$GPUS" "$TP" "$PUMA_DIR" "$G_OUT" "$PUMA_TRIALS" <<'PY'
import sys
from plws.artifacts import atomic_write_json, utc_now
atomic_write_json(sys.argv[1], {
    "schema_version": 1,
    "method": "dense_trials",
    "experiment": "dense_trials",
    "model_tag": sys.argv[2],
    "model": sys.argv[3],
    "dataset": sys.argv[4],
    "seed": int(sys.argv[5]),
    "gpus": sys.argv[6],
    "tensor_parallel_size": int(sys.argv[7]),
    "puma_dir": sys.argv[8],
    "g_output": sys.argv[9],
    "trial_reuse_mode": "reuse-puma-generated-trials-and-fill-missing-v1",
    "puma_trial_answers": sys.argv[10],
    "created_at": utc_now(),
})
PY
write_status running

[[ -f "$FILTERED" && -f "$ANSWERS" && -f "$PUMA_TRIALS" ]] || {
  echo "missing PUMA prerequisite under $PUMA_DIR" >&2
  exit 1
}

if [[ -f "$OUT/trial_answers.json" && -f "$G_OUT" ]]; then
  write_status succeeded "existing complete output reused"
  RUN_FINISHED=1
  echo "[dense-model] skip complete $MODEL_TAG $DATASET seed=$SEED"
  exit 0
fi
if [[ "${DENSE_GPU_ONLY:-}" == "1" && -f "$OUT/trial_answers.json" ]]; then
  write_status succeeded "existing GPU output reused"
  RUN_FINISHED=1
  echo "[dense-model] GPU phase already $OUT/trial_answers.json"
  exit 0
fi

mkdir -p "$OUT/shards"
ANSWERS_TMP="$OUT/.answers.json.tmp.$$"
cp -f "$ANSWERS" "$ANSWERS_TMP"
mv -f "$ANSWERS_TMP" "$OUT/answers.json"
META_TMP="$OUT/.meta.txt.tmp.$$"
printf '%s\n' \
  "arm=dense_puma model_tag=$MODEL_TAG model=$MODEL dataset=$DATASET" \
  "gpus=$GPUS tp=$TP n_shards=$N every_step=1" \
  > "$META_TMP"
mv -f "$META_TMP" "$OUT/meta.txt"

export VLLM_LENS_DISABLE=1
# Default 0.90 matches 14B. 32B on 80GB cannot init max_model_len=38000 at
# 0.90 (~4.5GiB KV, needs ~9.3GiB). The fill queue must pass
# DENSE_VLLM_GPU_MEMORY_UTILIZATION with the same value as PUMA (0.97 here).
# Do not inherit leftover-suppress 0.97 from VLLM_GPU_MEMORY_UTILIZATION.
export VLLM_GPU_MEMORY_UTILIZATION="${DENSE_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
plws_export_cuda_runtime

"$AE_PY" -m plws.dense_reuse prepare \
  --filtered-steps "$FILTERED" \
  --puma-trials "$PUMA_TRIALS" \
  --shards-dir "$OUT/shards" \
  --num-shards "$N"

cd "$PUMA_ROOT"
# shellcheck source=/dev/null
if [[ -f "$PUMA_DIR/_local.conf" ]]; then
  source "$PUMA_DIR/_local.conf"
elif [[ -f "$PUMA_DIR/_${CONF%.conf}.local.conf" ]]; then
  source "$PUMA_DIR/_${CONF%.conf}.local.conf"
else
  source "configs/$CONF"
fi

trial_extra_args=()
if [[ -n "${GPQA_SOFTMAX_TEMPERATURE:-}" ]]; then
  trial_extra_args+=(--gpqa-softmax-temperature "$GPQA_SOFTMAX_TEMPERATURE")
fi

pids=()
for i in $(seq 0 $((N - 1))); do
  shard_q="$OUT/shards/missing_steps_shard${i}.json"
  shard_t="$OUT/shards/trial_answers_missing_shard${i}.json"
  logf="$OUT/shards/missing_shard${i}.log"
  if [[ ! -f "$shard_q" ]]; then
    echo "[dense-model] shard$i has no missing PUMA trials"
    continue
  fi
  if [[ -f "$shard_t" ]]; then
    echo "[dense-model] skip existing $shard_t"
    continue
  fi
  start=$((i * TP))
  pair=()
  for k in $(seq 0 $((TP - 1))); do
    pair+=("${GPU_ARR[$((start + k))]}")
  done
  gpu_str=$(IFS=,; echo "${pair[*]}")
  echo "[dense-model] GPUS=$gpu_str shard=$i TP=$TP → $shard_t"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_str"
    "$PY" puma/gen_trial_answers.py \
      --questions-file "$shard_q" \
      --output-file "$shard_t" \
      --model "$MODEL" \
      --max-tokens "${MAX_TRIAL_TOKENS:-30}" \
      --temperature "${TEMPERATURE:-0.6}" \
      --top_p "${TOP_P:-0.95}" \
      --tensor-parallel-size "$TP" \
      --dataset "$DATASET" \
      --trial-decoding "${TRIAL_DECODING:-sampling}" \
      --confidence-mode "${CONFIDENCE_MODE:-token_in_boxed}" \
      --confidence-aggregation "${CONFIDENCE_AGGREGATION:-geometric}" \
      --prompt-version "${PROMPT_VERSION:-default}" \
      --seed "${SEED:-42}" \
      --respect-embedding-filter \
      "${trial_extra_args[@]}" \
      2>&1 | tee "$logf"
  ) &
  pids+=($!)
done

ec=0
for pid in "${pids[@]:-}"; do
  if ! wait "$pid"; then ec=1; fi
done
[[ $ec -eq 0 ]] || { echo "shard failed"; exit 1; }

"$AE_PY" -m plws.dense_reuse merge \
  --filtered-steps "$FILTERED" \
  --puma-trials "$PUMA_TRIALS" \
  --shards-dir "$OUT/shards" \
  --num-shards "$N" \
  --output "$OUT/trial_answers.json"

if [[ "${DENSE_GPU_ONLY:-}" == "1" ]]; then
  write_status succeeded "GPU phase complete"
  RUN_FINISHED=1
  echo "[dense-model] GPU phase done $OUT/trial_answers.json"
  exit 0
fi

mkdir -p "$(dirname "$G_OUT")"
"$AE_PY" "$AE/scripts/compute_dense_G.py" \
  --dataset "$DATASET" \
  --dense-root "$OUT" \
  --out "$G_OUT" \
  --workers 8

write_status succeeded
RUN_FINISHED=1
echo "[dense-model] DONE $MODEL_TAG $DATASET $(date -Is)"
