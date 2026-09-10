#!/usr/bin/env bash
# 补 14B olympiadbench s0/s1/s123 第一扇窗后压。只占 0/1。
# jobs 写 jobs_oly*.jsonl，不改 leftover_jump 五集原件。
set +e
set -u
AE="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PLWS_ROOT="$AE"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
AE_PY="$AE/.venv/bin/python"
[[ -x "$AE_PY" ]] || AE_PY="$PY"
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
RUN_ROOT="$AE/results/runs/plws/window_first/k_4/lexicon_core"
LOG="$RUN_ROOT/logs"
mkdir -p "$LOG"

G="$AE/results/dense_G_r1_14b/olympiadbench/seed_123/per_sample.json"
if [[ ! -s "$G" ]]; then
  echo "[oly14] G s123 background (not blocking GPU) $(date -Is)"
  PYTHONPATH="$AE" "$AE_PY" scripts/compute_dense_G.py \
    --dataset olympiadbench \
    --dense-root "$AE/results/dense_G_r1_14b/olympiadbench/seed_123/dense_puma" \
    --out "$G" \
    --workers 8 >> "$LOG/g_r1_14b_oly_s123.log" 2>&1 &
else
  echo "[oly14] skip G already there"
fi

echo "[oly14] export oly leftover s0/s1/s123 $(date -Is)"
"$PY" scripts/export_leftover_suppress_jobs.py \
  --model-tag r1_14b --kinds all --seeds 0,1,123 \
  --datasets olympiadbench --out-stem jobs_oly

suppress_kind() {
  local seed="$1" kind="$2"
  local folder="$RUN_ROOT/r1_14b/olympiadbench/seed_${seed}/jobs"
  local old_folder="$AE/results/leftover_jump/r1_14b_s${seed}"
  local jobs="$folder/${kind}.jsonl"
  local old_jobs="$old_folder/jobs_oly.jsonl"
  if [[ "$kind" != "low" ]]; then
    old_jobs="$old_folder/jobs_oly_${kind}.jsonl"
  fi
  if [[ ! -f "$jobs" ]]; then
    jobs="$old_jobs"
  fi
  if [[ ! -s "$jobs" ]]; then
    echo "[oly14] skip empty s${seed} $kind"
    return 0
  fi
  echo "[oly14] suppress s${seed} $kind n=$(wc -l < "$jobs") $(date -Is)"
  local outdir="$RUN_ROOT/r1_14b/olympiadbench/seed_${seed}/scores/$kind"
  mkdir -p "$outdir"
  local i gpu outfile
  local pids=()
  for i in 0 1; do
    gpu="$i"
    outfile="$outdir/shard_$((80 + i)).jsonl"
    CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
      --mode suppress --model-tag r1_14b --dataset olympiadbench \
      --run-kind "$kind" --seed "$seed" --lexicon core \
      --shard-id "$i" --num-shards 2 --max-context 32768 --think-tokens 0 --batch-size 16 \
      --jobs "$jobs" --out "$outfile" \
      >> "$LOG/oly14_s${seed}_${kind}_sh${i}.log" 2>&1 &
    pids+=("$!")
  done
  local pid rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  return "$rc"
}

for seed in 0 1 123; do
  for kind in high mix low; do
    suppress_kind "$seed" "$kind"
  done
done
echo "[oly14] 14B oly s0/s1/s123 suppress done $(date -Is)"
