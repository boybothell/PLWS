#!/usr/bin/env bash
# 7B 四种子：等到第一扇非 Low 再压。两卡 4/5 各自往下接，不等另一张。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
VENV_LIB=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/lib
LIBS="$(find "$VENV_LIB" -type d -path '*/nvidia/*/lib' | paste -sd:)"
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_LENS_DISABLE=1
OUT="$AE/results/first_hm_gate"
LOG="$OUT/logs"
mkdir -p "$LOG" "$OUT/jobs"
SEEDS=(42 0 1 123)

pending_shard() {
  local seed="$1"
  local shard="$2"
  local jobs="$OUT/jobs/r1_7b_s${seed}.jsonl"
  local folder="$OUT/r1_7b_s${seed}_suppress_hm"
  "$PY" - "$jobs" "$folder" "$shard" <<'PY'
import json, sys
from pathlib import Path
jobs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
jobs=[j for i,j in enumerate(jobs) if i % 2 == int(sys.argv[3])]
have=set()
folder=Path(sys.argv[2])
if folder.is_dir():
    for path in folder.glob("scores_shard*.jsonl"):
        for line in path.open():
            if not line.strip():
                continue
            try:
                rec=json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") in ("ok","too_long") and rec.get("uid"):
                have.add(str(rec["uid"]))
print(sum(1 for j in jobs if j["uid"] not in have))
PY
}

run_shard() {
  local seed="$1"
  local shard="$2"
  local gpu="$3"
  local jobs="$OUT/jobs/r1_7b_s${seed}.jsonl"
  mkdir -p "$OUT/r1_7b_s${seed}_suppress_hm"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag r1_7b \
    --run-kind hm \
    --seed "$seed" \
    --lexicon core \
    --shard-id "$shard" \
    --num-shards 2 \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 16 \
    --jobs "$jobs" \
    --out-root "$OUT" \
    >> "$LOG/s${seed}_sh${shard}.log" 2>&1
}

loop_gpu() {
  local shard="$1"
  local gpu="$2"
  for seed in "${SEEDS[@]}"; do
    local jobs="$OUT/jobs/r1_7b_s${seed}.jsonl"
    if [[ ! -s "$jobs" ]]; then
      "$PY" scripts/export_first_hm_jobs.py --model-tag r1_7b --seed "$seed" --out "$jobs"
    fi
    local pend
    pend="$(pending_shard "$seed" "$shard")"
    echo "gpu${gpu} s${seed} shard=${shard} pending=${pend}"
    if [[ "$pend" -eq 0 ]]; then
      continue
    fi
    while pgrep -f "score_leftover_suppress.py.*--seed ${seed} .*--shard-id ${shard} " >/dev/null; do
      echo "gpu${gpu} wait existing s${seed} shard=${shard}"
      sleep 8
    done
    pend="$(pending_shard "$seed" "$shard")"
    if [[ "$pend" -eq 0 ]]; then
      continue
    fi
    echo "gpu${gpu} launch s${seed} shard=${shard} pending=${pend}"
    run_shard "$seed" "$shard" "$gpu"
    echo "gpu${gpu} s${seed} shard=${shard} done pending=$(pending_shard "$seed" "$shard")"
  done
}

for seed in "${SEEDS[@]}"; do
  jobs="$OUT/jobs/r1_7b_s${seed}.jsonl"
  if [[ ! -s "$jobs" ]]; then
    "$PY" scripts/export_first_hm_jobs.py --model-tag r1_7b --seed "$seed" --out "$jobs"
  fi
done
"$PY" scripts/seed_first_hm_scores.py --model-tag r1_7b --seeds 42,0,1,123

loop_gpu 0 4 &
pid4=$!
loop_gpu 1 5 &
pid5=$!
echo "first-hm per-gpu loops pid4=$pid4 pid5=$pid5"
wait "$pid4"
wait "$pid5"
echo "first-hm 7B 4seed done"
