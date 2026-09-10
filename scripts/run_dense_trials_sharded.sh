#!/usr/bin/env bash
# Shard every-step trial generation (dense probe) across GPUs, then merge.
# No --respect-embedding-filter → probe every reasoning step.
#
#   DATASET=olympiadbench GPUS=0,1,2,3 bash scripts/run_dense_trials_sharded.sh
set -euo pipefail

AE=/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit
PUMA_ROOT=/mnt/d/lsj/visual-latent-tts/repos/PUMA
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
MODEL="${MODEL:-/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B}"
DATASET="${DATASET:?}"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}

case "$DATASET" in
  math-500) PUMA_DIR="${PUMA_DIR:-$AE/results/math500_official/puma_ds7b}" ;;
  *) PUMA_DIR="${PUMA_DIR:-$AE/results/puma_offline_r1_7b/$DATASET}" ;;
esac

OUT="${OUT:-$AE/results/cross_probe_gate_r1_7b/$DATASET/dense_puma}"
mkdir -p "$OUT/shards"
FILTERED="$PUMA_DIR/filtered_steps.json"
ANSWERS="$PUMA_DIR/answers.json"
[[ -f "$FILTERED" && -f "$ANSWERS" ]] || { echo "missing $PUMA_DIR"; exit 1; }

cp -f "$ANSWERS" "$OUT/answers.json"
printf '%s\n' "arm=dense_puma dataset=$DATASET sharded_gpus=$GPUS every_step=1" > "$OUT/meta.txt"

export VLLM_LENS_DISABLE=1
export LD_LIBRARY_PATH="$(
python3 - <<'PY'
from pathlib import Path
root = Path('/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm')
print(':'.join(sorted({str(p) for p in (root/'.venv'/'lib').glob('**/nvidia/*/lib') if p.is_dir()})))
PY
):${LD_LIBRARY_PATH:-}"

# split questions into N shards (1-based question_idx preserved via enumerate)
"$AE/.venv/bin/python" - <<PY
import json
from pathlib import Path
rows = json.loads(Path("$FILTERED").read_text())
n = $N
out = Path("$OUT/shards")
for i in range(n):
    shard = [r for j, r in enumerate(rows) if j % n == i]
    # annotate absolute 1-based idx for merge
    payload = []
    for j, r in enumerate(rows):
        if j % n != i:
            continue
        rr = dict(r)
        rr["_abs_question_idx"] = j + 1
        payload.append(rr)
    (out / f"filtered_steps_shard{i}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"shard{i}: {len(payload)} questions", flush=True)
PY

cd "$PUMA_ROOT"
if [[ -f "$PUMA_DIR/_DS-7B.local.conf" ]]; then
  # shellcheck source=/dev/null
  source "$PUMA_DIR/_DS-7B.local.conf"
else
  # shellcheck source=/dev/null
  source configs/DS-7B.conf
fi

pids=()
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  shard_q="$OUT/shards/filtered_steps_shard${i}.json"
  shard_t="$OUT/shards/trial_answers_shard${i}.json"
  logf="$OUT/shards/shard${i}.log"
  if [[ -f "$shard_t" ]]; then
    echo "[dense-shard] skip existing $shard_t"
    continue
  fi
  echo "[dense-shard] GPU=$gpu shard=$i → $shard_t"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" puma/gen_trial_answers.py \
      --questions-file "$shard_q" \
      --output-file "$shard_t" \
      --model "$MODEL" \
      --max-tokens "${MAX_TRIAL_TOKENS:-30}" \
      --temperature "${TEMPERATURE:-0.6}" \
      --top_p "${TOP_P:-0.95}" \
      --tensor-parallel-size 1 \
      --dataset "$DATASET" \
      --trial-decoding "${TRIAL_DECODING:-sampling}" \
      --confidence-mode "${CONFIDENCE_MODE:-token_in_boxed}" \
      --confidence-aggregation geometric \
      --prompt-version "${PROMPT_VERSION:-default}" \
      --seed "${SEED:-42}" \
      2>&1 | tee "$logf"
  ) &
  pids+=($!)
done

ec=0
for pid in "${pids[@]:-}"; do
  if ! wait "$pid"; then ec=1; fi
done
[[ $ec -eq 0 ]] || { echo "shard failed"; exit 1; }

# merge: remap question_idx to absolute
export OUT N
"$AE/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
n = int(os.environ["N"])
merged = []
for i in range(n):
    fq = out / "shards" / f"filtered_steps_shard{i}.json"
    ft = out / "shards" / f"trial_answers_shard{i}.json"
    qs = json.loads(fq.read_text())
    local_to_abs = {j + 1: int(q["_abs_question_idx"]) for j, q in enumerate(qs)}
    trials = json.loads(ft.read_text())
    for e in trials:
        loc = int(e["question_idx"])
        e["question_idx"] = local_to_abs[loc]
        merged.append(e)
merged.sort(key=lambda e: (int(e["question_idx"]), int(e["stopped_len"])))
path = out / "trial_answers.json"
path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
print(f"merged {len(merged)} trials → {path}", flush=True)
PY

echo "[dense-shard] DONE $OUT $(date -Is)"
