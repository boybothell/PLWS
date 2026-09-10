#!/usr/bin/env bash
# MATH / 奥赛 / GPQA missing seeds: official PUMA → dense_G → 密探k4 rewrite.
# 4 GPUs only (0-3). Does not overwrite seed-42 dense flats.
#
#   GPUS=0,1,2,3 bash scripts/run_missing_nonaime_seeds_4gpu.sh
set -euo pipefail
AE=/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit
LOGD="$AE/results/default_dense_gate/nonaime_seeds_4gpu"
mkdir -p "$LOGD"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ ${#GPU_ARR[@]} -ne 4 ]]; then
  echo "need exactly 4 GPUs, got ${GPUS}" >&2
  exit 1
fi

export VLLM_LENS_DISABLE=1
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

model_path() {
  case "$1" in
    r1_7b) echo /mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B ;;
    nemotron_8b) echo /mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1 ;;
    r1_14b) echo /mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B ;;
    qwen3_30b_a3b) echo /mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507 ;;
    *) echo "unknown model $1" >&2; return 1 ;;
  esac
}

run_puma_list() {
  local gpu="$1"
  local list="$2"
  local log="$3"
  {
    echo "[puma gpu=$gpu] start $(date -Is)"
    while read -r MODEL_TAG DATASET SEED; do
      [[ -z "${MODEL_TAG:-}" || "$MODEL_TAG" == \#* ]] && continue
      echo "[puma gpu=$gpu] $MODEL_TAG $DATASET seed=$SEED $(date -Is)"
      MODEL="$(model_path "$MODEL_TAG")" \
      MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" SEED="$SEED" GPU="$gpu" \
        bash "$AE/scripts/run_puma_official.sh" \
        || echo "[puma gpu=$gpu] FAIL $MODEL_TAG $DATASET s$SEED"
    done < "$list"
    echo "[puma gpu=$gpu] done $(date -Is)"
  } >> "$log" 2>&1
}

run_dense_list() {
  local gpus="$1"
  local tp="$2"
  local list="$3"
  local log="$4"
  {
    echo "[dense gpus=$gpus tp=$tp] start $(date -Is)"
    while read -r MODEL_TAG DATASET SEED; do
      [[ -z "${MODEL_TAG:-}" || "$MODEL_TAG" == \#* ]] && continue
      echo "[dense gpus=$gpus] $MODEL_TAG $DATASET seed=$SEED $(date -Is)"
      MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" SEED="$SEED" \
      GPUS="$gpus" TP="$tp" SEED_LAYOUT=1 \
        bash "$AE/scripts/run_dense_trials_model.sh" \
        || echo "[dense gpus=$gpus] FAIL $MODEL_TAG $DATASET s$SEED"
    done < "$list"
    echo "[dense gpus=$gpus] done $(date -Is)"
  } >> "$log" 2>&1
}

run_regen_list() {
  local gpu="$1"
  local list="$2"
  local log="$3"
  {
    echo "[regen gpu=$gpu] start $(date -Is)"
    while read -r MODEL_TAG DATASET SEED TP; do
      [[ -z "${MODEL_TAG:-}" || "$MODEL_TAG" == \#* ]] && continue
      echo "[regen gpu=$gpu] $MODEL_TAG $DATASET seed=$SEED TP=$TP $(date -Is)"
      MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" SEED="$SEED" GPU="$gpu" TP="$TP" \
        bash "$AE/scripts/run_default_regen.sh" \
        || echo "[regen gpu=$gpu] FAIL $MODEL_TAG $DATASET s$SEED"
    done < "$list"
    echo "[regen gpu=$gpu] done $(date -Is)"
  } >> "$log" 2>&1
}

# --- job files --------------------------------------------------------------
P1_0="$LOGD/puma_tp1_g0.jobs"
P1_1="$LOGD/puma_tp1_g1.jobs"
P1_2="$LOGD/puma_tp1_g2.jobs"
P1_3="$LOGD/puma_tp1_g3.jobs"
cat > "$P1_0" <<'EOF'
r1_7b math-500 0
r1_7b math-500 1
r1_7b math-500 123
r1_7b gpqa-diamond 0
r1_7b gpqa-diamond 1
EOF
cat > "$P1_1" <<'EOF'
r1_7b gpqa-diamond 123
r1_7b olympiadbench 0
r1_7b olympiadbench 1
r1_7b olympiadbench 123
nemotron_8b gpqa-diamond 0
EOF
cat > "$P1_2" <<'EOF'
nemotron_8b math-500 0
nemotron_8b math-500 1
nemotron_8b math-500 123
nemotron_8b gpqa-diamond 1
nemotron_8b gpqa-diamond 123
EOF
cat > "$P1_3" <<'EOF'
nemotron_8b olympiadbench 0
nemotron_8b olympiadbench 1
nemotron_8b olympiadbench 123
EOF

P2_A="$LOGD/puma_tp2_a.jobs"
P2_B="$LOGD/puma_tp2_b.jobs"
cat > "$P2_A" <<'EOF'
r1_14b math-500 0
r1_14b math-500 1
r1_14b math-500 123
r1_14b olympiadbench 0
r1_14b olympiadbench 1
r1_14b olympiadbench 123
r1_14b gpqa-diamond 0
r1_14b gpqa-diamond 1
r1_14b gpqa-diamond 123
EOF
cat > "$P2_B" <<'EOF'
qwen3_30b_a3b math-500 0
qwen3_30b_a3b math-500 1
qwen3_30b_a3b math-500 123
qwen3_30b_a3b olympiadbench 0
qwen3_30b_a3b olympiadbench 1
qwen3_30b_a3b olympiadbench 123
qwen3_30b_a3b gpqa-diamond 0
qwen3_30b_a3b gpqa-diamond 1
qwen3_30b_a3b gpqa-diamond 123
EOF

D1_0="$LOGD/dense_tp1_g0.jobs"
D1_1="$LOGD/dense_tp1_g1.jobs"
D1_2="$LOGD/dense_tp1_g2.jobs"
D1_3="$LOGD/dense_tp1_g3.jobs"
cat > "$D1_0" <<'EOF'
r1_7b math-500 0
r1_7b math-500 1
r1_7b math-500 123
r1_7b gpqa-diamond 0
r1_7b gpqa-diamond 1
r1_7b gpqa-diamond 123
EOF
cat > "$D1_1" <<'EOF'
r1_7b olympiadbench 0
r1_7b olympiadbench 1
r1_7b olympiadbench 123
EOF
cat > "$D1_2" <<'EOF'
nemotron_8b math-500 0
nemotron_8b math-500 1
nemotron_8b math-500 123
nemotron_8b gpqa-diamond 0
nemotron_8b gpqa-diamond 1
nemotron_8b gpqa-diamond 123
EOF
cat > "$D1_3" <<'EOF'
nemotron_8b olympiadbench 0
nemotron_8b olympiadbench 1
nemotron_8b olympiadbench 123
EOF

D2_A="$LOGD/dense_tp2_a.jobs"
D2_B="$LOGD/dense_tp2_b.jobs"
cat > "$D2_A" <<'EOF'
r1_14b math-500 0
r1_14b math-500 1
r1_14b math-500 123
r1_14b olympiadbench 0
r1_14b olympiadbench 1
r1_14b olympiadbench 123
r1_14b gpqa-diamond 0
r1_14b gpqa-diamond 1
r1_14b gpqa-diamond 123
EOF
cat > "$D2_B" <<'EOF'
qwen3_30b_a3b math-500 0
qwen3_30b_a3b math-500 1
qwen3_30b_a3b math-500 123
qwen3_30b_a3b olympiadbench 42
qwen3_30b_a3b olympiadbench 0
qwen3_30b_a3b olympiadbench 1
qwen3_30b_a3b olympiadbench 123
qwen3_30b_a3b gpqa-diamond 0
qwen3_30b_a3b gpqa-diamond 1
qwen3_30b_a3b gpqa-diamond 123
EOF

R1_0="$LOGD/regen_tp1_g0.jobs"
R1_1="$LOGD/regen_tp1_g1.jobs"
R1_2="$LOGD/regen_tp1_g2.jobs"
R1_3="$LOGD/regen_tp1_g3.jobs"
cat > "$R1_0" <<'EOF'
r1_7b math-500 0 1
r1_7b math-500 1 1
r1_7b math-500 123 1
r1_7b gpqa-diamond 0 1
r1_7b gpqa-diamond 1 1
r1_7b gpqa-diamond 123 1
r1_14b math-500 0 1
r1_14b math-500 1 1
r1_14b math-500 123 1
EOF
cat > "$R1_1" <<'EOF'
r1_7b olympiadbench 0 1
r1_7b olympiadbench 1 1
r1_7b olympiadbench 123 1
r1_14b olympiadbench 0 1
r1_14b olympiadbench 1 1
r1_14b olympiadbench 123 1
EOF
cat > "$R1_2" <<'EOF'
nemotron_8b math-500 0 1
nemotron_8b math-500 1 1
nemotron_8b math-500 123 1
nemotron_8b gpqa-diamond 0 1
nemotron_8b gpqa-diamond 1 1
nemotron_8b gpqa-diamond 123 1
r1_14b gpqa-diamond 0 1
r1_14b gpqa-diamond 1 1
r1_14b gpqa-diamond 123 1
EOF
cat > "$R1_3" <<'EOF'
nemotron_8b olympiadbench 0 1
nemotron_8b olympiadbench 1 1
nemotron_8b olympiadbench 123 1
EOF

R2_A="$LOGD/regen_tp2_a.jobs"
R2_B="$LOGD/regen_tp2_b.jobs"
cat > "$R2_A" <<'EOF'
qwen3_30b_a3b math-500 0 2
qwen3_30b_a3b math-500 1 2
qwen3_30b_a3b math-500 123 2
qwen3_30b_a3b gpqa-diamond 0 2
qwen3_30b_a3b gpqa-diamond 1 2
qwen3_30b_a3b gpqa-diamond 123 2
EOF
cat > "$R2_B" <<'EOF'
qwen3_30b_a3b olympiadbench 42 2
qwen3_30b_a3b olympiadbench 0 2
qwen3_30b_a3b olympiadbench 1 2
qwen3_30b_a3b olympiadbench 123 2
EOF

ec=0
echo "[suite] phase1 PUMA TP=1 $(date -Is)" | tee -a "$LOGD/main.log"
run_puma_list "${GPU_ARR[0]}" "$P1_0" "$LOGD/puma_g0.log" &
p0=$!
run_puma_list "${GPU_ARR[1]}" "$P1_1" "$LOGD/puma_g1.log" &
p1=$!
run_puma_list "${GPU_ARR[2]}" "$P1_2" "$LOGD/puma_g2.log" &
p2=$!
run_puma_list "${GPU_ARR[3]}" "$P1_3" "$LOGD/puma_g3.log" &
p3=$!
wait "$p0" || ec=1
wait "$p1" || ec=1
wait "$p2" || ec=1
wait "$p3" || ec=1
echo "[suite] phase1 done ec=$ec $(date -Is)" | tee -a "$LOGD/main.log"

echo "[suite] phase2 PUMA TP=2 $(date -Is)" | tee -a "$LOGD/main.log"
run_puma_list "${GPU_ARR[0]},${GPU_ARR[1]}" "$P2_A" "$LOGD/puma_tp2_a.log" &
pa=$!
run_puma_list "${GPU_ARR[2]},${GPU_ARR[3]}" "$P2_B" "$LOGD/puma_tp2_b.log" &
pb=$!
wait "$pa" || ec=1
wait "$pb" || ec=1
echo "[suite] phase2 done ec=$ec $(date -Is)" | tee -a "$LOGD/main.log"

echo "[suite] phase3 dense TP=1 $(date -Is)" | tee -a "$LOGD/main.log"
run_dense_list "${GPU_ARR[0]}" 1 "$D1_0" "$LOGD/dense_g0.log" &
p0=$!
run_dense_list "${GPU_ARR[1]}" 1 "$D1_1" "$LOGD/dense_g1.log" &
p1=$!
run_dense_list "${GPU_ARR[2]}" 1 "$D1_2" "$LOGD/dense_g2.log" &
p2=$!
run_dense_list "${GPU_ARR[3]}" 1 "$D1_3" "$LOGD/dense_g3.log" &
p3=$!
wait "$p0" || ec=1
wait "$p1" || ec=1
wait "$p2" || ec=1
wait "$p3" || ec=1
echo "[suite] phase3 done ec=$ec $(date -Is)" | tee -a "$LOGD/main.log"

echo "[suite] phase4 dense TP=2 $(date -Is)" | tee -a "$LOGD/main.log"
run_dense_list "${GPU_ARR[0]},${GPU_ARR[1]}" 2 "$D2_A" "$LOGD/dense_tp2_a.log" &
pa=$!
run_dense_list "${GPU_ARR[2]},${GPU_ARR[3]}" 2 "$D2_B" "$LOGD/dense_tp2_b.log" &
pb=$!
wait "$pa" || ec=1
wait "$pb" || ec=1
echo "[suite] phase4 done ec=$ec $(date -Is)" | tee -a "$LOGD/main.log"

echo "[suite] export candidates $(date -Is)" | tee -a "$LOGD/main.log"
"$AE/.venv/bin/python" - <<'PY' >> "$LOGD/export.log" 2>&1
import sys
from pathlib import Path
AE = Path("/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit")
sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd

jobs = []
for model in ("r1_7b", "nemotron_8b", "r1_14b"):
    for ds in ("math-500", "olympiadbench", "gpqa-diamond"):
        for seed in (0, 1, 123):
            jobs.append((model, ds, seed))
for ds in ("math-500", "gpqa-diamond"):
    for seed in (0, 1, 123):
        jobs.append(("qwen3_30b_a3b", ds, seed))
for seed in (42, 0, 1, 123):
    jobs.append(("qwen3_30b_a3b", "olympiadbench", seed))
for model, dataset, seed in jobs:
    try:
        meta = dd.export_candidates(model, dataset, seed)
        if meta.get("missing"):
            print(f"skip {model} {dataset} s{seed} missing", flush=True)
    except Exception as exc:
        print(f"fail {model} {dataset} s{seed}: {exc}", flush=True)
PY
echo "[suite] export done $(date -Is)" | tee -a "$LOGD/main.log"

echo "[suite] phase5 regen TP=1 $(date -Is)" | tee -a "$LOGD/main.log"
run_regen_list "${GPU_ARR[0]}" "$R1_0" "$LOGD/regen_g0.log" &
p0=$!
run_regen_list "${GPU_ARR[1]}" "$R1_1" "$LOGD/regen_g1.log" &
p1=$!
run_regen_list "${GPU_ARR[2]}" "$R1_2" "$LOGD/regen_g2.log" &
p2=$!
run_regen_list "${GPU_ARR[3]}" "$R1_3" "$LOGD/regen_g3.log" &
p3=$!
wait "$p0" || ec=1
wait "$p1" || ec=1
wait "$p2" || ec=1
wait "$p3" || ec=1
echo "[suite] phase5 done ec=$ec $(date -Is)" | tee -a "$LOGD/main.log"

echo "[suite] phase6 regen TP=2 $(date -Is)" | tee -a "$LOGD/main.log"
run_regen_list "${GPU_ARR[0]},${GPU_ARR[1]}" "$R2_A" "$LOGD/regen_tp2_a.log" &
pa=$!
run_regen_list "${GPU_ARR[2]},${GPU_ARR[3]}" "$R2_B" "$LOGD/regen_tp2_b.log" &
pb=$!
wait "$pa" || ec=1
wait "$pb" || ec=1
echo "[suite] all done ec=$ec $(date -Is)" | tee -a "$LOGD/main.log"
exit "$ec"
