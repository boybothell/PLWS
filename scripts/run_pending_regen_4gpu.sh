#!/usr/bin/env bash
# Rewrite 密探k4 prefixes on 4 GPUs, then grade vs official PUMA.
# Usage: GPUS=0,1,2,3 bash scripts/run_pending_regen_4gpu.sh
set -euo pipefail
AE=/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit
LOGD="$AE/results/default_dense_gate/regen_4gpu"
mkdir -p "$LOGD"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ ${#GPU_ARR[@]} -ne 4 ]]; then
  echo "need exactly 4 GPUs, got ${GPUS}" >&2
  exit 1
fi

run_list() {
  local gpu="$1"
  local list="$2"
  local log="$3"
  {
    echo "[worker gpu=$gpu] start $(date -Is)"
    while read -r MODEL_TAG DATASET SEED TP; do
      [[ -z "${MODEL_TAG:-}" || "$MODEL_TAG" == \#* ]] && continue
      echo "[worker gpu=$gpu] $MODEL_TAG $DATASET seed=$SEED TP=$TP $(date -Is)"
      MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" SEED="$SEED" GPU="$gpu" TP="$TP" \
        VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}" \
        VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
        bash "$AE/scripts/run_default_regen.sh" || echo "[worker gpu=$gpu] FAIL $MODEL_TAG $DATASET s$SEED"
    done < "$list"
    echo "[worker gpu=$gpu] done $(date -Is)"
  } >> "$log" 2>&1
}

TP1_0="$LOGD/tp1_gpu0.jobs"
TP1_1="$LOGD/tp1_gpu1.jobs"
TP1_2="$LOGD/tp1_gpu2.jobs"
TP1_3="$LOGD/tp1_gpu3.jobs"
cat > "$TP1_0" <<'EOF'
r1_14b math-500 42 1
r1_14b gpqa-diamond 42 1
r1_14b aime24 42 1
r1_14b aime24 0 1
r1_14b aime24 1 1
r1_14b aime24 123 1
EOF
cat > "$TP1_1" <<'EOF'
r1_14b olympiadbench 42 1
r1_14b aime25 42 1
r1_14b aime25 0 1
r1_14b aime25 1 1
r1_14b aime25 123 1
EOF
cat > "$TP1_2" <<'EOF'
qwen3_4b olympiadbench 42 1
qwen3_4b gpqa-diamond 42 1
qwen3_4b aime24 42 1
qwen3_4b aime24 0 1
qwen3_4b aime24 1 1
EOF
cat > "$TP1_3" <<'EOF'
qwen3_8b gpqa-diamond 42 1
qwen3_8b aime24 42 1
qwen3_8b aime24 0 1
qwen3_8b aime24 1 1
qwen3_8b aime24 123 1
qwen3_8b aime25 0 1
qwen3_8b aime25 1 1
qwen3_8b aime25 123 1
EOF

TP2_A="$LOGD/tp2_a.jobs"
TP2_B="$LOGD/tp2_b.jobs"
cat > "$TP2_A" <<'EOF'
r1_32b olympiadbench 42 2
r1_32b math-500 42 2
r1_32b gpqa-diamond 42 2
r1_32b aime24 42 2
r1_32b aime24 0 2
r1_32b aime24 1 2
r1_32b aime24 123 2
r1_32b aime25 42 2
r1_32b aime25 0 2
r1_32b aime25 1 2
r1_32b aime25 123 2
EOF
cat > "$TP2_B" <<'EOF'
qwen3_30b_a3b math-500 42 2
qwen3_30b_a3b gpqa-diamond 42 2
qwen3_30b_a3b aime24 42 2
qwen3_30b_a3b aime24 0 2
qwen3_30b_a3b aime24 1 2
qwen3_30b_a3b aime24 123 2
qwen3_30b_a3b aime25 42 2
qwen3_30b_a3b aime25 0 2
qwen3_30b_a3b aime25 1 2
qwen3_30b_a3b aime25 123 2
EOF

echo "[4gpu] wave1 TP=1 on ${GPU_ARR[*]} $(date -Is)" | tee -a "$LOGD/main.log"
run_list "${GPU_ARR[0]}" "$TP1_0" "$LOGD/gpu0.log" &
p0=$!
run_list "${GPU_ARR[1]}" "$TP1_1" "$LOGD/gpu1.log" &
p1=$!
run_list "${GPU_ARR[2]}" "$TP1_2" "$LOGD/gpu2.log" &
p2=$!
run_list "${GPU_ARR[3]}" "$TP1_3" "$LOGD/gpu3.log" &
p3=$!
ec=0
wait "$p0" || ec=1
wait "$p1" || ec=1
wait "$p2" || ec=1
wait "$p3" || ec=1
echo "[4gpu] wave1 done ec=$ec $(date -Is)" | tee -a "$LOGD/main.log"

echo "[4gpu] wave2 TP=2 $(date -Is)" | tee -a "$LOGD/main.log"
run_list "${GPU_ARR[0]},${GPU_ARR[1]}" "$TP2_A" "$LOGD/tp2_a.log" &
pa=$!
run_list "${GPU_ARR[2]},${GPU_ARR[3]}" "$TP2_B" "$LOGD/tp2_b.log" &
pb=$!
wait "$pa" || ec=1
wait "$pb" || ec=1
echo "[4gpu] all done ec=$ec $(date -Is)" | tee -a "$LOGD/main.log"
exit "$ec"
