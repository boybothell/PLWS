#!/usr/bin/env bash
# Run a list of DEFAULT regen jobs on one GPU assignment.
# Each line: MODEL_TAG DATASET SEED [TP]
#
#   GPU=2 bash scripts/run_default_regen_queue.sh <<'EOF'
#   r1_7b math-500 42 1
#   r1_7b gpqa-diamond 42 1
#   EOF
set -euo pipefail
AE=/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit
GPU="${GPU:?}"
while read -r MODEL_TAG DATASET SEED TP rest; do
  [[ -z "${MODEL_TAG:-}" || "$MODEL_TAG" == \#* ]] && continue
  TP="${TP:-1}"
  echo "[queue] $MODEL_TAG $DATASET seed=$SEED GPU=$GPU TP=$TP"
  MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" TP="$TP" \
    bash "$AE/scripts/run_default_regen.sh"
done
echo "[queue] all done GPU=$GPU $(date -Is)"
