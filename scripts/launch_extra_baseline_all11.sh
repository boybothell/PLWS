#!/usr/bin/env bash
# Extra-baseline queue: AC + Dynasor on official five datasets, 11 models, seeds 42/0/1.
set -u
export PLWS_ROOT="${PLWS_ROOT:-/mnt/d/lsj/visual-latent-tts/repos/plws}"
export PUMA_ROOT="${PUMA_ROOT:-/mnt/d/lsj/visual-latent-tts/repos/PUMA}"
export PLWS_PY="${PLWS_PY:-/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python}"
export PYTHONPATH="${PLWS_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export EXTRA_BASELINE_RUN_ROOT="${EXTRA_BASELINE_RUN_ROOT:-${PLWS_ROOT}/results/runs/extra_baseline_fill}"
cd "$PLWS_ROOT" || exit 1
while true; do
  echo "$(date '+%F %T') start extra gpus=6,7 all-11-models"
  "$PLWS_PY" "$PLWS_ROOT/scripts/run_extra_baseline_queue.py" \
    --gpus 6,7 \
    --models r1_1p5b,r1_7b,r1_14b,r1_llama_8b,nemotron_8b,qwen3_4b,qwen3_8b,qwen3_32b,qwq_32b,qwen3_30b_a3b,r1_32b \
    --datasets amc23,aime25,gpqa-diamond,math-500,olympiadbench \
    --seeds 42,0,1 \
    --methods answer_convergence,dynasor
  code=$?
  echo "$(date '+%F %T') queue exit=$code"
  if [ "$code" -eq 0 ]; then
    echo 'clean finish'
    break
  fi
  echo 'non-zero, restart in 15s'
  sleep 15
done
echo 'wrapper parked'
read
