#!/usr/bin/env bash
# Post-</think> trial probes: OlympiadBench R1-7B seed 42, fullcot + count-bias.
# GPUs 0-3, one cold load at a time.
set -euo pipefail
ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG=$ROOT/tmp/plws_step_probe/logs
PY=$ROOT/.venv/bin/python
SCRIPT=$ROOT/scripts/olympiad_post_agreement/run_post_think_probes.py
mkdir -p "$LOG"
export PATH="$ROOT/.venv/bin:$PATH"
unset PLWS_TP PLWS_LARGE_TP

gpu_mem() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

for i in 0 1 2 3; do
  log="$LOG/post_think_shard_${i}.log"
  : >"$log"
  echo "$(date -Is) start shard=$i gpu=$i"
  CUDA_VISIBLE_DEVICES=$i \
    VLLM_MAX_MODEL_LEN=38000 \
    VLLM_GPU_MEMORY_UTILIZATION=0.78 \
    PYTHONUNBUFFERED=1 \
    setsid --wait "$PY" "$SCRIPT" --shard-id "$i" --num-shards 4 \
      --protocol both \
      >>"$log" 2>&1 </dev/null &
  echo $! >"$LOG/post_think_shard_${i}.pid"
  ready=0
  for _ in $(seq 1 300); do
    if grep -q '\[post\] wrote=' "$log" 2>/dev/null; then
      ready=1
      break
    fi
    if grep -q 'nothing pending\|pending=0/' "$log" 2>/dev/null; then
      ready=1
      break
    fi
    if grep -q 'Traceback' "$log" 2>/dev/null; then
      echo "$(date -Is) shard $i failed" >&2
      tail -40 "$log" >&2 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "$(date -Is) shard $i did not start generating" >&2
    tail -40 "$log" >&2 || true
    exit 1
  fi
  echo "$(date -Is) shard=$i gpu_mem=$(gpu_mem "$i")MiB next cold load allowed"
done
echo "$(date -Is) all four shards launched"
wait
echo "$(date -Is) all four shards exited"
echo "$(date -Is) launcher exit=$?"
