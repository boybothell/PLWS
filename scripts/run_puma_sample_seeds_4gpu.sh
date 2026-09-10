#!/usr/bin/env bash
# Fill samples/<model>/<dataset>/seed_{42,123,0,1} to 4 seeds (PUMA-aligned).
# Phase A: non-32B on 4 GPUs. Phase B: r1_32b TP=2 last.
#
# Usage:
#   bash scripts/run_puma_sample_seeds_4gpu.sh
#   GPUS=0,1,2,3 TP_GPUS=0,1 bash scripts/run_puma_sample_seeds_4gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_puma_aligned_sample.sh"
SEEDS=(42 123 0 1)
GPUS_CSV="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
TP_GPUS="${TP_GPUS:-${GPUS[0]},${GPUS[1]}}"
LOGDIR="$ROOT/samples/_logs"
mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/seed_fill_manifest.txt"
: > "$MANIFEST"

declare -A MODEL_PATH ALIGN
MODEL_PATH[r1_7b]=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B
ALIGN[r1_7b]=DS-7B.conf
MODEL_PATH[r1_14b]=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B
ALIGN[r1_14b]=DS-14B.conf
MODEL_PATH[nemotron_8b]=/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1
ALIGN[nemotron_8b]=Nemotron.conf
MODEL_PATH[qwen3_30b_a3b]=/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507
ALIGN[qwen3_30b_a3b]=Q30B-T.conf
MODEL_PATH[r1_32b]=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-32B
ALIGN[r1_32b]=DS-32B.conf

# existing combos (exclude mathl5_140 — not PUMA data / different sampler)
PHASE_A_MODELS=(r1_7b r1_14b nemotron_8b qwen3_30b_a3b)
PHASE_B_MODELS=(r1_32b)
DATASETS_FIVE=(aime24 aime25 math-500 gpqa-diamond olympiadbench)

have_answers() {
  local m="$1" d="$2" s="$3"
  [[ -f "$ROOT/samples/$m/$d/seed_${s}/answers.json" ]]
}

datasets_for_model() {
  local m="$1"
  if [[ "$m" == "qwen3_30b_a3b" ]]; then
    echo "math-500"
  else
    printf '%s\n' "${DATASETS_FIVE[@]}"
  fi
}

build_jobs() {
  local -n _out=$1
  shift
  local models=("$@")
  _out=()
  for m in "${models[@]}"; do
    while IFS= read -r d; do
      [[ -z "$d" ]] && continue
      # only fill combos that already exist (at least one seed dir)
      if [[ ! -d "$ROOT/samples/$m/$d" ]]; then
        continue
      fi
      for s in "${SEEDS[@]}"; do
        if have_answers "$m" "$d" "$s"; then
          echo "HAVE $m $d seed_$s" >> "$MANIFEST"
        else
          _out+=("$m|$d|$s")
          echo "NEED $m $d seed_$s" >> "$MANIFEST"
        fi
      done
    done < <(datasets_for_model "$m")
  done
}

run_worker() {
  local gpu="$1"
  shift
  local jobs=("$@")
  local sess="attn-ee-seedfill-gpu${gpu//,/-}"
  local log="$LOGDIR/seedfill_gpu${gpu//,/-}.log"
  tmux kill-session -t "$sess" 2>/dev/null || true
  # serialize jobs for this GPU into a bash script
  local jobfile="$LOGDIR/jobs_gpu${gpu//,/-}.sh"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo "ROOT='$ROOT'"
    echo "SCRIPT='$SCRIPT'"
    echo "LOG='$log'"
    echo "echo \"[worker GPU=$gpu] start \$(date -Is) n=${#jobs[@]}\" | tee \"\$LOG\""
    for spec in "${jobs[@]}"; do
      IFS='|' read -r m d s <<<"$spec"
      echo "echo \"[worker] === $m / $d / seed_$s ===\" | tee -a \"\$LOG\""
      echo "MODEL='${MODEL_PATH[$m]}' MODEL_TAG='$m' ALIGN_CONF='${ALIGN[$m]}' \\"
      echo "  GPU='$gpu' DATASET='$d' SEED='$s' bash \"\$SCRIPT\" 2>&1 | tee -a \"\$LOG\""
    done
    echo "echo \"[worker GPU=$gpu] ALL done \$(date -Is)\" | tee -a \"\$LOG\""
    echo 'exec bash'
  } > "$jobfile"
  chmod +x "$jobfile"
  tmux new-session -d -s "$sess" bash "$jobfile"
  echo "[suite] launched $sess with ${#jobs[@]} jobs → $log"
}

# --- Phase A ---
JOBS_A=()
build_jobs JOBS_A "${PHASE_A_MODELS[@]}"
echo "[suite] Phase A missing jobs: ${#JOBS_A[@]}"
# round-robin to 4 GPUs
declare -a Q0 Q1 Q2 Q3
idx=0
for spec in "${JOBS_A[@]+"${JOBS_A[@]}"}"; do
  case $((idx % ${#GPUS[@]})) in
    0) Q0+=("$spec") ;;
    1) Q1+=("$spec") ;;
    2) Q2+=("$spec") ;;
    3) Q3+=("$spec") ;;
  esac
  idx=$((idx + 1))
done

for i in "${!GPUS[@]}"; do
  var="Q$i"
  eval "arr=(\"\${${var}[@]+\"\${${var}[@]}\"}\")"
  if ((${#arr[@]})); then
    run_worker "${GPUS[$i]}" "${arr[@]}"
  else
    echo "[suite] GPU ${GPUS[$i]} idle (no Phase A jobs)"
  fi
done

# --- Phase B waiter: after Phase A sessions end, run 32B TP=2 ---
JOBS_B=()
build_jobs JOBS_B "${PHASE_B_MODELS[@]}"
echo "[suite] Phase B (r1_32b) missing jobs: ${#JOBS_B[@]}"
sess32="attn-ee-seedfill-r1_32b"
tmux kill-session -t "$sess32" 2>/dev/null || true
job32="$LOGDIR/jobs_r1_32b.sh"
log32="$LOGDIR/seedfill_r1_32b.log"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "ROOT='$ROOT'; SCRIPT='$SCRIPT'; LOG='$log32'"
  echo "echo \"[r1_32b] waiting Phase A \$(date -Is)\" | tee \"\$LOG\""
  for i in "${!GPUS[@]}"; do
    echo "while ! grep -q 'ALL done' \"$LOGDIR/seedfill_gpu${GPUS[$i]}.log\" 2>/dev/null; do sleep 60; done"
    echo "echo \"[r1_32b] Phase A GPU ${GPUS[$i]} done\" | tee -a \"\$LOG\""
  done
  echo "echo \"[r1_32b] Phase A done, start TP=2 on $TP_GPUS \$(date -Is)\" | tee -a \"\$LOG\""
  if ((${#JOBS_B[@]} == 0)); then
    echo "echo '[r1_32b] nothing to do' | tee -a \"\$LOG\""
  else
    for spec in "${JOBS_B[@]}"; do
      IFS='|' read -r m d s <<<"$spec"
      echo "echo \"[r1_32b] === $m / $d / seed_$s ===\" | tee -a \"\$LOG\""
      echo "MODEL='${MODEL_PATH[$m]}' MODEL_TAG='$m' ALIGN_CONF='${ALIGN[$m]}' \\"
      echo "  GPU='$TP_GPUS' DATASET='$d' SEED='$s' bash \"\$SCRIPT\" 2>&1 | tee -a \"\$LOG\""
    done
  fi
  echo "echo \"[r1_32b] ALL done \$(date -Is)\" | tee -a \"\$LOG\""
  echo 'exec bash'
} > "$job32"
chmod +x "$job32"
# If no Phase A jobs, touch empty ALL done logs so 32B can start
for i in "${!GPUS[@]}"; do
  lg="$LOGDIR/seedfill_gpu${GPUS[$i]}.log"
  if [[ ! -f "$lg" ]]; then
    echo "ALL done (no jobs)" > "$lg"
  fi
done
tmux new-session -d -s "$sess32" bash "$job32"
echo "[suite] launched $sess32 (waits Phase A) → $log32"
echo "[suite] manifest → $MANIFEST"
wc -l "$MANIFEST"
rg -c '^NEED' "$MANIFEST" || true
rg -c '^HAVE' "$MANIFEST" || true
