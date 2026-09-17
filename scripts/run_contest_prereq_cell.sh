#!/usr/bin/env bash
# Fill Full-CoT / PUMA / dense / window jobs for one contest cell.
# 7B / Nemotron / 14B: reuse existing Full-CoT; sample only when answers are absent.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set dataset}"
SEED="${SEED:?set seed}"
GPU="${GPU:?set one GPU or a TP lane}"

# shellcheck source=lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"
plws_runtime_init
MODEL="$(plws_model_path "$MODEL_TAG")"
ALIGN_CONF="$(plws_align_conf "$MODEL_TAG")"

TP="$(awk -F',' '{print NF}' <<<"$GPU")"
EXPECTED_TP="${PLWS_TP:-$("$PY" - "$ROOT" "$MODEL_TAG" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import gpu_count
print(gpu_count(sys.argv[2]))
PY
)}"
if [[ "$TP" -ne "$EXPECTED_TP" ]]; then
  echo "ERROR: $MODEL_TAG contest cells require TP=$EXPECTED_TP, got GPU=$GPU" >&2
  exit 1
fi

eval "$("$PY" - "$ROOT" "$MODEL_TAG" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import protocol_for
p = protocol_for(sys.argv[2])
print(f"PROTOCOL_ID={p.protocol_id}")
print(f"MAX_TOKENS={p.generation_tokens}")
print(f"ANSWER_FIX={p.answer_fix_tokens}")
print(f"PROMPT_RESERVE={p.prompt_reserve_tokens}")
print(f"MAX_MODEL_LEN={p.max_model_len}")
PY
)"
export PROTOCOL_ID MAX_TOKENS ANSWER_FIX PROMPT_RESERVE MAX_MODEL_LEN

export PLWS_ROOT="$ROOT"
export VLLM_LENS_DISABLE=1
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
plws_export_cuda_runtime

if [[ "$SEED" == 42 ]]; then
  PUMA_DIR="$ROOT/results/baselines/puma/puma_offline_${MODEL_TAG}/$DATASET"
else
  PUMA_DIR="$ROOT/results/baselines/puma/puma_offline_${MODEL_TAG}_s${SEED}/$DATASET"
fi
SAMPLE="$ROOT/samples/$MODEL_TAG/$DATASET/seed_${SEED}"

jobs_ready() {
  "$PY" - "$ROOT" "$MODEL_TAG" "$DATASET" "$SEED" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.matrix import jobs_present
from plws.paths import PLWSPaths
raise SystemExit(0 if jobs_present(PLWSPaths(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])) else 1)
PY
}

sample_matches_protocol() {
  [[ -f "$SAMPLE/answers.json" && -f "$SAMPLE/sample_meta.json" ]] || return 1
  "$PY" - "$ROOT" "$MODEL_TAG" "$DATASET" "$SEED" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import sample_matches_protocol
from plws.paths import PLWSPaths
raise SystemExit(0 if sample_matches_protocol(
    PLWSPaths(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
) else 1)
PY
}

may_sample_fullcot() {
  "$PY" - "$ROOT" "$MODEL_TAG" "$DATASET" "$SEED" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from plws.contest import may_sample_fullcot
from plws.paths import PLWSPaths
raise SystemExit(0 if may_sample_fullcot(
    PLWSPaths(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
) else 1)
PY
}

stash() {
  local src="$1" dest="$2"
  [[ -e "$src" ]] || return 0
  mkdir -p "$(dirname "$dest")"
  mv "$src" "$dest"
  echo "[contest-prereq] stashed $src -> $dest"
}

puma_ready() {
  [[ -f "$PUMA_DIR/statistics.json" && -f "$PUMA_DIR/prefixed_answers.json" ]]
}

if [[ "${FULLCOT_ONLY:-0}" == "1" ]] && sample_matches_protocol && [[ -f "$SAMPLE/answers.json" ]]; then
  echo "[contest-prereq] skip complete fullcot-only $MODEL_TAG $DATASET seed=$SEED $PROTOCOL_ID"
  exit 0
fi

if puma_ready && jobs_ready && sample_matches_protocol; then
  echo "[contest-prereq] skip complete $MODEL_TAG $DATASET seed=$SEED $PROTOCOL_ID"
  exit 0
fi

if ! sample_matches_protocol; then
  if ! may_sample_fullcot; then
    echo "ERROR: refused to regenerate Full-CoT for $MODEL_TAG $DATASET seed=$SEED" >&2
    exit 2
  fi
  if [[ -e "$SAMPLE" ]]; then
    stamp="$(date +%Y%m%dT%H%M%S)"
    stash "$SAMPLE" "$ROOT/samples/_stale_non_contest/$MODEL_TAG/$DATASET/seed_${SEED}_$stamp"
    stash "$PUMA_DIR" "$ROOT/results/baselines/puma/_stale_non_contest/${MODEL_TAG}_s${SEED}/${DATASET}_$stamp"
    stash "$ROOT/results/upstream/dense_trials/dense_G_${MODEL_TAG}/$DATASET/seed_${SEED}" \
      "$ROOT/results/upstream/dense_trials/_stale_non_contest/dense_G_${MODEL_TAG}/$DATASET/seed_${SEED}_$stamp"
    stash "$ROOT/results/runs/plws/window_first/k_4/lexicon_core/$MODEL_TAG/$DATASET/seed_${SEED}" \
      "$ROOT/results/runs/plws/_stale_non_contest/$MODEL_TAG/$DATASET/seed_${SEED}_$stamp"
  fi
fi

echo "[contest-prereq] start $(date -Is) $MODEL_TAG $DATASET seed=$SEED gpu=$GPU $PROTOCOL_ID gen=$MAX_TOKENS"

if [[ "${FULLCOT_ONLY:-0}" == "1" ]]; then
  if ! sample_matches_protocol || [[ ! -f "$SAMPLE/answers.json" ]]; then
    if ! may_sample_fullcot; then
      echo "ERROR: refused to regenerate Full-CoT for $MODEL_TAG $DATASET seed=$SEED" >&2
      exit 2
    fi
    PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
      ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
      PROTOCOL_ID="$PROTOCOL_ID" MAX_TOKENS="$MAX_TOKENS" ANSWER_FIX="$ANSWER_FIX" \
      PROMPT_RESERVE="$PROMPT_RESERVE" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      bash "$ROOT/scripts/run_puma_aligned_sample.sh"
  fi
  echo "[contest-prereq] fullcot-only stop $MODEL_TAG $DATASET seed=$SEED"
  exit 0
fi

if puma_ready && sample_matches_protocol; then
  echo "[contest-prereq] reuse complete PUMA $PUMA_DIR"
elif [[ -f "$PUMA_DIR/answers.json" ]] && sample_matches_protocol; then
  echo "[contest-prereq] finish incomplete PUMA from existing answers $PUMA_DIR"
  PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
    ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    PUMA_DIR="$PUMA_DIR" \
    bash "$ROOT/scripts/run_puma_official.sh"
else
  PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
    ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    PROTOCOL_ID="$PROTOCOL_ID" MAX_TOKENS="$MAX_TOKENS" ANSWER_FIX="$ANSWER_FIX" \
    PROMPT_RESERVE="$PROMPT_RESERVE" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    bash "$ROOT/scripts/run_puma_aligned_sample.sh"
  PLWS_ROOT="$ROOT" MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" \
    ALIGN_CONF="$ALIGN_CONF" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    PUMA_DIR="$PUMA_DIR" \
    bash "$ROOT/scripts/run_puma_official.sh"
fi

DENSE_GPU_ONLY=1 PLWS_ROOT="$ROOT" MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" \
  SEED="$SEED" GPUS="$GPU" TP="$TP" SEED_LAYOUT=1 PUMA_DIR="$PUMA_DIR" \
  bash "$ROOT/scripts/run_dense_trials_model.sh"

PLWS_ROOT="$ROOT" PLWS_PY="$PY" MODEL_TAG="$MODEL_TAG" DATASET="$DATASET" \
  SEED="$SEED" CONTEST_RUN_ROOT="${CONTEST_RUN_ROOT:-$ROOT/results/runs/contest_queue}" \
  bash "$ROOT/scripts/detach_export_leftover_jobs.sh"

echo "[contest-prereq] gpu released $(date -Is) $MODEL_TAG $DATASET seed=$SEED"
