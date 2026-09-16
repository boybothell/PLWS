#!/usr/bin/env bash
# Shared portable runtime paths for PLWS launch scripts.

plws_runtime_init() {
  local fallback="/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python"
  if [[ -f "${ROOT:-}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
  if [[ -n "${PLWS_PY:-}" ]]; then
    PY="$PLWS_PY"
  elif [[ -n "${PY:-}" ]]; then
    PY="$PY"
  elif [[ -x "$fallback" ]]; then
    PY="$fallback"
  else
    PY="$(command -v python3)"
  fi
  [[ -x "$PY" ]] || {
    echo "ERROR: Python is not executable: $PY" >&2
    return 2
  }

  PLWS_MODELS_ROOT="${PLWS_MODELS_ROOT:-/mnt/d/lsj/models}"
  PUMA_ROOT="${PUMA_ROOT:-$ROOT/../PUMA}"
  PLWS_DATA_ROOT="${PLWS_DATA_ROOT:-$PUMA_ROOT/data}"
  export PY PLWS_PY="$PY" PLWS_MODELS_ROOT PUMA_ROOT PLWS_DATA_ROOT
}

plws_model_path() {
  local directory
  case "$1" in
    qwen3_30b_a3b) directory="Qwen3-30B-A3B-Thinking-2507" ;;
    qwq_32b) directory="QwQ-32B" ;;
    qwen3_32b) directory="Qwen3-32B" ;;
    r1_7b) directory="DeepSeek-R1-Distill-Qwen-7B" ;;
    nemotron_8b) directory="Llama-3.1-Nemotron-Nano-8B-v1" ;;
    r1_14b) directory="DeepSeek-R1-Distill-Qwen-14B" ;;
    r1_1p5b) directory="DeepSeek-R1-Distill-Qwen-1.5B" ;;
    r1_llama_8b) directory="DeepSeek-R1-Distill-Llama-8B" ;;
    r1_32b) directory="DeepSeek-R1-Distill-Qwen-32B" ;;
    qwen3_4b) directory="Qwen3-4B" ;;
    qwen3_8b) directory="Qwen3-8B" ;;
    *)
      echo "ERROR: unknown MODEL_TAG=$1" >&2
      return 2
      ;;
  esac
  printf '%s/%s\n' "$PLWS_MODELS_ROOT" "$directory"
}

plws_align_conf() {
  case "$1" in
    qwen3_30b_a3b) echo "Q30B-T.conf" ;;
    nemotron_8b) echo "Nemotron.conf" ;;
    r1_14b) echo "DS-14B.conf" ;;
    r1_32b|qwen3_32b|qwq_32b) echo "DS-32B.conf" ;;
    *) echo "DS-7B.conf" ;;
  esac
}

plws_nvidia_library_path() {
  "$PY" - <<'PY'
import sys
from pathlib import Path

root = Path(sys.prefix)
print(":".join(sorted({
    str(path)
    for path in (root / "lib").glob("**/nvidia/*/lib")
    if path.is_dir()
})))
PY
}

plws_export_cuda_runtime() {
  local extra
  extra="$(plws_nvidia_library_path)"
  if [[ -n "$extra" ]]; then
    export LD_LIBRARY_PATH="$extra${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}
