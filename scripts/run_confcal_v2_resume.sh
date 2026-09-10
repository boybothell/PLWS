#!/bin/bash
# Resume remaining v2 extracts. Already-written keys are skipped.
set -u
AE="$(cd "$(dirname "$0")/.." && pwd)"
VPY="/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python"
export VLLM_LENS_DISABLE=1
GPUS="${GPUS:-1,4,6,7}"
cd "$AE"
echo "==== $(date) universal H-B/H-C ===="
"$VPY" scripts/run_confcal_universal_serial.py --gpus "$GPUS" || echo "WARN universal exited $?"
echo "==== $(date) keytoken H-A math-500 ===="
"$VPY" scripts/run_confcal_keytoken_serial.py --gpus "$GPUS" --only math-500 || echo "WARN keytoken math-500 exited $?"
echo "==== $(date) keytoken H-A olympiadbench ===="
"$VPY" scripts/run_confcal_keytoken_serial.py --gpus "$GPUS" --only olympiadbench || echo "WARN keytoken olympiadbench exited $?"
echo "==== $(date) keytoken H-A gpqa-diamond ===="
"$VPY" scripts/run_confcal_keytoken_serial.py --gpus "$GPUS" --only gpqa-diamond || echo "WARN keytoken gpqa-diamond exited $?"
echo "==== $(date) analyze ===="
python3 scripts/analyze_confcal_v2.py || python3 scripts/analyze_confcal_v2_available.py
echo "==== $(date) done ===="
