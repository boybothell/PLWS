#!/usr/bin/env bash
# One-question GPU checks for DEER's standard and Qwen3 family policies.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PUMA_ROOT="$ROOT/../PUMA"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
GPU="${GPU:-6}"
OUT="$ROOT/results/experiments/deer_family_policy_pilot"
DATASET=math-500

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
export LD_LIBRARY_PATH="$(
  "$PY" - <<'PY'
from pathlib import Path
root = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
print(":".join(sorted({
    str(path)
    for path in (root / ".venv" / "lib").glob("**/nvidia/*/lib")
    if path.is_dir()
})))
PY
):${LD_LIBRARY_PATH:-}"

run_one() {
  local name="$1"
  local model="$2"
  local seed="$3"
  local target="$OUT/$name/deer.jsonl"
  mkdir -p "$(dirname "$target")"
  cd "$PUMA_ROOT"
  "$PY" baselines/deer/vllm_deer.py \
    --model_name_or_path "$model" \
    --dataset_dir "$PUMA_ROOT/data" \
    --dataset "$DATASET" \
    --output_path "$target" \
    --limit 1 \
    --batch_size 1 \
    --seed "$seed" \
    --max_generated_tokens 32768 \
    --answer_fix_max_tokens 2048 \
    --max-model-len 37888 \
    --prompt_version default \
    --protocol_id puma-fullcot-32k-v2 \
    --max_judge_steps 10 \
    --threshold 0.95 \
    --points 1 \
    --dump_checks \
    --trust-remote-code
}

run_one standard /mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B 42
run_one qwen3 /mnt/d/lsj/models/Qwen3-4B 42

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
standard = json.loads((root / "standard/deer.jsonl").read_text().strip())
qwen3 = json.loads((root / "qwen3/deer.jsonl").read_text().strip())
valid = (
    standard.get("deer_family_policy") == "standard"
    and standard.get("think_ratio") == 0.6
    and standard.get("confidence_policy") == "avg1"
    and standard.get("require_probe_think_close") is False
    and qwen3.get("deer_family_policy") == "qwen3"
    and qwen3.get("think_ratio") == 0.8
    and qwen3.get("confidence_policy") == "avg2"
    and qwen3.get("require_probe_think_close") is True
    and (
        not qwen3.get("high_prob")
        or qwen3.get("probe_think_closed") is True
    )
    and all(
        not row.get("method_limit_reached")
        or not row.get("used_truncated_answer_fix")
        for row in (standard, qwen3)
    )
)
status = {
    "state": "succeeded" if valid else "failed",
    "standard_exit": standard.get("exit_reason"),
    "qwen3_exit": qwen3.get("exit_reason"),
    "qwen3_probe_think_closed": qwen3.get("probe_think_closed"),
}
(root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
raise SystemExit(0 if valid else 1)
PY
