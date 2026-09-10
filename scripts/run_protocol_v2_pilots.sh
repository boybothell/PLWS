#!/usr/bin/env bash
# One-question GPU integration checks for canonical DEER and Answer Convergence.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PUMA_ROOT="$ROOT/../PUMA"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
GPU="${GPU:-6}"
MODEL=/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B
MODEL_TAG=r1_7b
DATASET=math-500
SEED=42
PILOT="$ROOT/results/experiments/protocol_v2_pilot"
FULLCOT="$PILOT/fullcot/$MODEL_TAG/$DATASET/seed_$SEED"
DEER="$PILOT/deer/$MODEL_TAG/$DATASET/seed_$SEED"
AC="$PILOT/answer_convergence/$MODEL_TAG/$DATASET/seed_$SEED"

mkdir -p "$FULLCOT" "$DEER" "$AC"
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

cd "$PUMA_ROOT"
"$PY" puma/run_vllm.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --dataset_path "$PUMA_ROOT/data/${DATASET}_test.jsonl" \
  --output_path "$FULLCOT/answers.json" \
  --limit 1 \
  --max_tokens 32768 \
  --answer_max_tokens 2048 \
  --prompt-version default \
  --seed "$SEED"

"$PY" - "$FULLCOT/sample_meta.json" "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "protocol_id": "puma-fullcot-32k-v2",
    "model": sys.argv[2],
    "model_tag": "r1_7b",
    "dataset": "math-500",
    "seed": 42,
    "max_tokens": 32768,
    "answer_fix_max_tokens": 2048,
    "prompt_reserve_tokens": 3072,
    "max_model_len": 37888,
    "prompt_version": "default",
    "n_questions": 1,
}, indent=2) + "\n")
PY

"$PY" baselines/deer/vllm_deer.py \
  --model_name_or_path "$MODEL" \
  --dataset_dir "$PUMA_ROOT/data" \
  --dataset "$DATASET" \
  --output_path "$DEER/deer.jsonl" \
  --limit 1 \
  --batch_size 1 \
  --seed "$SEED" \
  --max_generated_tokens 32768 \
  --answer_fix_max_tokens 2048 \
  --max-model-len 37888 \
  --prompt_version default \
  --protocol_id puma-fullcot-32k-v2 \
  --max_judge_steps 10 \
  --threshold 0.95 \
  --points 1 \
  --trust-remote-code

cd "$ROOT"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
  "$ROOT/scripts/run_answer_convergence_cell.py" \
  --model "$MODEL" \
  --model-tag "$MODEL_TAG" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --sample "$FULLCOT/answers.json" \
  --output-dir "$AC" \
  --threshold 10 \
  --probe-max-tokens 100 \
  --max-model-len 37888 \
  --limit 1

"$PY" - "$DEER/deer.jsonl" "$AC/manifest.json" "$PILOT/status.json" <<'PY'
import json
import sys
from pathlib import Path

deer = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text().splitlines()
    if line.strip()
]
ac = json.loads(Path(sys.argv[2]).read_text())
valid = (
    len(deer) == 1
    and deer[0].get("protocol_id") == "puma-fullcot-32k-v2"
    and ac.get("protocol_id") == "puma-fullcot-32k-v2"
    and ac.get("expected_records") == 1
)
status = {"state": "succeeded" if valid else "failed", "deer_rows": len(deer)}
Path(sys.argv[3]).write_text(json.dumps(status, indent=2) + "\n")
raise SystemExit(0 if valid else 1)
PY
