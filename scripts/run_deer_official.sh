#!/usr/bin/env bash
# Run one DEER cell under the canonical PUMA Full-CoT host protocol.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
GPU="${GPU:?set GPU}"
# shellcheck source=lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"
plws_runtime_init
PUMA_ROOT="$(cd "$PUMA_ROOT" && pwd)"
MODEL="${MODEL:-$(plws_model_path "$MODEL_TAG")}"
OUT="${OUT:-$ROOT/results/baselines/deer/puma_fullcot_32k_v2/$MODEL_TAG/$DATASET/seed_$SEED}"
LIMIT="${LIMIT:-0}"
RESULT="$OUT/deer.jsonl"
DATA="$PUMA_ROOT/data/${DATASET}_test.jsonl"

[[ -f "$DATA" ]] || { echo "ERROR: missing dataset $DATA" >&2; exit 1; }
mkdir -p "$OUT"

expected="$("$PY" - "$DATA" "$LIMIT" <<'PY'
import sys
from pathlib import Path
count = sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip())
limit = int(sys.argv[2])
print(min(count, limit) if limit > 0 else count)
PY
)"
if "$PY" - "$RESULT" "$OUT/manifest.json" "$expected" "$MODEL_TAG" "$DATASET" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

result, manifest_path = map(Path, sys.argv[1:3])
expected, model, dataset, seed = int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
if not result.is_file() or not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
rows = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
is_qwen3 = model.startswith("qwen3_")
expected_profile = "qwen3" if is_qwen3 else "standard"
expected_ratio = 0.8 if is_qwen3 else 0.6
expected_policy = "avg2" if is_qwen3 else "avg1"
deer = manifest.get("deer", {})
valid = (
    len(rows) == expected
    and manifest.get("method") == "deer"
    and manifest.get("protocol_id") == "puma-fullcot-32k-v2"
    and manifest.get("model_tag") == model
    and manifest.get("dataset") == dataset
    and manifest.get("seed") == seed
    and manifest.get("fullcot_generation_tokens") == 32768
    and manifest.get("truncated_answer_fix_tokens") == 2048
    and manifest.get("max_model_len") == 37888
    and deer.get("family_policy") == expected_profile
    and deer.get("think_ratio") == expected_ratio
    and deer.get("confidence_policy") == expected_policy
    and deer.get("require_probe_think_close") is is_qwen3
    and all(row.get("protocol_id") == "puma-fullcot-32k-v2" for row in rows)
    and all(row.get("deer_family_policy") == expected_profile for row in rows)
    and all(row.get("think_ratio") == expected_ratio for row in rows)
    and all(row.get("confidence_policy") == expected_policy for row in rows)
    and all("Error:" not in str(row.get("generated_text", "")) for row in rows)
)
raise SystemExit(0 if valid else 1)
PY
then
  echo "[deer-v2] skip complete $MODEL_TAG $DATASET seed=$SEED"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LENS_DISABLE=1
plws_export_cuda_runtime

tmp="$OUT/.deer.jsonl.$$.part"
rm -f "$tmp"
cd "$PUMA_ROOT"
limit_args=()
if [[ "$LIMIT" -gt 0 ]]; then
  limit_args+=(--limit "$LIMIT")
fi
"$PY" baselines/deer/vllm_deer.py \
  --model_name_or_path "$MODEL" \
  --dataset_dir "$PUMA_ROOT/data" \
  --dataset "$DATASET" \
  --output_path "$tmp" \
  --seed "$SEED" \
  --gpu-memory-utilization "${DEER_GPU_MEMORY_UTILIZATION:-0.90}" \
  --max_generated_tokens 32768 \
  --answer_fix_max_tokens 2048 \
  --max-model-len 37888 \
  --prompt_version default \
  --protocol_id puma-fullcot-32k-v2 \
  --max_judge_steps 10 \
  --threshold 0.95 \
  --points 1 \
  --trust-remote-code \
  "${limit_args[@]}" \
  2>&1 | tee "$OUT/run.log"

actual="$("$PY" - "$tmp" <<'PY'
import sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip()))
PY
)"
[[ "$actual" == "$expected" ]] || {
  echo "ERROR: incomplete DEER output ($actual/$expected rows)" >&2
  exit 1
}
mv -f "$tmp" "$RESULT"
"$PY" - "$RESULT" "$OUT/manifest.json" "$MODEL" "$MODEL_TAG" "$DATASET" "$SEED" "$expected" <<'PY'
import json
import sys
from pathlib import Path
from transformers import GenerationConfig

result, manifest_path = map(Path, sys.argv[1:3])
model, model_tag, dataset = sys.argv[3:6]
seed, expected = int(sys.argv[6]), int(sys.argv[7])
rows = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
first = rows[0] if rows else {}
family_fields = (
    "deer_family_policy",
    "think_ratio",
    "confidence_policy",
    "threshold",
    "require_probe_think_close",
)
if (
    len(rows) != expected
    or any(row.get("protocol_id") != "puma-fullcot-32k-v2" for row in rows)
    or any("Error:" in str(row.get("generated_text", "")) for row in rows)
    or any(
        any(row.get(field) != first.get(field) for field in family_fields)
        for row in rows
    )
):
    raise SystemExit("DEER v2 row validation failed")
config = GenerationConfig.from_pretrained(model, trust_remote_code=True)
top_k = getattr(config, "top_k", -1)
if top_k is None:
    top_k = -1
manifest = {
    "schema_version": 2,
    "method": "deer",
    "protocol_id": "puma-fullcot-32k-v2",
    "model": model,
    "model_tag": model_tag,
    "dataset": dataset,
    "seed": seed,
    "status": "succeeded",
    "expected_records": expected,
    "fullcot_generation_tokens": 32768,
    "prompt_reserve_tokens": 3072,
    "truncated_answer_fix_tokens": 2048,
    "max_model_len": 37888,
    "prompt_version": "default",
    "sampling": {
        "temperature": getattr(config, "temperature", 0.6),
        "top_p": getattr(config, "top_p", 0.95),
        "top_k": top_k,
    },
    "deer": {
        "family_policy": first["deer_family_policy"],
        "think_ratio": first["think_ratio"],
        "confidence_policy": first["confidence_policy"],
        "require_probe_think_close": first["require_probe_think_close"],
        "threshold": first["threshold"],
        "max_judge_steps": 10,
        "probe_decoding": "greedy",
        "points": 1,
    },
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
PY
echo "[deer-v2] done $OUT"
