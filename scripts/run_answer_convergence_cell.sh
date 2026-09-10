#!/usr/bin/env bash
# Verified one-cell wrapper for Answer Convergence.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python}"
MODEL="${MODEL:?set MODEL}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
GPU="${GPU:?set GPU}"
OUT="${OUT:?set OUT}"
LIMIT="${LIMIT:-0}"

SAMPLE="$ROOT/samples/$MODEL_TAG/$DATASET/seed_$SEED/answers.json"
EXPECTED="$("$PY" - "$SAMPLE" "$LIMIT" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
limit = int(sys.argv[2])
print(min(len(rows), limit) if limit else len(rows))
PY
)"

complete_output() {
  "$PY" - "$OUT/manifest.json" "$OUT/final_answers.jsonl" "$EXPECTED" \
    "$MODEL_TAG" "$DATASET" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, output_path = map(Path, sys.argv[1:3])
expected, model, dataset, seed = int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
if not manifest_path.is_file() or not output_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
rows = [
    json.loads(line)
    for line in output_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
count = len(rows)
ok = (
    manifest.get("method") == "answer_convergence"
    and manifest.get("protocol_id") == "puma-fullcot-32k-v2"
    and manifest.get("model_tag") == model
    and manifest.get("dataset") == dataset
    and manifest.get("seed") == seed
    and manifest.get("threshold") == 10
    and manifest.get("fullcot_generation_tokens") == 32768
    and manifest.get("prompt_reserve_tokens") == 3072
    and manifest.get("truncated_answer_fix_tokens") == 2048
    and manifest.get("max_model_len") == 37888
    and manifest.get("expected_records") == expected
    and count == expected
    and all(row.get("protocol_id") == "puma-fullcot-32k-v2" for row in rows)
)
raise SystemExit(0 if ok else 1)
PY
}

if complete_output; then
  echo "[answer-convergence-cell] skip complete $MODEL_TAG $DATASET seed=$SEED n=$EXPECTED"
  exit 0
fi

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

echo "[answer-convergence-cell] preflight $MODEL_TAG $DATASET seed=$SEED gpu=$GPU"
"$PY" - <<'PY'
import nltk
import vllm
nltk.sent_tokenize("Preflight sentence.")
print(f"vLLM={vllm.__version__}; NLTK punkt=ok")
PY

args=(
  "$ROOT/scripts/run_answer_convergence_cell.py"
  --model "$MODEL"
  --model-tag "$MODEL_TAG"
  --dataset "$DATASET"
  --seed "$SEED"
  --sample "$SAMPLE"
  --output-dir "$OUT"
  --threshold 10
  --probe-max-tokens 100
  --max-model-len 37888
)
if [[ "$LIMIT" -gt 0 ]]; then
  args+=(--limit "$LIMIT")
fi

echo "[answer-convergence-cell] start $(date -Is)"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "${args[@]}"

if ! complete_output; then
  echo "ERROR: incomplete Answer Convergence cell n=$EXPECTED" >&2
  exit 1
fi
echo "[answer-convergence-cell] done $(date -Is) n=$EXPECTED"
