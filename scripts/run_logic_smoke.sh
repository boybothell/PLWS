#!/usr/bin/env bash
# Isolated 2-question logic smoke for PUMA / DEER / AC / Dynasor.
# Writes only under results/_smoke/. Never touches official cells.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck source=lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"
plws_runtime_init
export PLWS_ROOT="$ROOT"
export PLWS_DEPLOY_PROFILE="${PLWS_DEPLOY_PROFILE:-local_48g}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SMOKE_ROOT="${SMOKE_ROOT:-$ROOT/results/_smoke/logic_2q}"
LIMIT="${LIMIT:-2}"
SEED="${SEED:-42}"
DATASET="${DATASET:-aime24}"
MODELS="${MODELS:-r1_1p5b qwen3_4b}"
WAIT_GPUS="${WAIT_GPUS:-3,7}"
GPU="${GPU:-}"
LOG="$SMOKE_ROOT/run.log"

mkdir -p "$SMOKE_ROOT"
exec > >(tee -a "$LOG") 2>&1

prepare_host_sample() {
  local model_tag="$1"
  "$PY" - "$ROOT" "$SMOKE_ROOT" "$model_tag" "$DATASET" "$SEED" "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

root, smoke, model_tag, dataset, seed, limit = sys.argv[1:7]
limit = int(limit)
src = Path(root) / "samples" / model_tag / dataset / f"seed_{seed}"
answers = json.loads((src / "answers.json").read_text(encoding="utf-8"))
rows = answers[:limit]
if len(rows) < limit:
    raise SystemExit(f"{src}: only {len(rows)} rows, need {limit}")
dst = Path(smoke) / "host_samples" / model_tag / dataset / f"seed_{seed}"
dst.mkdir(parents=True, exist_ok=True)
(dst / "answers.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
meta = {
    "protocol_id": "puma-fullcot-32k-v2",
    "tag": f"smoke_{model_tag}_{dataset}",
    "path": f"{model_tag}/{dataset}/seed_{seed}",
    "layout": "samples/<model_tag>/<dataset>/seed_<seed>",
    "dataset": dataset,
    "benchmark": str(Path(smoke) / "bench" / f"{dataset}_{limit}q.jsonl"),
    "model_tag": model_tag,
    "seed": int(seed),
    "max_tokens": 32768,
    "answer_fix_max_tokens": 2048,
    "prompt_reserve_tokens": 3072,
    "max_model_len": 37888,
    "prompt_version": "default",
    "n_questions": limit,
    "smoke": True,
}
(dst / "sample_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(dst)
PY
}

prepare_bench() {
  "$PY" - "$PLWS_DATA_ROOT" "$SMOKE_ROOT" "$DATASET" "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

data_root, smoke, dataset, limit = sys.argv[1:5]
limit = int(limit)
src = Path(data_root) / f"{dataset}_test.jsonl"
out = Path(smoke) / "bench" / f"{dataset}_{limit}q.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) < limit:
    raise SystemExit(f"{src}: only {len(rows)} rows, need {limit}")
out.write_text("\n".join(rows[:limit]) + "\n", encoding="utf-8")
print(out)
PY
}

gpu_free() {
  local idx="$1"
  local used pids
  used="$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  pids="$(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -E '^[0-9]+$' || true)"
  [[ -z "$pids" && "${used:-99999}" -lt 2000 ]]
}

wait_for_gpu() {
  if [[ -n "$GPU" ]]; then
    echo "[smoke] using preset GPU=$GPU"
    return 0
  fi
  echo "[smoke] waiting for an idle card among $WAIT_GPUS (used<2000MiB, no compute apps)"
  local IFS=','
  while true; do
    local g
    for g in $WAIT_GPUS; do
      if gpu_free "$g"; then
        GPU="$g"
        echo "[smoke] claimed GPU=$GPU"
        return 0
      fi
    done
    sleep 30
  done
}

check_cell() {
  local method="$1"
  local out="$2"
  "$PY" - "$method" "$out" "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

method, out, limit = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
errors = []
if method == "puma":
    prefixed = out / "prefixed_answers.json"
    if not prefixed.is_file():
        raise SystemExit(f"missing {prefixed}")
    rows = json.loads(prefixed.read_text(encoding="utf-8"))
    if len(rows) != limit:
        errors.append(f"prefixed n={len(rows)} expected={limit}")
    from plws.puma_budget import audit_rows
    audit = audit_rows(rows)
    print(json.dumps({"method": method, "n": len(rows), "audit": audit}, ensure_ascii=False))
elif method == "deer":
    rows = [
        json.loads(line)
        for line in (out / "deer.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if len(rows) != limit:
        errors.append(f"deer n={len(rows)} expected={limit}")
    if manifest.get("protocol_id") != "puma-fullcot-32k-v2":
        errors.append("deer protocol mismatch")
    print(json.dumps({
        "method": method,
        "n": len(rows),
        "family": manifest.get("deer", {}).get("family_policy"),
        "deployment": manifest.get("deployment"),
    }, ensure_ascii=False))
else:
    final = out / "final_answers.jsonl"
    rows = [
        json.loads(line)
        for line in final.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if len(rows) != limit:
        errors.append(f"{method} n={len(rows)} expected={limit}")
    if manifest.get("protocol_id") != "puma-fullcot-32k-v2":
        errors.append(f"{method} protocol mismatch")
    print(json.dumps({
        "method": method,
        "n": len(rows),
        "early": sum(1 for row in rows if row.get("stopped_early") or row.get("converged")),
        "deployment": manifest.get("deployment"),
    }, ensure_ascii=False))
if errors:
    raise SystemExit("; ".join(errors))
PY
}

run_model() {
  local model_tag="$1"
  local sample out_base
  sample="$(prepare_host_sample "$model_tag")"
  out_base="$SMOKE_ROOT/$model_tag/$DATASET/seed_$SEED"
  mkdir -p "$out_base"
  echo "[smoke] ===== $model_tag $DATASET seed=$SEED gpu=$GPU limit=$LIMIT ====="

  if check_cell puma "$out_base/puma"; then
    echo "[smoke] skip complete PUMA $model_tag"
  else
    echo "[smoke] PUMA $model_tag"
    MODEL="$(plws_model_path "$model_tag")" \
    MODEL_TAG="$model_tag" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    SAMPLE="$sample" \
    PUMA_DIR="$out_base/puma" \
    BENCH="$SMOKE_ROOT/bench/${DATASET}_${LIMIT}q.jsonl" \
    bash "$ROOT/scripts/run_puma_official.sh"
    check_cell puma "$out_base/puma"
  fi

  if check_cell deer "$out_base/deer"; then
    echo "[smoke] skip complete DEER $model_tag"
  else
    echo "[smoke] DEER $model_tag"
    MODEL_TAG="$model_tag" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    LIMIT="$LIMIT" OUT="$out_base/deer" \
    bash "$ROOT/scripts/run_deer_official.sh"
    check_cell deer "$out_base/deer"
  fi

  if check_cell answer_convergence "$out_base/answer_convergence"; then
    echo "[smoke] skip complete Answer Convergence $model_tag"
  else
    echo "[smoke] Answer Convergence $model_tag"
    MODEL_TAG="$model_tag" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    LIMIT="$LIMIT" SAMPLE="$sample/answers.json" \
    OUT="$out_base/answer_convergence" \
    bash "$ROOT/scripts/run_answer_convergence_cell.sh"
    check_cell answer_convergence "$out_base/answer_convergence"
  fi

  if check_cell dynasor "$out_base/dynasor"; then
    echo "[smoke] skip complete Dynasor $model_tag"
  else
    echo "[smoke] Dynasor $model_tag"
    MODEL_TAG="$model_tag" DATASET="$DATASET" SEED="$SEED" GPU="$GPU" \
    LIMIT="$LIMIT" SAMPLE="$sample/answers.json" \
    OUT="$out_base/dynasor" \
    bash "$ROOT/baselines/dynasor/run_cell.sh"
    check_cell dynasor "$out_base/dynasor"
  fi
}

echo "[smoke] start $(date -Is) models=$MODELS dataset=$DATASET limit=$LIMIT"
prepare_bench
wait_for_gpu
export CUDA_VISIBLE_DEVICES="$GPU"
plws_export_cuda_runtime

for model_tag in $MODELS; do
  run_model "$model_tag"
done

echo "[smoke] all methods passed $(date -Is) gpu=$GPU"
echo "[smoke] outputs $SMOKE_ROOT"
