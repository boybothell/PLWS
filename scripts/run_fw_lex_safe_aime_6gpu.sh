#!/usr/bin/env bash
# 6 卡只跑 8B / 14B 第一扇窗 AIME 扩词表。8B 用 0–1，14B 用 2–5。不动 7。
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
ROOT="${AE}/results/fw_lex_safe"
LOG="${ROOT}/logs"
mkdir -p "$LOG"

pack_aime() {
  local tag="$1"
  local out="${ROOT}/${tag}_s42/jobs_aime.jsonl"
  mkdir -p "${ROOT}/${tag}_s42"
  python3 - "$AE" "$tag" "$out" <<'PY'
import json, sys
from pathlib import Path
ae, tag, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
root = ae / "results/leftover_jump" / f"{tag}_s42"
n = empty = 0
with out.open("w") as handle:
    for name, kind in (("jobs.jsonl", "low"), ("jobs_high.jsonl", "high"), ("jobs_mix.jsonl", "mix")):
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            job = json.loads(line)
            if job.get("dataset") not in ("aime24", "aime25"):
                continue
            job["kind"] = job.get("kind") or kind
            if not str(job.get("thought") or "").strip():
                empty += 1
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
            n += 1
print(f"jobs {tag} aime {n} empty={empty} -> {out}")
PY
}

run_shard() {
  local tag="$1"
  local gpu="$2"
  local shard="$3"
  local nshard="$4"
  local jobs="${ROOT}/${tag}_s42/jobs_aime.jsonl"
  echo "launch ${tag} aime-lex-safe gpu=${gpu} shard=${shard}/${nshard}"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_LENS_DISABLE=1 "$PY" scripts/score_leftover_suppress.py \
    --mode suppress \
    --model-tag "$tag" \
    --lexicon safe \
    --seed 42 \
    --shard-id "$shard" \
    --num-shards "$nshard" \
    --max-context 32768 \
    --think-tokens 0 \
    --batch-size 0 \
    --jobs "$jobs" \
    --out-root "$ROOT" \
    > "${LOG}/${tag}_aime_s${shard}.log" 2>&1
  echo "done ${tag} shard ${shard}"
}

pack_aime nemotron_8b
pack_aime r1_14b

run_shard nemotron_8b 0 0 2 > "${LOG}/gpu0_aime.log" 2>&1 &
echo "gpu0 8B shard0 pid=$!"
run_shard nemotron_8b 1 1 2 > "${LOG}/gpu1_aime.log" 2>&1 &
echo "gpu1 8B shard1 pid=$!"
run_shard r1_14b 2 0 4 > "${LOG}/gpu2_aime.log" 2>&1 &
echo "gpu2 14B shard0 pid=$!"
run_shard r1_14b 3 1 4 > "${LOG}/gpu3_aime.log" 2>&1 &
echo "gpu3 14B shard1 pid=$!"
run_shard r1_14b 4 2 4 > "${LOG}/gpu4_aime.log" 2>&1 &
echo "gpu4 14B shard2 pid=$!"
run_shard r1_14b 5 3 4 > "${LOG}/gpu5_aime.log" 2>&1 &
echo "gpu5 14B shard3 pid=$!"
wait
echo "8B/14B AIME lex-safe done"
