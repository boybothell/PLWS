#!/usr/bin/env bash
set -euo pipefail
AE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AE"
PY=/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python
mkdir -p results/first_hm_gate/jobs results/first_hm_gate/logs
for seed in 42 0 1 123; do
  jobs="results/first_hm_gate/jobs/r1_7b_s${seed}.jsonl"
  if [[ ! -s "$jobs" ]]; then
    "$PY" scripts/export_first_hm_jobs.py --model-tag r1_7b --seed "$seed" --out "$jobs"
  else
    echo "jobs exist s${seed}"
  fi
done
"$PY" scripts/seed_first_hm_scores.py --model-tag r1_7b --seeds 42,0,1,123
"$PY" - <<'PY'
import json
from pathlib import Path
from collections import Counter
AE=Path("results/first_hm_gate")
def have(folder):
    s=set()
    p=Path(folder)
    if not p.is_dir(): return s
    for f in p.glob("scores_shard*.jsonl"):
        for line in f.open():
            if not line.strip(): continue
            try: r=json.loads(line)
            except: continue
            if r.get("status") in ("ok","too_long") and r.get("uid"):
                s.add(str(r["uid"]))
    return s
for seed in (42,0,1,123):
    jobs=[json.loads(l) for l in open(AE/f"jobs/r1_7b_s{seed}.jsonl") if l.strip()]
    sc=have(AE/f"r1_7b_s{seed}_suppress_hm")
    pend=[j for j in jobs if j["uid"] not in sc]
    print(f"s{seed} jobs={len(jobs)} delayed={sum(1 for j in jobs if j.get('delayed'))} reused={sum(1 for j in jobs if not j.get('delayed'))} scored={len(sc)} pending={len(pend)} {dict(Counter(j['dataset'] for j in pend))}")
PY
