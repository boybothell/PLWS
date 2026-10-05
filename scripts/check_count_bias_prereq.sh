#!/usr/bin/env bash
# Preflight for count-bias / 随频次: every cell must have a non-empty
# firstwin.jsonl. Does not touch GPUs.
#
#   MODELS=nemotron_8b,qwen3_4b,qwen3_8b \
#     DATASETS=amc23,aime25,gpqa-diamond,math-500,olympiadbench \
#     SEEDS=42,0,1 RHOS=0.98 \
#     bash scripts/check_count_bias_prereq.sh
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODELS="${MODELS:?set MODELS}"
SEEDS="${SEEDS:-42,0,1}"
DATASETS="${DATASETS:-amc23,aime25,gpqa-diamond,math-500,olympiadbench}"
RHOS="${RHOS:-${RHO:-0.98}}"
export PLWS_ROOT="$ROOT"

exec "$ROOT/.venv/bin/python" - "$ROOT" "$MODELS" "$SEEDS" "$DATASETS" "$RHOS" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
MODELS = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
SEEDS = [int(x) for x in sys.argv[3].split(",") if x.strip()]
DATASETS = [x.strip() for x in sys.argv[4].split(",") if x.strip()]
RHOS = [x.strip() for x in sys.argv[5].split(",") if x.strip()]


def rho_tag(rho: str) -> str:
    return "rho_" + rho.replace(".", "p")


def jobs_path(model: str, dataset: str, seed: int) -> Path | None:
    relative = Path(model) / dataset / f"seed_{seed}" / "jobs/firstwin.jsonl"
    candidates = (
        ROOT / "tmp/incoming_main" / relative,
        ROOT
        / "tmp/incoming_main/results/runs/plws/window_first/k_4/lexicon_core"
        / relative,
        ROOT
        / "results/runs/plws/window_first/k_4/lexicon_core"
        / relative,
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def finished_count(model: str, dataset: str, seed: int, rho: str) -> int:
    folder = (
        ROOT
        / "results/runs/plws/count_bias"
        / rho_tag(rho)
        / model
        / dataset
        / f"seed_{seed}"
    )
    if not folder.is_dir():
        return 0
    target = float(rho)
    found: set[str] = set()
    for path in sorted(folder.glob("shard_*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("bias_schedule") != "count":
                continue
            if row.get("bias_formula") != "one_minus_rho_pow_n":
                continue
            if abs(float(row.get("bias_rho") or -1) - target) > 1e-9:
                continue
            if row.get("status") in {"ok", "too_long"} and row.get("uid"):
                found.add(str(row["uid"]))
    return len(found)


missing: list[str] = []
rows: list[dict] = []
for rho in RHOS:
    value = float(rho)
    if not 0 < value < 1:
        raise SystemExit(f"rho must be in (0, 1), got {rho}")
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                path = jobs_path(model, dataset, seed)
                if path is None:
                    missing.append(f"{model} {dataset} seed={seed}")
                    continue
                n_jobs = sum(1 for line in path.open() if line.strip())
                n_done = finished_count(model, dataset, seed, rho)
                rows.append(
                    {
                        "rho": rho,
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "jobs": n_jobs,
                        "done": n_done,
                        "pending": max(0, n_jobs - n_done),
                        "jobs_path": str(path),
                    }
                )

if missing:
    print(f"MISSING_JOBS {len(missing)}", flush=True)
    for item in missing:
        print(f"  {item}", flush=True)
    raise SystemExit(1)

pending_cells = sum(1 for row in rows if row["pending"] > 0)
done_cells = sum(1 for row in rows if row["pending"] == 0)
pending_jobs = sum(row["pending"] for row in rows)
print(
    f"ok cells={len(rows)} done_cells={done_cells} "
    f"pending_cells={pending_cells} pending_jobs={pending_jobs}",
    flush=True,
)
for row in rows:
    if row["pending"] == 0:
        continue
    print(
        f"  pending {rho_tag(row['rho'])} {row['model']} {row['dataset']} "
        f"seed={row['seed']} {row['done']}/{row['jobs']}",
        flush=True,
    )
PY
