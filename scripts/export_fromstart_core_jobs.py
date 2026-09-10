#!/usr/bin/env python3
"""从空前缀压核三词的 jobs。官方齐的模型×数据集×种子全出。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
import export_full_suppress_jobs as full  # noqa: E402

MODELS = ("r1_7b", "nemotron_8b", "r1_14b")
SEEDS = (42, 0, 1, 123)
ALL = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
ROOT = AE / "results/fromstart_core"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    models = (args.model_tag,) if args.model_tag else MODELS
    seeds = (args.seed,) if args.seed is not None else SEEDS
    for tag in models:
        for seed in seeds:
            jobs = [
                job
                for job in full.build_jobs(tag, seed)
                if job["dataset"] in ALL
            ]
            out = ROOT / f"{tag}_s{seed}" / "jobs.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as handle:
                for job in jobs:
                    job["thought"] = ""
                    handle.write(json.dumps(job, ensure_ascii=False) + "\n")
            print(f"{tag} s{seed} {len(jobs)} {dict(Counter(j['dataset'] for j in jobs))} -> {out}", flush=True)


if __name__ == "__main__":
    main()
