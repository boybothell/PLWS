#!/usr/bin/env python3
"""7B 全量题：从空前缀开写，对照官方 Full-CoT。不是剩窗续写。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd  # noqa: E402

DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
OUT = AE / "results/full_suppress_safe/r1_7b_s42/jobs.jsonl"


def build_jobs(model: str = "r1_7b", seed: int = 42) -> list[dict]:
    jobs: list[dict] = []
    for dataset in DATASETS:
        path = dd.puma_stat_path(model, dataset, seed)
        if not path.is_file():
            print(f"skip missing {path}", flush=True)
            continue
        for i, row in enumerate(dd.load_json(path), start=1):
            qi = int(row["question_idx"]) if row.get("question_idx") is not None else i
            question = str(row.get("question") or "")
            gt = row.get("ground_truth")
            if not question or gt is None:
                continue
            jobs.append(
                {
                    "uid": f"{model}:{dataset}:{seed}:{qi}",
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "question_idx": qi,
                    "left_step": 0,
                    "kind": "full",
                    "confidence": None,
                    "left_ok": False,
                    "wait_helps": False,
                    "will_change": False,
                    "never_high": True,
                    "same_as_high": False,
                    "host_ok": bool(row.get("original_correct")),
                    "old_answer": str(row.get("original_answer") or ""),
                    "question": question,
                    "thought": "",
                    "gt": gt,
                    "original": row.get("original_answer"),
                    "orig_ok": bool(row.get("original_correct")),
                    "original_tokens": row.get("original_tokens"),
                }
            )
    jobs.sort(key=lambda x: (x["dataset"], x["question_idx"]))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    jobs = build_jobs(args.model_tag, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(f"wrote {len(jobs)} -> {args.out} {dict(Counter(j['dataset'] for j in jobs))}", flush=True)


if __name__ == "__main__":
    main()
