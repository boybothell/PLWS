#!/usr/bin/env python3
"""Export 密探k4 final_candidates for cells that still need rewrite."""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd  # noqa: E402
import report_first_lock_room as room  # noqa: E402

JOBS = (
    ("r1_14b", "math-500", 42),
    ("r1_14b", "olympiadbench", 42),
    ("r1_14b", "gpqa-diamond", 42),
    ("r1_32b", "math-500", 42),
    ("r1_32b", "olympiadbench", 42),
    ("r1_32b", "gpqa-diamond", 42),
    ("qwen3_30b_a3b", "math-500", 42),
    ("qwen3_30b_a3b", "gpqa-diamond", 42),
    ("qwen3_4b", "olympiadbench", 42),
    ("qwen3_4b", "gpqa-diamond", 42),
    ("qwen3_8b", "gpqa-diamond", 42),
)
for seed in dd.AIME_SEEDS:
    JOBS += (
        ("r1_14b", "aime24", seed),
        ("r1_14b", "aime25", seed),
        ("r1_32b", "aime24", seed),
        ("r1_32b", "aime25", seed),
        ("qwen3_30b_a3b", "aime24", seed),
        ("qwen3_30b_a3b", "aime25", seed),
        ("qwen3_8b", "aime24", seed),
    )
for seed in (42, 0, 1):
    JOBS += (("qwen3_4b", "aime24", seed),)
for seed in (0, 1, 123):
    JOBS += (("qwen3_8b", "aime25", seed),)


def main() -> None:
    for model, dataset, seed in JOBS:
        trials = room.dense_trial_path(model, dataset, seed)
        puma = dd.puma_stat_path(model, dataset, seed)
        if not trials.is_file() or not puma.is_file():
            print(f"skip {model} {dataset} s{seed} missing trials={trials.is_file()} puma={puma.is_file()}")
            continue
        try:
            meta = dd.export_candidates(model, dataset, seed)
        except Exception as exc:
            print(f"fail {model} {dataset} s{seed}: {exc}")
            continue
        if meta.get("missing"):
            print(f"skip {model} {dataset} s{seed} export missing")


if __name__ == "__main__":
    main()
