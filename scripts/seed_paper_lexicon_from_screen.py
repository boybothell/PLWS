#!/usr/bin/env python3
"""Copy finished 7B screen rows into official firstwin score dirs."""
from __future__ import annotations

import json
from pathlib import Path

from plws.paths import PLWSPaths
from plws.protocol import (
    MAX_MODEL_LEN,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "results" / "experiments" / "lexicon_ablation" / "screen"
MODEL = "r1_7b"
SEED = 42
DATASETS = ("math-500", "gpqa-diamond")
# screen config name -> lexicon directory
CONFIGS = (
    "wait",
    "alternatively",
    "hmm",
    "but",
    "so",
    "therefore",
    "core_no_wait",
    "core_no_alternatively",
    "core_no_hmm",
)


def main() -> None:
    paths = PLWSPaths(ROOT)
    jobs_by_ds = {}
    for dataset in DATASETS:
        path = paths.jobs_path(MODEL, dataset, SEED, "firstwin", k=4, lexicon="core")
        jobs_by_ds[dataset] = {
            json.loads(line)["uid"]
            for line in path.read_text().splitlines()
            if line.strip()
        }
    for config in CONFIGS:
        src = SCREEN / config / "scores.jsonl"
        if not src.is_file():
            print(f"skip missing {src}")
            continue
        rows = [
            json.loads(line)
            for line in src.read_text().splitlines()
            if line.strip()
        ]
        kept = 0
        for dataset in DATASETS:
            allowed = jobs_by_ds[dataset]
            chosen = [
                row
                for row in rows
                if row.get("uid") in allowed
                and row.get("status") in {"ok", "too_long"}
                and row.get("protocol_id") == PROTOCOL_ID
                and row.get("max_model_len") == MAX_MODEL_LEN
                and int(row.get("truncated_answer_fix_tokens") or 0)
                == TRUNCATED_ANSWER_FIX_TOKENS
            ]
            dest = (
                paths.score_dir(
                    MODEL, dataset, SEED, "firstwin", k=4, lexicon=config
                )
                / "scores.jsonl"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in chosen
                )
            )
            kept += len(chosen)
            print(f"{config} {dataset} seeded {len(chosen)} -> {dest}")
        print(f"{config} total seeded {kept} / screen {len(rows)}")


if __name__ == "__main__":
    main()
