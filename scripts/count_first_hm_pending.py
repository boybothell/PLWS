#!/usr/bin/env python3
"""第一扇 High/Mix 门控还剩多少题没分。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
from score_leftover_jump import done_uids  # noqa: E402

JOBS = AE / "results/first_hm_gate/jobs/r1_7b_s42.jsonl"
FOLDER = AE / "results/first_hm_gate/r1_7b_s42_suppress_hm"


def pending() -> int:
    if not JOBS.is_file():
        return -1
    uids = [json.loads(line)["uid"] for line in JOBS.open() if line.strip()]
    already: set[str] = set()
    if FOLDER.is_dir():
        for path in FOLDER.glob("scores_shard*.jsonl"):
            already |= done_uids(path)
    return sum(1 for uid in uids if uid not in already)


if __name__ == "__main__":
    print(pending())
