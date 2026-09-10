#!/usr/bin/env python3
"""Export compact Olympiad 7B trial jobs for 4B thinking HC."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_first_lock_room as room  # noqa: E402

NONE = "__NONE__"
OUT = AE / "results/hc_think/olympiadbench_s42/jobs.jsonl"


def candidates(history: list[dict], current: str) -> list[str]:
    seen: list[str] = []
    for row in history:
        ans = str(row.get("final_answer") or "")
        if ans and not any(rg.same(ans, old) for old in seen):
            seen.append(ans)
    if current and not any(rg.same(current, old) for old in seen):
        seen.append(current)
    if len(seen) > 3:
        others = [x for x in seen if not rg.same(x, current)][:2]
        seen = others + [current]
    return seen + [NONE]


def main() -> None:
    trials_path = room.dense_trial_path("r1_7b", "olympiadbench", 42)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        grouped[int(row["question_idx"])].append(row)
    jobs = []
    for qi, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r["stopped_len"]))
        question = str(rows[0].get("question") or "")
        if not question:
            continue
        for index, row in enumerate(rows):
            ans = str(row.get("final_answer") or "")
            jobs.append(
                {
                    "question_idx": qi,
                    "decision_step": int(row["stopped_len"]),
                    "trial_answer": ans,
                    "geo_conf": row.get("confidence"),
                    "question": question,
                    "cands": candidates(rows[: index + 1], ans),
                }
            )
    jobs.sort(key=lambda x: (x["question_idx"], x["decision_step"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} n={len(jobs)} q={len(grouped)}", flush=True)


if __name__ == "__main__":
    main()
