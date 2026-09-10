#!/usr/bin/env python3
"""剩窗题：导出写完终答处的前缀，用来抽终答隐状态。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room

WAIT = AE / "results/leftover_waithelp"
OUT = AE / "results/leftover_final_h"
MODELS = ("r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b", "qwen3_8b")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        src = WAIT / f"{model}.jsonl"
        if not src.is_file():
            continue
        jobs = load_jsonl(src)
        by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in jobs:
            by[(row["dataset"], int(row["seed"]))].append(row)
        out_rows: list[dict[str, Any]] = []
        for (dataset, seed), rows in sorted(by.items()):
            trials_path = room.dense_trial_path(model, dataset, seed)
            if not trials_path.is_file():
                continue
            gp = low.gpath(model, dataset, seed)
            gmap = {int(r["question_idx"]): r for r in json.loads(gp.read_text())} if gp.is_file() else {}
            trials: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in json.loads(trials_path.read_text()):
                trials[int(row["question_idx"])].append(row)
            for row in rows:
                qi = int(row["question_idx"])
                xs = trials.get(qi)
                if not xs:
                    continue
                last = max(xs, key=lambda x: int(x["stopped_len"]))
                a_final = (gmap.get(qi) or {}).get("A_final") or last.get("final_answer")
                if not a_final:
                    continue
                prefix = str(last.get("reasoning_prefix") or row.get("reasoning_prefix") or "")
                out_rows.append(
                    {
                        "uid": row["uid"],
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "question_idx": qi,
                        "question": row.get("question") or last.get("question") or "",
                        "reasoning_prefix": prefix,
                        "answer": a_final,
                        "leftover_answer": row.get("answer"),
                        "committed": bool(rg.same(row.get("answer"), a_final)),
                    }
                )
        path = OUT / f"{model}.jsonl"
        with path.open("w") as handle:
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path} n={len(out_rows)}", flush=True)


if __name__ == "__main__":
    main()
