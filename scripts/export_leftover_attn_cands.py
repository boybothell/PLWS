#!/usr/bin/env python3
"""Export 7B first leftover 4-same windows for question-attn / Wait scoring."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_cand_collapse as cc
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_hyps as hy
import report_k4_second_lock as sl

OUT = AE / "results/leftover_attn"
WAIT_RE = re.compile(r"\bWait\b")
JOBS = (
    ("r1_7b", "math-500", 42),
    ("r1_7b", "gpqa-diamond", 42),
)


def export_one(model: str, dataset: str, seed: int) -> Path:
    official = {int(r["question_idx"]): r for r in dd.load_json(dd.puma_stat_path(model, dataset, seed))}
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen = {int(r["question_idx"]): r for r in dd.load_json(regen_path)} if regen_path.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(room.dense_trial_path(model, dataset, seed)):
        by[int(row["question_idx"])].append(row)
    jobs = []
    n_wait = 0
    n_pos = 0
    for qi, info in sorted(official.items()):
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        gt = info.get("ground_truth")
        original = info.get("original_answer")
        a_final = (gmap.get(qi) or {}).get("A_final") or original
        orig_tok = int(info.get("original_tokens") or 0)
        host = regen.get(qi)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        host_step = int(host.get("stopped_len") or sim["step"]) if host else int(sim["step"])
        wins = sl.same_windows(rows)
        left = next((w for w in wins if hy.leftover(w)), None)
        if left is None:
            continue
        row = rows[left["end"]]
        prefix = str(row.get("reasoning_prefix") or "")
        pos = cc.stoppable(left["ans"], gt, original, a_final)
        n_pos += int(pos)
        n_w = len(WAIT_RE.findall(prefix))
        n_wait += int(n_w > 0)
        jobs.append(
            {
                "question_idx": qi,
                "decision_step": int(left["step"]),
                "question": info.get("question") or row.get("question") or "",
                "reasoning_prefix": prefix,
                "answer": left["ans"],
                "confidence": left["c"],
                "kind": left["kind"],
                "pos": pos,
                "n_wait": n_w,
                "host_step": host_step,
                "before_host": int(left["step"]) < host_step,
            }
        )
    out = OUT / f"{model}_{dataset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in jobs))
    print(
        f"{model} {dataset} leftover={len(jobs)} pos={n_pos} with_wait={n_wait} -> {out}",
        flush=True,
    )
    return out


def main() -> None:
    for model, dataset, seed in JOBS:
        export_one(model, dataset, seed)


if __name__ == "__main__":
    main()
