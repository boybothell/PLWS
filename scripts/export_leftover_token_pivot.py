#!/usr/bin/env python3
"""导出第一扇剩窗后第一步（及改口当步）的思路正文，供逐步 next-token 熵。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_hyps as hy
import report_k4_second_lock as sl
import report_leftover_after as after

OUT = AE / "results/leftover_token_pivot"


def prefix_of(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("reasoning_prefix") or "")


def export_cell(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    xs = after.load_left(model, dataset, seed)
    if not xs:
        return []
    trials_path = room.dense_trial_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    qis = sorted(official) if official else sorted(set(gmap) & set(by) if gmap else by)
    left_by = {x["question_idx"]: x for x in xs}
    jobs = []
    for qi in qis:
        rec = left_by.get(qi)
        if rec is None:
            continue
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        wins = sl.same_windows(rows)
        left = next((w for w in wins if hy.leftover(w)), None)
        if left is None:
            continue
        after_rows = rows[left["end"] + 1 :]
        if not after_rows:
            continue
        nxt = after_rows[0]
        brk = next(
            (row for row in after_rows if not rg.same(row.get("final_answer"), left["ans"])),
            None,
        )
        left_row = rows[left["end"]]
        leftover_prefix = prefix_of(left_row)
        next_prefix = prefix_of(nxt)
        break_prefix = prefix_of(brk)
        if not leftover_prefix or not next_prefix:
            continue
        if not next_prefix.startswith(leftover_prefix[:80]):
            continue
        info = official.get(qi) or {}
        jobs.append(
            {
                "uid": f"{dataset}:{seed}:{qi}",
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "question_idx": qi,
                "question": info.get("question") or left_row.get("question") or "",
                "leftover_prefix": leftover_prefix,
                "next_prefix": next_prefix,
                "break_prefix": break_prefix if brk is not None else "",
                "left_step": int(left["step"]),
                "next_step": int(nxt["stopped_len"]),
                "break_step": int(brk["stopped_len"]) if brk is not None else None,
                "answer": left["ans"],
                "next_answer": nxt.get("final_answer"),
                "next_same": bool(rg.same(nxt.get("final_answer"), left["ans"])),
                "kind": left["kind"],
                "confidence": rec.get("left_c", left.get("c")),
                "left_ok": rec["left_ok"],
                "wait_helps": rec["wait_helps"],
                "will_change": rec["will_change"],
                "same_as_high": rec["same_as_high"],
                "never_high": rec["never_high"],
                "persist": rec["persist"],
                "gap_diff": rec["gap_diff"],
            }
        )
    return jobs


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    OUT.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _zh, model in after.MODELS:
        for _ds_zh, dataset in after.DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                part = export_cell(model, dataset, seed)
                by_model[model].extend(part)
                print(f"{model} {dataset} s{seed} n={len(part)}", flush=True)
    for model, jobs in by_model.items():
        path = OUT / f"{model}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in jobs))
        print(f"wrote {path} n={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
