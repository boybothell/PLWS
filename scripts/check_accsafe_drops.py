#!/usr/bin/env python3
"""Did Acc-safe ceiling hide drops? Score the same stops three ways."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
from report_accsafe_stop_ceiling import first_lock, DS, MODELS


def eval_cell(model: str, dataset: str, seed: int) -> dict[str, int] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dd.trial_path(model, dataset, seed)
    if dataset in ("aime24", "aime25"):
        gpath = AE / f"results/dense_G_{model}/{dataset}/seed_{seed}/per_sample.json"
    else:
        gpath = AE / f"results/dense_G_{model}/{dataset}/per_sample.json"
    if not puma_path.is_file() or not trials_path.is_file() or not gpath.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gpath)}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    z = defaultdict(int)
    for qi, info in official.items():
        trials = by.get(qi)
        gg = gmap.get(qi)
        if not trials or not gg:
            continue
        original_answer = info.get("original_answer")
        a_final = gg.get("A_final") or original_answer
        gt = info.get("ground_truth")
        orig_ok = bool(info.get("original_correct"))
        rows = rg.usable_rows(trials)
        z["n"] += 1
        z["orig_ok"] += int(orig_ok)
        z["orig_same_gt"] += int(rg.same(original_answer, gt))
        z["af_same_gt"] += int(rg.same(a_final, gt))
        z["af_ne_orig"] += int(not rg.same(a_final, original_answer))
        z["grader_disagree"] += int(orig_ok != rg.same(original_answer, gt))
        kw = dict(original_answer=original_answer, a_final=a_final, gt=gt, orig_ok=orig_ok)
        end_safe = first_lock(rows, mode="safe", **kw)
        end_raw = None
        for end, row in enumerate(rows):
            if not rg.window_ok(rows, end) or int(row["stopped_len"]) < rg.MSS:
                continue
            ans = row.get("final_answer")
            if rg.same(ans, original_answer) or rg.same(ans, a_final) or rg.same(ans, gt):
                end_raw = end
                break
        if end_raw is None:
            z["raw_ok"] += int(orig_ok)
            z["safe_ok"] += int(orig_ok)
            z["raw_same"] += int(rg.same(original_answer, gt))
            continue
        ans = rows[end_raw].get("final_answer")
        raw_same = bool(rg.same(ans, gt))
        z["raw_stop"] += 1
        z["raw_same"] += int(raw_same)
        z["raw_ok"] += int(raw_same)
        if orig_ok and not raw_same:
            z["raw_drop"] += 1
            if rg.same(ans, a_final) and not rg.same(ans, original_answer):
                z["drop_af_only"] += 1
            elif rg.same(ans, original_answer):
                z["drop_orig_str"] += 1
        if end_safe is None:
            z["safe_skip_to_avoid_drop"] += int(end_raw is not None and orig_ok and not raw_same)
            z["safe_ok"] += int(orig_ok)
            continue
        sans = rows[end_safe].get("final_answer")
        safe_same = bool(rg.same(sans, gt))
        inherit = 1 if safe_same else (int(orig_ok) if rg.same(sans, original_answer) else 0)
        z["safe_stop"] += 1
        z["safe_ok"] += inherit
        z["safe_same"] += int(safe_same)
        z["inherit_hide"] += int(inherit == 1 and not safe_same and orig_ok)
    return dict(z)


def add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def main() -> None:
    rg.K = 4
    header = (
        "| 集 | n | 官方写完对 | same(写完,金标) | 不过滤锁：same(试答,金标) | "
        "不过滤会降 | 其中只像A_final | 其中像官方写完串 | 现行计分藏住 |"
    )
    print(header, flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            if dataset in ("aime24", "aime25"):
                rec = None
                for seed in dd.AIME_SEEDS:
                    one = eval_cell(model, dataset, seed)
                    if one is None:
                        rec = None
                        break
                    rec = one if rec is None else add(rec, one)
                if rec is None:
                    print(f"skip {zh} {ds_zh}", flush=True)
                    continue
            else:
                rec = eval_cell(model, dataset, 42)
                if rec is None:
                    print(f"skip {zh} {ds_zh}", flush=True)
                    continue
            n = rec["n"]
            g = rec.get
            row = (
                f"| {zh} {ds_zh} | {n} | {g('orig_ok', 0)} | {g('orig_same_gt', 0)} | "
                f"{g('raw_same', 0)} | {g('raw_drop', 0)} | {g('drop_af_only', 0)} | "
                f"{g('drop_orig_str', 0)} | {g('inherit_hide', 0)} |"
            )
            print(row, flush=True)


if __name__ == "__main__":
    main()
