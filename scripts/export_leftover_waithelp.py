#!/usr/bin/env python3
"""导出第一扇剩窗：标签是「再等会不会更好 Acc」，不是可停。"""
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
import report_cand_collapse as cc
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_full_no_fs as nofs
import report_k4_hyps as hy
import report_k4_second_lock as sl

OUT = AE / "results/leftover_waithelp"
MODELS = {
    "r1_7b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B",
    "nemotron_8b": "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1",
    "r1_14b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B",
    "qwen3_4b": "/mnt/d/lsj/models/Qwen3-4B",
    "qwen3_8b": "/mnt/d/lsj/models/Qwen3-8B",
}


def alts_of(rows: list[dict[str, Any]], end: int, ans: Any) -> list[str]:
    seen: list[str] = []
    for row in rows[: end + 1]:
        other = str(row.get("final_answer") or "")
        if not other or rg.same(other, ans):
            continue
        if not any(rg.same(other, x) for x in seen):
            seen.append(other)
        if len(seen) >= 8:
            break
    return seen


def run_len(rows: list[dict[str, Any]], end: int) -> int:
    ans = rows[end].get("final_answer")
    n = 0
    for row in reversed(rows[: end + 1]):
        if not rg.same(row.get("final_answer"), ans):
            break
        n += 1
    return n


def export_cell(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not trials_path.is_file():
        return []
    puma_path = dd.puma_stat_path(model, dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)} if regen_path.is_file() else {}
    )
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    if official:
        qis = sorted(official)
    elif gmap:
        qis = sorted(set(gmap) & set(by))
    else:
        qis = sorted(by)
    jobs = []
    for qi in qis:
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        info = official.get(qi) or {}
        g = gmap.get(qi) or {}
        last = max(trials, key=lambda x: int(x["stopped_len"]))
        gt = info.get("ground_truth") or g.get("ground_truth")
        original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
        a_final = g.get("A_final") or original
        orig_ok = bool(info.get("original_correct")) if info else bool(
            low.credit(original, gt, original, True)
        )
        orig_tok = int(info.get("original_tokens") or last.get("count_reasoning_tokens") or 0)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        host = regen.get(qi)
        if sim["branch"] == "consec":
            if host:
                nofs_ok = bool(host.get("compressed_correct"))
            else:
                nofs_ok = bool(low.credit(sim["answer"], gt, original, orig_ok))
            nofs_branch = "consec"
        else:
            nofs_ok = orig_ok
            nofs_branch = "full"
        wins = sl.same_windows(rows)
        left = next((w for w in wins if hy.leftover(w)), None)
        if left is None:
            continue
        high = next((w for w in wins if w["kind"] == "high" and w["step"] > left["step"]), None)
        row = rows[left["end"]]
        leftover_ok = bool(low.credit(left["ans"], gt, original, orig_ok))
        wait_helps = bool(nofs_ok and not leftover_ok)
        jobs.append(
            {
                "uid": f"{dataset}:{seed}:{qi}",
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "question_idx": qi,
                "decision_step": int(left["step"]),
                "question": info.get("question") or row.get("question") or "",
                "reasoning_prefix": str(row.get("reasoning_prefix") or ""),
                "answer": left["ans"],
                "confidence": left["c"],
                "kind": left["kind"],
                "alts": alts_of(rows, left["end"], left["ans"]),
                "run_len": run_len(rows, left["end"]),
                "n_distinct": len(alts_of(rows, left["end"], left["ans"])) + 1,
                "stoppable": cc.stoppable(left["ans"], gt, original, a_final),
                "leftover_ok": leftover_ok,
                "nofs_ok": nofs_ok,
                "nofs_branch": nofs_branch,
                "wait_helps": wait_helps,
                "high_later": high is not None,
                "same_as_high": bool(high is not None and rg.same(left["ans"], high["ans"])),
                "high_step": int(high["step"]) if high else None,
                "orig_ok": orig_ok,
            }
        )
    return jobs


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    OUT.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in MODELS:
        for ds_zh, dataset in nofs.DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            n = 0
            for seed in seeds:
                jobs = export_cell(model, dataset, seed)
                n += len(jobs)
                by_model[model].extend(jobs)
            if n:
                helps = sum(1 for j in by_model[model] if j["dataset"] == dataset and j["wait_helps"])
                print(f"{model} {dataset} leftover={n} wait_helps={helps}", flush=True)
    for model, jobs in by_model.items():
        if not jobs:
            continue
        path = OUT / f"{model}.jsonl"
        with path.open("w") as handle:
            for row in jobs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = len(jobs)
        helps = sum(1 for j in jobs if j["wait_helps"])
        safe = n - helps
        print(f"wrote {path} n={n} safe={safe} wait_helps={helps}", flush=True)


if __name__ == "__main__":
    main()
