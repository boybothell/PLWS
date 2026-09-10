#!/usr/bin/env python3
"""四步同答窗：四步都过某条件，对/错能不能拉开。零 GPU。"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_second_lock as sl
import report_samewin_4b_all as sw

TABLE = AE / "tables/samewin_4step_cond.md"


def finite_list(xs: list[Any]) -> list[float]:
    return [sw.finite(x) for x in xs]


def all_ge(xs: list[float], thr: float) -> bool:
    return bool(xs) and all(x == x and x >= thr for x in xs)


def count_ge(xs: list[float], thr: float) -> int:
    return sum(1 for x in xs if x == x and x >= thr)


def load_windows() -> list[dict[str, Any]]:
    rg.K = 4
    rg.TAU = 0.995
    out: list[dict[str, Any]] = []
    for zh, dataset, seeds in sw.CELLS:
        if zh not in ("MATH", "奥赛", "GPQA"):
            continue
        for seed in seeds:
            scores = sw.load_var(dataset, seed)
            extra = sw.load_v1_extra(dataset) if seed == 42 else {}
            trials_path = room.dense_trial_path("r1_7b", dataset, seed)
            if not trials_path.is_file() or not scores:
                continue
            puma_path = dd.puma_stat_path("r1_7b", dataset, seed)
            official = (
                {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
                if puma_path.is_file()
                else {}
            )
            gp = low.gpath("r1_7b", dataset, seed)
            gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
            by: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in dd.load_json(trials_path):
                by[int(row["question_idx"])].append(row)
            n = 0
            for qi, trials in by.items():
                rows = rg.usable_rows(trials)
                if not rows:
                    continue
                info = official.get(qi) or {}
                g = gmap.get(qi) or {}
                last = max(trials, key=lambda x: int(x["stopped_len"]))
                gt = info.get("ground_truth") or g.get("ground_truth")
                original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
                orig_ok = (
                    bool(info.get("original_correct"))
                    if "original_correct" in info
                    else bool(low.credit(original, gt, original, True))
                )
                for win in sl.same_windows(rows):
                    window = rows[win["end"] + 1 - rg.K : win["end"] + 1]
                    steps = [int(x["stopped_len"]) for x in window]
                    geos = [sw.finite(x.get("confidence")) for x in window]
                    packs = []
                    for step in steps:
                        pack = dict(scores.get((qi, step)) or {})
                        pack.update(extra.get((qi, step)) or {})
                        packs.append(pack)
                    rec = {
                        "zh": zh,
                        "kind": win["kind"],
                        "gold_ok": bool(low.credit(win["ans"], gt, original, orig_ok)),
                        "geo": geos,
                        "B_final": [sw.finite(p.get("B_final")) for p in packs],
                        "EAGLE_yn": [sw.finite(p.get("EAGLE_yn")) for p in packs],
                        "SteerConf_mean": [sw.finite(p.get("SteerConf_mean")) for p in packs],
                        "SteerConf_min": [sw.finite(p.get("SteerConf_min")) for p in packs],
                        "verbal": [sw.finite(p.get("verbal")) for p in packs],
                    }
                    out.append(rec)
                    n += 1
            print(f"{dataset} s{seed} n={n}", flush=True)
    return out


def complete(xs: list[float]) -> bool:
    return len(xs) == 4 and all(x == x for x in xs)


def conds() -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    return [
        ("四步同答（无额外）", lambda w: True),
        ("末步把握≥0.995", lambda w: complete(w["geo"]) and w["geo"][-1] >= 0.995),
        ("四步把握都≥0.995", lambda w: all_ge(w["geo"], 0.995)),
        ("四步把握都≥0.90", lambda w: all_ge(w["geo"], 0.90)),
        ("四步把握都≥0.80", lambda w: all_ge(w["geo"], 0.80)),
        ("至少三步把握≥0.995", lambda w: complete(w["geo"]) and count_ge(w["geo"], 0.995) >= 3),
        ("把握不降（末≥首）", lambda w: complete(w["geo"]) and w["geo"][-1] >= w["geo"][0]),
        ("末步把握最高", lambda w: complete(w["geo"]) and w["geo"][-1] >= max(w["geo"]) - 1e-12),
        ("把握极差<0.02", lambda w: complete(w["geo"]) and max(w["geo"]) - min(w["geo"]) < 0.02),
        ("末步 Direct≥0.90", lambda w: w["B_final"][-1] == w["B_final"][-1] and w["B_final"][-1] >= 0.90),
        ("四步 Direct 都≥0.90", lambda w: all_ge(w["B_final"], 0.90)),
        ("四步 Direct 都≥0.50", lambda w: all_ge(w["B_final"], 0.50)),
        ("至少三步 Direct≥0.90", lambda w: complete(w["B_final"]) and count_ge(w["B_final"], 0.90) >= 3),
        ("至少两步 Direct≥0.90", lambda w: complete(w["B_final"]) and count_ge(w["B_final"], 0.90) >= 2),
        ("四步 Direct 同侧", lambda w: complete(w["B_final"]) and (all(x >= 0.5 for x in w["B_final"]) or all(x < 0.5 for x in w["B_final"]))),
        ("末步 EAGLE≥0.90", lambda w: w["EAGLE_yn"][-1] == w["EAGLE_yn"][-1] and w["EAGLE_yn"][-1] >= 0.90),
        ("四步 EAGLE 都≥0.90", lambda w: all_ge(w["EAGLE_yn"], 0.90)),
        ("至少三步 EAGLE≥0.90", lambda w: complete(w["EAGLE_yn"]) and count_ge(w["EAGLE_yn"], 0.90) >= 3),
        ("末步 Steer均值≥0.67", lambda w: w["SteerConf_mean"][-1] == w["SteerConf_mean"][-1] and w["SteerConf_mean"][-1] >= 0.67),
        ("四步 Steer均值都≥0.67", lambda w: all_ge(w["SteerConf_mean"], 0.67)),
        ("四步 Steer最小都≥0.50", lambda w: all_ge(w["SteerConf_min"], 0.50)),
        ("末步口头≥0.90", lambda w: w["verbal"][-1] == w["verbal"][-1] and w["verbal"][-1] >= 0.90),
        ("四步口头都≥0.90", lambda w: all_ge(w["verbal"], 0.90)),
        ("至少三步口头≥0.90", lambda w: complete(w["verbal"]) and count_ge(w["verbal"], 0.90) >= 3),
        ("高把握且四步Direct≥0.90", lambda w: w["kind"] == "high" and all_ge(w["B_final"], 0.90)),
        ("混合且四步把握≥0.90", lambda w: w["kind"] == "mix" and all_ge(w["geo"], 0.90)),
        ("低把握且四步Direct≥0.90", lambda w: w["kind"] == "low" and all_ge(w["B_final"], 0.90)),
        ("低把握且末步Direct≥0.90", lambda w: w["kind"] == "low" and w["B_final"][-1] == w["B_final"][-1] and w["B_final"][-1] >= 0.90),
        ("低把握且四步EAGLE≥0.90", lambda w: w["kind"] == "low" and all_ge(w["EAGLE_yn"], 0.90)),
        ("低把握且四步口头≥0.90", lambda w: w["kind"] == "low" and all_ge(w["verbal"], 0.90)),
    ]


def summarize(xs: list[dict[str, Any]], pred: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    hit = [x for x in xs if pred(x)]
    ok = sum(1 for x in hit if x["gold_ok"])
    bad = len(hit) - ok
    all_ok = sum(1 for x in xs if x["gold_ok"])
    prec = ok / len(hit) if hit else float("nan")
    rec = ok / all_ok if all_ok else float("nan")
    return {"n": len(hit), "ok": ok, "bad": bad, "prec": prec, "rec": rec}


def fmt(x: float) -> str:
    return "—" if x != x else f"{x:.2f}"


def pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.0f}%"


def main() -> None:
    rows = load_windows()
    base = summarize(rows, lambda _w: True)
    leftover = [x for x in rows if x["kind"] != "high"]
    low_only = [x for x in rows if x["kind"] == "low"]
    lines = [
        "# 四步同答窗：四步都过某条件，对/错能不能拉开",
        "",
        "7B MATH/奥赛/GPQA。对窗 = 试答对金标。",
        "「四步都」= 窗内每一步都有分且都过线；对照是只看最后一步。",
        f"底：四步同答 {base['ok']}/{base['bad']}，对窗占比 {pct(base['prec'])}。",
        "",
        "## 1. 全体四步同答窗",
        "",
        "| 条件 | 对/错 | 对窗占比 | 盖住对窗 | 比对无条件 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, pred in conds():
        s = summarize(rows, pred)
        lift = s["prec"] - base["prec"] if s["prec"] == s["prec"] else float("nan")
        lines.append(
            f"| {name} | {s['ok']}/{s['bad']} | {pct(s['prec'])} | {pct(s['rec'])} | {lift:+.0f}pp |"
            if lift == lift
            else f"| {name} | {s['ok']}/{s['bad']} | {pct(s['prec'])} | {pct(s['rec'])} | — |"
        )

    lines += [
        "",
        "## 2. 只在低把握窗上加条件",
        "",
        "| 条件 | 对/错 | 对窗占比 | 盖住低把握对窗 |",
        "|---|---:|---:|---:|",
    ]
    low_base_ok = sum(1 for x in low_only if x["gold_ok"])
    low_preds = [
        ("低把握（无额外）", lambda w: True),
        ("末步把握≥0.90", lambda w: complete(w["geo"]) and w["geo"][-1] >= 0.90),
        ("四步把握都≥0.90", lambda w: all_ge(w["geo"], 0.90)),
        ("末步 Direct≥0.90", lambda w: w["B_final"][-1] == w["B_final"][-1] and w["B_final"][-1] >= 0.90),
        ("四步 Direct 都≥0.90", lambda w: all_ge(w["B_final"], 0.90)),
        ("至少三步 Direct≥0.90", lambda w: complete(w["B_final"]) and count_ge(w["B_final"], 0.90) >= 3),
        ("末步 EAGLE≥0.90", lambda w: w["EAGLE_yn"][-1] == w["EAGLE_yn"][-1] and w["EAGLE_yn"][-1] >= 0.90),
        ("四步 EAGLE 都≥0.90", lambda w: all_ge(w["EAGLE_yn"], 0.90)),
        ("末步口头≥0.90", lambda w: w["verbal"][-1] == w["verbal"][-1] and w["verbal"][-1] >= 0.90),
        ("四步口头都≥0.90", lambda w: all_ge(w["verbal"], 0.90)),
        ("四步 Steer均值都≥0.67", lambda w: all_ge(w["SteerConf_mean"], 0.67)),
        ("把握不降", lambda w: complete(w["geo"]) and w["geo"][-1] >= w["geo"][0]),
        ("把握极差<0.02", lambda w: complete(w["geo"]) and max(w["geo"]) - min(w["geo"]) < 0.02),
    ]
    for name, pred in low_preds:
        s = summarize(low_only, pred)
        rec = s["ok"] / low_base_ok if low_base_ok else float("nan")
        lines.append(f"| {name} | {s['ok']}/{s['bad']} | {pct(s['prec'])} | {pct(rec)} |")

    # coverage of 4-step 4B
    n4b = sum(1 for x in rows if complete(x["B_final"]))
    nverb = sum(1 for x in rows if complete(x["verbal"]))
    lines += [
        "",
        f"四步都有 Direct 的窗 {n4b}/{len(rows)}，都有口头分 {nverb}/{len(rows)}。",
        "对窗占比升高但盖住对窗掉很多，是在挑高把握锁，不是新切口。",
        "四步都过若几乎等于末步过，窗内 4B 没有额外信息。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(rows)} leftover={len(leftover)}", flush=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
