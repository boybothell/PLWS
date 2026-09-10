#!/usr/bin/env python3
"""全部四步同答窗：已有 4B 口头 0–100 / P(Yes) 接到 7B MATH/奥赛/GPQA。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_confcal_v1 as v1
import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_second_lock as sl
import report_leftover_after as after
import train_leftover_commit_probe as tp

TABLE = AE / "tables/samewin_verbal_4b.md"
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
)


def finite(x: Any) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x if x == x else float("nan")


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def med(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return float(np.median(xs)) if xs else float("nan")


def load_wins(dataset: str) -> list[dict[str, Any]]:
    scores = {}
    for row in v1.assemble(dataset):
        scores[(row["question_idx"], row["decision_step"])] = row
    trials_path = room.dense_trial_path("r1_7b", dataset, 42)
    puma_path = dd.puma_stat_path("r1_7b", dataset, 42)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    gp = low.gpath("r1_7b", dataset, 42)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    out = []
    for qi, trials in by.items():
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        info = official.get(qi) or {}
        g = gmap.get(qi) or {}
        last = max(trials, key=lambda x: int(x["stopped_len"]))
        gt = info.get("ground_truth") or g.get("ground_truth")
        original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
        orig_ok = bool(info.get("original_correct")) if "original_correct" in info else bool(
            low.credit(original, gt, original, True)
        )
        for win in sl.same_windows(rows):
            rec = scores.get((qi, win["step"]))
            if rec is None:
                continue
            gold_ok = bool(low.credit(win["ans"], gt, original, orig_ok))
            af_ok = bool(rg.same(win["ans"], original))
            out.append(
                {
                    "dataset": dataset,
                    "kind": win["kind"],
                    "gold_ok": gold_ok,
                    "af_ok": af_ok,
                    "geo": finite(rec.get("geo_conf")),
                    "verbal": finite(rec.get("verbal_h3")),
                    "yes": finite(rec.get("direct_yes")),
                }
            )
    return out


def auroc(xs: list[dict[str, Any]], ykey: str, skey: str) -> float:
    y = np.asarray([int(bool(x[ykey])) for x in xs])
    s = np.asarray([float(x[skey]) for x in xs])
    return tp.auroc(y, s)


def block(name: str, xs: list[dict[str, Any]], ykey: str) -> list[str]:
    ok = [x for x in xs if x[ykey]]
    bad = [x for x in xs if not x[ykey]]
    return [
        f"| {name} | {len(ok)}/{len(bad)} | "
        f"{mean([x['verbal'] for x in ok]):.2f} / {mean([x['verbal'] for x in bad]):.2f} | "
        f"{med([x['verbal'] for x in ok]):.2f} / {med([x['verbal'] for x in bad]):.2f} | "
        f"{mean([x['yes'] for x in ok]):.2f} / {mean([x['yes'] for x in bad]):.2f} | "
        f"{mean([x['geo'] for x in ok]):.2f} / {mean([x['geo'] for x in bad]):.2f} | "
        f"{tp.fmt(auroc(xs, ykey, 'verbal'))} / {tp.fmt(auroc(xs, ykey, 'yes'))} / "
        f"{tp.fmt(auroc(xs, ykey, 'geo'))} |"
    ]


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    all_rows: list[dict[str, Any]] = []
    lines = [
        "# 全部四步同答窗：4B 口头置信度",
        "",
        "不是只剩窗。每一扇连续 4 步同一 boxed 都进表，含高把握 / 混合 / 低把握。",
        "4B 是已有 v1 抽取：闭思考，口头 0–100（只许数字），以及 Yes/No 的 P(Yes)。",
        "解题模型是 7B。对窗 = 这扇试答对金标。另报等于写完终答。",
        "口头分已除以 100。AUROC 越大越像对窗。",
        "",
        "## 1. 对金标",
        "",
        "| 切片 | 对/错 | 口头均值 | 口头中位 | P(Yes) 均值 | 试答把握均值 | AUROC 口头 / P(Yes) / 把握 |",
        "|---|---:|---|---|---|---|---|",
    ]
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for zh, dataset in DS:
        part = load_wins(dataset)
        by_ds[dataset] = part
        all_rows.extend(part)
        print(f"{dataset} windows={len(part)}", flush=True)
        lines += block(zh, part, "gold_ok")
        for kind, kind_zh in (("high", "高把握"), ("mix", "混合"), ("low", "低把握")):
            lines += block(f"{zh}·{kind_zh}", [x for x in part if x["kind"] == kind], "gold_ok")
    lines += block("三集合计", all_rows, "gold_ok")
    for kind, kind_zh in (("high", "高把握"), ("mix", "混合"), ("low", "低把握")):
        lines += block(f"合计·{kind_zh}", [x for x in all_rows if x["kind"] == kind], "gold_ok")
    lines += [
        "",
        "## 2. 对写完终答",
        "",
        "| 切片 | 等于写完/不等 | 口头均值 | 口头中位 | P(Yes) 均值 | 试答把握均值 | AUROC 口头 / P(Yes) / 把握 |",
        "|---|---:|---|---|---|---|---|",
    ]
    for zh, dataset in DS:
        lines += block(zh, by_ds[dataset], "af_ok")
    lines += block("三集合计", all_rows, "af_ok")
    n_v = sum(1 for x in all_rows if x["verbal"] == x["verbal"])
    lines += [
        "",
        f"接到口头分的窗 {n_v}/{len(all_rows)}。滑动窗：长平台会重复计。",
        "",
        "## 3. 读法",
        "",
        "高把握窗对错在口头分上若仍叠在一起，口头分没有比试答把握多出一截。",
        "低把握才是剩窗那种题。口头分若只在高把握上分开，整体区分度是锁带来的。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
