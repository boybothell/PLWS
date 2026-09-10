#!/usr/bin/env python3
"""延迟 4 步后仍同答的低把握厚平台：残留误杀和已对还分不分得开。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import report_leftover_after as after

TABLE = AE / "tables/leftover_thick.md"


def med(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def fmt(v: float) -> str:
    return "—" if v != v else f"{v:.2f}"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos = [float(a) for a, t in zip(s, y) if t == 1 and a == a]
    neg = [float(a) for a, t in zip(s, y) if t == 0 and a == a]
    return rg.auroc(pos, neg)


def is_thick(x: dict[str, Any]) -> bool:
    high_first = x["high_gap"] is not None and x["high_gap"] <= 4 and x["same_as_high"]
    return x["kind"] == "low" and bool(x["same_for"][4]) and not high_first


def cls(x: dict[str, Any]) -> str:
    if x["wait_helps"]:
        return "wait"
    if x["left_ok"]:
        return "ok"
    return "both"


def fate(x: dict[str, Any]) -> str:
    if x["will_change"]:
        return "later_change"
    if x["same_as_high"]:
        return "later_same"
    return "never_high"


def eval_rule(all_rows: list[dict[str, Any]], pred) -> dict[str, float]:
    acc = tok = fire = wait = host_acc = host_tok = 0.0
    for x in all_rows:
        host_acc += int(x["host_ok"])
        host_tok += x["host_tok"]
        if pred(x):
            acc += int(x["left_ok"])
            tok += x["left_tok"]
            fire += 1
            wait += int(x["wait_helps"])
        else:
            acc += int(x["host_ok"])
            tok += x["host_tok"]
    n = float(len(all_rows))
    return {
        "d_acc": 100.0 * (acc / n - host_acc / n),
        "d_tok": tok / n - host_tok / n,
        "n_fire": fire,
        "wait_fire": wait,
        "n": n,
        "ok": acc + 1e-12 >= host_acc,
    }


def cell(rec: dict[str, float]) -> str:
    flag = "" if rec["ok"] else " 伤"
    return (
        f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f}"
        f"（{int(rec['n_fire'])}/{int(rec['n'])}，误杀 {int(rec['wait_fire'])}）{flag}"
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    rows: list[dict[str, Any]] = []
    for _zh, model in after.MODELS:
        for _ds_zh, dataset in after.DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                part = after.load_left(model, dataset, seed)
                rows.extend(part)
                print(f"{model} {dataset} s{seed} leftover={len(part)}", flush=True)
    thick = [x for x in rows if is_thick(x)]
    wait = [x for x in thick if x["wait_helps"]]
    ok = [x for x in thick if x["left_ok"]]
    both = [x for x in thick if cls(x) == "both"]
    lines = [
        "# 厚平台：低把握、再同答至少 4 步",
        "",
        "薄假平台（剩窗后 1–2 步就改口）延迟已经能挡。这里只看还挡住的：",
        "第一扇是低把握，再走 4 步仍是同一答，且这 4 步里密探还没高把握锁上。",
        "数字仍是剩窗题相对密探。token 用剩窗切点近似。",
        "",
        f"全体剩窗 {len(rows)}，厚平台 {len(thick)}：已对 {len(ok)}，误杀 {len(wait)}，两边都错 {len(both)}。",
        "",
        "## 1. 厚平台里三摊各是什么",
        "",
        "| 切片 | 题 | 后面换答 | 后面同答锁 | 写完才停 | 中位再坚持 | 4 步后把握 | 把握升降 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, xs in (
        ("厚·已对", ok),
        ("厚·误杀", wait),
        ("厚·两边都错", both),
        ("厚·已对·后面同答锁", [x for x in ok if x["same_as_high"]]),
        ("厚·已对·写完才停", [x for x in ok if x["never_high"]]),
        ("厚·误杀·后面换答", [x for x in wait if x["will_change"]]),
        ("厚·误杀·写完才对", [x for x in wait if x["never_high"]]),
    ):
        if not xs:
            lines.append(f"| {name} | 0 | — | — | — | — | — | — |")
            continue
        more = [x["persist"] - 4 for x in xs]
        lines.append(
            f"| {name} | {len(xs)} | {sum(1 for x in xs if x['will_change'])} | "
            f"{sum(1 for x in xs if x['same_as_high'])} | {sum(1 for x in xs if x['never_high'])} | "
            f"{med(more):.0f} | {mean([x['c4'] for x in xs]):.3f} | {mean([x['rise4'] for x in xs]):+.3f} |"
        )

    lines += [
        "",
        "## 2. 厚平台落在哪一集",
        "",
        "| 集 | 厚平台 | 已对 | 误杀 | 已对里写完 | 误杀里换答 | 误杀里写完 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in thick:
        by_cell[f"{x['model']}\t{x['dataset']}"].append(x)
    for key in sorted(by_cell):
        model, ds = key.split("\t")
        xs = by_cell[key]
        lines.append(
            f"| {after.MODEL_ZH[model]} {after.DS_ZH[ds]} | {len(xs)} | "
            f"{sum(1 for x in xs if x['left_ok'])} | {sum(1 for x in xs if x['wait_helps'])} | "
            f"{sum(1 for x in xs if x['left_ok'] and x['never_high'])} | "
            f"{sum(1 for x in xs if x['wait_helps'] and x['will_change'])} | "
            f"{sum(1 for x in xs if x['wait_helps'] and x['never_high'])} |"
        )

    lines += [
        "",
        "## 3. 4 步后这扇窗长什么样",
        "",
        "extra 窗 = 剩窗后再连续 4 步同答的把握档。",
        "",
        "| 4 步后档 | 厚·已对 | 厚·误杀 | 厚·两边都错 |",
        "|---|---:|---:|---:|",
    ]
    for kind in ("low", "mix", "high"):
        lines.append(
            f"| {kind} | {sum(1 for x in ok if x.get('extra_kind') == kind)} | "
            f"{sum(1 for x in wait if x.get('extra_kind') == kind)} | "
            f"{sum(1 for x in both if x.get('extra_kind') == kind)} |"
        )

    lines += [
        "",
        "## 4. 厚平台上，4 步后的把握还分不分得开",
        "",
        "只在厚平台里比。越大越像「已对」。留一数据集防认出哪一集。",
        "",
        "| 尺子 | 全体厚平台 | 留一奥赛 | 留一 GPQA | 留一 AIME |",
        "|---|---|---|---|---|",
    ]
    y_ok = np.asarray([int(x["left_ok"]) for x in thick])
    y_w = np.asarray([int(x["wait_helps"]) for x in thick])

    def col(xs: list[dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray([float(x.get(key, float("nan"))) for x in xs])

    def loto(key: str, pred) -> str:
        parts = []
        groups = {
            "奥赛": [x for x in thick if x["dataset"] == "olympiadbench"],
            "GPQA": [x for x in thick if x["dataset"] == "gpqa-diamond"],
            "AIME": [x for x in thick if x["dataset"] in ("aime24", "aime25")],
        }
        for name, test in groups.items():
            train = [x for x in thick if x not in test]
            if len(test) < 8 or len(train) < 8:
                parts.append("—")
                continue
            y_te = np.asarray([int(pred(x)) for x in test])
            s_te = col(test, key)
            if y_te.min() == y_te.max():
                parts.append("—")
                continue
            parts.append(fmt(auroc(y_te, s_te)))
        return " | ".join(parts)

    for name, key in (
        ("剩窗末把握", "left_c"),
        ("再走 4 步后把握", "c4"),
        ("4 步把握升降", "rise4"),
        ("4 步最低把握", "min4"),
        ("再坚持多少步", "persist"),
    ):
        s = col(thick, key)
        lines.append(
            f"| {name} | {fmt(auroc(y_ok, s))}（误杀 {fmt(auroc(y_w, s))}） | {loto(key, lambda x: x['left_ok'])} |"
        )

    lines += [
        "",
        "## 5. 还是 Acc 不降再省步",
        "",
        "规则都只允许在厚平台上开火，薄的一律留密探。",
        "",
        "| 规则 | 全体剩窗 | 奥赛+AIME | GPQA |",
        "|---|---|---|---|",
    ]
    oly = [x for x in rows if x["dataset"] != "gpqa-diamond"]
    gpq = [x for x in rows if x["dataset"] == "gpqa-diamond"]
    rules = (
        ("厚平台全交", lambda x: is_thick(x)),
        ("厚平台且 4 步后是混合窗", lambda x: is_thick(x) and x.get("extra_kind") == "mix"),
        ("厚平台且 4 步后把握升", lambda x: is_thick(x) and x.get("rise4") == x.get("rise4") and x["rise4"] > 0),
        ("厚平台且 4 步后把握 ≥0.90", lambda x: is_thick(x) and x.get("c4") == x.get("c4") and x["c4"] >= 0.90),
        ("厚平台且 4 步后把握 ≥0.95", lambda x: is_thick(x) and x.get("c4") == x.get("c4") and x["c4"] >= 0.95),
        ("先知：厚平台且试答已对", lambda x: is_thick(x) and x["left_ok"]),
    )
    for name, pred in rules:
        lines.append(
            f"| {name} | {cell(eval_rule(rows, pred))} | "
            f"{cell(eval_rule(oly, pred))} | {cell(eval_rule(gpq, pred))} |"
        )
    lines += [
        "",
        "先知是上限。好规则 = 三列都不伤，且 token 为负。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} leftover={len(rows)} thick={len(thick)}", flush=True)


if __name__ == "__main__":
    main()
