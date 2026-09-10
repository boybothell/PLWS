#!/usr/bin/env python3
"""剩窗窗末把握：一条门槛跨数据集，相对密探能提多少。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import train_leftover_commit_probe as tp

TABLE = AE / "tables/leftover_conf_tau.md"
HARD = ("olympiadbench", "gpqa-diamond", "aime24", "aime25")
TAUS = (0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)


def apply(xs: list[dict[str, Any]], tau: float) -> dict[str, float]:
    n = max(len(xs), 1)
    acc = tok = fire = host_acc = host_tok = 0.0
    for x in xs:
        host_acc += int(x["nofs_ok"])
        host_tok += float(x["keep_tok"])
        if float(x.get("confidence") or 0) >= tau:
            acc += int(x["leftover_ok"])
            tok += float(x["decision_step"])
            fire += 1
        else:
            acc += int(x["nofs_ok"])
            tok += float(x["keep_tok"])
    return {
        "acc": acc / n,
        "tok": tok / n,
        "fire": fire,
        "n": float(len(xs)),
        "d_acc": 100.0 * (acc / n - host_acc / n),
        "d_tok": tok / n - host_tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
    }


def cell(rec: dict[str, float]) -> str:
    return f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f}（{int(rec['fire'])}/{int(rec['n'])}）"


def pick_safe(rows: list[dict[str, Any]], groups: list[list[dict[str, Any]]]) -> tuple[float, dict[str, float]]:
    """一条门槛：每个子集 Acc 都不低于宿主，再尽量省步。"""
    best: tuple[float, dict[str, float]] | None = None
    for tau in TAUS:
        recs = [apply(g, tau) for g in groups if g]
        if not recs or any(r["d_acc"] < -1e-9 for r in recs):
            continue
        all_rec = apply(rows, tau)
        key = (-all_rec["d_tok"], all_rec["d_acc"])
        if best is None or key > (-best[1]["d_tok"], best[1]["d_acc"]):
            best = (tau, all_rec)
    if best is None:
        return float("nan"), apply(rows, 99.0)
    return best


def main() -> None:
    rows = [r for r in tp.load_scored() if r["_has_af"]]
    by_model = tp.group_model(rows)
    lines = [
        "# 剩窗窗末把握：一条门槛跨数据集",
        "",
        "只切第一扇非高把握同答窗。窗末把握 ≥ τ 就交这扇试答，否则留密探（无密探则留写完/PUMA 对错）。",
        "数字是**这批剩窗题**上相对「全部留密探」。步数近似 token。",
        "共用 = 这些集用同一个 τ。Acc 不降 = 每个子集都不低于宿主，再选最省步的 τ。",
        "",
        "## 1. 每模型一条门槛：难集 Acc 都不降",
        "",
        "| 模型 | τ | 难集合计 | 奥赛 | GPQA | AIME24 | AIME25 | MATH（同 τ，不参与选门槛） |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for model, xs in by_model.items():
        hard = [x for x in xs if x["dataset"] in HARD]
        groups = [[x for x in hard if x["dataset"] == ds] for ds in HARD]
        tau, rec = pick_safe(hard, groups)
        math = [x for x in xs if x["dataset"] == "math-500"]
        math_rec = apply(math, tau) if math and tau == tau else None
        def one(ds: str) -> str:
            g = [x for x in hard if x["dataset"] == ds]
            return cell(apply(g, tau)) if g and tau == tau else "—"
        lines.append(
            f"| {tp.MODEL_ZH[model]} | {tau:.3f} | {cell(rec)} | "
            f"{one('olympiadbench')} | {one('gpqa-diamond')} | "
            f"{one('aime24')} | {one('aime25')} | "
            f"{cell(math_rec) if math_rec else '—'} |"
        )

    lines += [
        "",
        "## 2. 全体一条门槛：所有模型难集 Acc 都不降",
        "",
    ]
    hard_all = [x for x in rows if x["dataset"] in HARD]
    groups = []
    for model, xs in by_model.items():
        for ds in HARD:
            g = [x for x in xs if x["dataset"] == ds]
            if g:
                groups.append(g)
    tau, rec = pick_safe(hard_all, groups)
    lines.append(f"选中 τ = **{tau:.3f}**。难集合计：{cell(rec)}。")
    lines += [
        "",
        "| 集 | ΔAcc / 步 |",
        "|---|---|",
    ]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[tp.cell_name(row)].append(row)
    for name in sorted(by_cell):
        if "MATH" in name:
            continue
        lines.append(f"| {name} | {cell(apply(by_cell[name], tau))} |")
    lines.append(f"| 难集合计 | {cell(rec)} |")

    lines += [
        "",
        "## 3. 扫门槛（难集合计，不要求每集都不伤）",
        "",
        "| τ | 难集合计 | 其中伤几格（模型×难集） |",
        "|---:|---|---:|",
    ]
    n_cells = len(groups)
    for t in TAUS:
        rec = apply(hard_all, t)
        hurt = sum(1 for g in groups if apply(g, t)["d_acc"] < -1e-9)
        lines.append(f"| {t:.3f} | {cell(rec)} | {hurt}/{n_cells} |")

    lines += [
        "",
        "## 4. 难集合计 Acc 不降（不要求每一格）",
        "",
        "一条 τ，所有难集剩窗加在一起 Acc 不低于宿主，再最省步。这是「同一套阈值」能报的最好合计，会掩盖某集受伤。",
        "",
        "| 范围 | τ | 难集合计 | 伤几格 |",
        "|---|---:|---|---:|",
    ]
    def pick_pool(xs: list[dict[str, Any]], groups: list[list[dict[str, Any]]]) -> tuple[float, dict[str, float], int]:
        best = None
        for t in TAUS:
            rec = apply(xs, t)
            if rec["d_acc"] < -1e-9:
                continue
            hurt = sum(1 for g in groups if g and apply(g, t)["d_acc"] < -1e-9)
            key = (-rec["d_tok"], rec["d_acc"], -hurt)
            if best is None or key > (-best[1]["d_tok"], best[1]["d_acc"], -best[2]):
                best = (t, rec, hurt)
        if best is None:
            return 99.0, apply(xs, 99.0), 0
        return best

    t, rec, hurt = pick_pool(hard_all, groups)
    lines.append(f"| 全体模型难集 | {t:.3f} | {cell(rec)} | {hurt}/{n_cells} |")
    for model, xs in by_model.items():
        hard = [x for x in xs if x["dataset"] in HARD]
        gs = [[x for x in hard if x["dataset"] == ds] for ds in HARD]
        t, rec, hurt = pick_pool(hard, gs)
        lines.append(
            f"| {tp.MODEL_ZH[model]} 难集 | {t:.3f} | {cell(rec)} | {hurt}/{sum(1 for g in gs if g)} |"
        )

    lines += [
        "",
        "读法：第 1、2 节才是「各数据集同一套阈值且每集不伤」。第 4 节是合计不伤。第 3 节会虚高。",
        "步数不是官方 token，只看相对多少。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
