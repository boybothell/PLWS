#!/usr/bin/env python3
"""剩窗后推理 token 熵：假平台改口前有没有逐步分叉。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import report_leftover_after as after
import train_leftover_commit_probe as tp

IN = AE / "results/leftover_token_pivot"
TABLE = AE / "tables/leftover_token_pivot.md"


def load_scores() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(IN.glob("*_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def mean(xs: list[float]) -> float:
    finite = [x for x in xs if x == x]
    return float(np.mean(finite)) if finite else float("nan")


def auroc_key(xs: list[dict[str, Any]], yfn, key: str) -> float:
    y = np.asarray([int(bool(yfn(x))) for x in xs])
    s = np.asarray([float(x.get(key, float("nan"))) for x in xs])
    return tp.auroc(y, s)


def loto(xs: list[dict[str, Any]], yfn, key: str) -> str:
    groups = {
        "奥赛": [x for x in xs if x["dataset"] == "olympiadbench"],
        "GPQA": [x for x in xs if x["dataset"] == "gpqa-diamond"],
        "AIME": [x for x in xs if x["dataset"] in ("aime24", "aime25")],
    }
    parts = []
    for _name, test in groups.items():
        y = np.asarray([int(bool(yfn(x))) for x in test])
        s = np.asarray([float(x.get(key, float("nan"))) for x in test])
        if len(test) < 8 or y.min() == y.max():
            parts.append("—")
        else:
            parts.append(tp.fmt(tp.auroc(y, s)))
    return " | ".join(parts)


def line(name: str, xs: list[dict[str, Any]]) -> str:
    return (
        f"| {name} | {len(xs)} | {mean([x['next_n'] for x in xs]):.0f} | "
        f"{mean([x['next_ent_first'] for x in xs]):.2f} | "
        f"{mean([x['next_ent_max'] for x in xs]):.2f} | "
        f"{mean([x['next_ent_mean'] for x in xs]):.2f} | "
        f"{mean([x['next_min_p1'] for x in xs]):.2f} | "
        f"{mean([x['next_frac_not_top1'] for x in xs]):.2f} | "
        f"{mean([x['last32_ent_max'] for x in xs]):.2f} | "
        f"{mean([x['gap_at_max'] for x in xs]):.2f} |"
    )


def main() -> None:
    rows = load_scores()
    wait_ch = [x for x in rows if x.get("wait_helps") and x.get("will_change")]
    ok_same = [x for x in rows if x.get("left_ok") and x.get("same_as_high")]
    wait_hold = [x for x in wait_ch if x.get("next_same")]
    ok_hold = [x for x in ok_same if x.get("next_same")]
    wait_now = [x for x in wait_ch if not x.get("next_same")]
    wait_nv = [x for x in rows if x.get("wait_helps") and x.get("never_high")]
    ok_nv = [x for x in rows if x.get("left_ok") and x.get("never_high")]
    hold = [x for x in rows if x.get("next_same")]
    y_all = lambda x: bool(x.get("wait_helps") and x.get("will_change"))
    lines = [
        "# 剩窗后推理 token 有没有分叉",
        "",
        "只看第一扇剩窗之后、下一步试答之前写下的思路词。",
        "不是试答把握，是每个新词的下一步分布：熵高 = 当时好几个词都像；",
        "最小第一候选概率低 / 没走第一候选 = 当时几乎在两条路上。",
        "下一步仍同答 = 改口还没发生，这时干预还来得及。",
        "改口当步最后 32 词 = 新试答马上要写出来之前。",
        "",
        f"已打分 {len(rows)}。状态不是 ok 的不进表。",
        "",
        "## 1. 切片均值",
        "",
        "| 切片 | 题 | 下一步新词数 | 第一步熵 | 最高熵 | 平均熵 | 最小第一候选概率 | 没走第一候选比例 | 改口前最后32词最高熵 | 最高熵处第一−第二候选差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        line("误杀·后面换答", wait_ch),
        line("误杀·换答、下一步仍同答", wait_hold),
        line("误杀·换答、下一步就改口", wait_now),
        line("已对·后面同答锁", ok_same),
        line("已对·同答锁、下一步仍同答", ok_hold),
        line("误杀·写完才对", wait_nv),
        line("已对·写完才停", ok_nv),
        "",
        "## 2. 还能不能分开「会换答」",
        "",
        "正类 = 误杀且后面换答。分数越大越像正类。留一集防认数据集。",
        "",
        "| 尺子 | 全体 | 只看下一步仍同答 | 留一奥赛 | 留一 GPQA | 留一 AIME |",
        "|---|---|---|---|---|---|",
    ]
    for name, key, xs in (
        ("下一步最高熵", "next_ent_max", rows),
        ("下一步平均熵", "next_ent_mean", rows),
        ("下一步第一步熵", "next_ent_first", rows),
        ("1−下一步最小第一候选概率", "next_min_p1_inv", rows),
        ("下一步没走第一候选比例", "next_frac_not_top1", rows),
        ("最后32词最高熵", "last32_ent_max", rows),
        ("1−最高熵处第一−第二差", "gap_at_max_inv", rows),
        ("下一步最高熵（下一步仍同答）", "next_ent_max", hold),
    ):
        for x in rows:
            x["next_min_p1_inv"] = (
                -x["next_min_p1"] if x.get("next_min_p1") == x.get("next_min_p1") else float("nan")
            )
            x["gap_at_max_inv"] = (
                -x["gap_at_max"] if x.get("gap_at_max") == x.get("gap_at_max") else float("nan")
            )
        use = xs
        lines.append(
            f"| {name} | {tp.fmt(auroc_key(use, y_all, key))} | "
            f"{tp.fmt(auroc_key([x for x in use if x.get('next_same')], y_all, key))} | "
            f"{loto(use, y_all, key)} |"
        )
    n_status = {}
    for path in sorted(IN.glob("*_shard*.jsonl")):
        for raw in path.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            n_status[row.get("status", "?")] = n_status.get(row.get("status", "?"), 0) + 1
    by_model = {}
    for x in rows:
        by_model[x["model"]] = by_model.get(x["model"], 0) + 1
    lines += [
        "",
        f"打分状态：{n_status}。按模型 ok：{by_model}。",
        "",
        "## 3. 读法",
        "",
        "若「下一步仍同答」时误杀换答已经最高熵明显更高、更常不走第一候选，才有逐步分叉可干预。",
        "若只有「下一步就改口」或最后 32 词才分开，干预晚了。留一 GPQA 掉到 0.5 仍是认数据集。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(rows)} wait_ch={len(wait_ch)} hold={len(wait_hold)}", flush=True)


if __name__ == "__main__":
    main()
