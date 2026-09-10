#!/usr/bin/env python3
"""boxed 首词质量 / 第二候选整串：对窗 vs 错窗，对照 geo 和再抽裂开。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/samewin_firsttok"
RESAMPLE = AE / "results/samewin_resample"
TABLE = AE / "tables/samewin_firsttok.md"

SIGNALS = (
    ("geo_stored", "原把握", True),
    ("first_p", "首词概率", True),
    ("first_margin", "首词 top1−top2", True),
    ("first_entropy", "首词熵", False),
    ("first_top2_p", "首词第二候选概率", False),
    ("tf_geo", "整串 TF geo", True),
    ("seq_margin", "整串 − 最好历史答", True),
    ("first_tf_margin", "首词 − 最好历史首词", True),
)


def load_folder(folder: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def fin(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def med(xs: list[dict[str, Any]], key: str) -> str:
    vals = np.array([fin(x.get(key)) for x in xs])
    vals = vals[np.isfinite(vals)]
    return "—" if len(vals) == 0 else f"{float(np.median(vals)):.3f}"


def auc(xs: list[dict[str, Any]], key: str, higher_pos: bool) -> str:
    y = [1 if x["gold_ok"] else 0 for x in xs]
    s = [fin(x.get(key)) for x in xs]
    pair = [(yy, ss) for yy, ss in zip(y, s) if ss == ss]
    if len({p[0] for p in pair}) < 2:
        return "—"
    scores = [p[1] if higher_pos else -p[1] for p in pair]
    return f"{float(roc_auc_score([p[0] for p in pair], scores)):.3f}"


def block(title: str, xs: list[dict[str, Any]], split_map: dict[tuple[int, int], bool]) -> list[str]:
    ok = [x for x in xs if x["gold_ok"]]
    bad = [x for x in xs if not x["gold_ok"]]
    lines = [
        f"### {title}  对/错 {len(ok)}/{len(bad)}",
        "",
        "| 信号 | 对窗中位 | 错窗中位 | AUROC |",
        "|---|---:|---:|---:|",
    ]
    for key, zh, higher in SIGNALS:
        lines.append(f"| {zh} | {med(ok, key)} | {med(bad, key)} | {auc(xs, key, higher)} |")
    if split_map:
        tagged = [x for x in xs if (x["question_idx"], x["decision_step"]) in split_map]
        y = [1 if split_map[(x["question_idx"], x["decision_step"])] else 0 for x in tagged]
        if len(set(y)) == 2:
            lines.append("")
            lines.append("用同一信号预测再抽是否裂开（高=更像裂开）：")
            lines.append("")
            lines.append("| 信号 | AUROC |")
            lines.append("|---|---:|")
            for key, zh, higher in SIGNALS:
                s = [fin(x.get(key)) for x in tagged]
                pair = [(yy, ss) for yy, ss in zip(y, s) if ss == ss]
                if len({p[0] for p in pair}) < 2:
                    val = "—"
                else:
                    scores = [p[1] if not higher else -p[1] for p in pair]
                    val = f"{float(roc_auc_score([p[0] for p in pair], scores)):.3f}"
                lines.append(f"| {zh} | {val} |")
    lines.append("")
    return lines


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "math-500"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rows = load_folder(ROOT / f"{dataset}_s{seed}")
    res = load_folder(RESAMPLE / f"{dataset}_s{seed}")
    split_map = {(int(x["question_idx"]), int(x["decision_step"])): bool(x.get("split")) for x in res}
    n_alt = sum(1 for x in rows if int(x.get("n_hist_alt") or 0) > 0)
    lines = [
        "# 四步同答窗：boxed 首词质量和第二候选整串",
        "",
        "每题每种窗档第一扇。前缀停在 `\\\\boxed`，贪心 24 token 读首个内容词 top-20；",
        "再 teacher-force `{原答}` 和本题更早的别的试答。",
        f"已打 {len(rows)}。有历史别答 {n_alt}。",
        "",
    ]
    lines += block("全体", rows, split_map)
    for kind, zh in (("high", "高把握"), ("mix", "混合"), ("low", "低把握")):
        part = [x for x in rows if x.get("kind") == kind]
        if part:
            lines += block(zh, part, split_map)
    lines += [
        "AUROC 高=更像对窗。首词熵 / 第二候选概率反过来。对照是原把握。",
        "后半段预测的是再抽裂开，不是对错。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
