#!/usr/bin/env python3
"""DTSR sufficiency on leftover windows vs geo / wait_helps."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/leftover_dtsr"
TABLE = AE / "tables/leftover_dtsr.md"


def fin(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def load_rows(folder: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and fin(row.get("suff")) == fin(row.get("suff")):
                rows.append(row)
    return rows


def med(xs: list[dict[str, Any]], key: str) -> str:
    vals = np.array([fin(x.get(key)) for x in xs])
    vals = vals[np.isfinite(vals)]
    return "—" if len(vals) == 0 else f"{float(np.median(vals)):.1f}"


def auc(xs: list[dict[str, Any]], yfn, key: str, higher_pos: bool) -> str:
    pair = [(int(bool(yfn(x))), fin(x.get(key))) for x in xs]
    pair = [(y, s) for y, s in pair if s == s]
    if len({y for y, _ in pair}) < 2:
        return "—"
    scores = [s if higher_pos else -s for _, s in pair]
    return f"{float(roc_auc_score([y for y, _ in pair], scores)):.3f}"


def spearman(xs: list[dict[str, Any]], a: str, b: str) -> str:
    va = np.array([fin(x.get(a)) for x in xs])
    vb = np.array([fin(x.get(b)) for x in xs])
    m = np.isfinite(va) & np.isfinite(vb)
    if m.sum() < 8:
        return "—"
    ra = np.argsort(np.argsort(va[m]))
    rb = np.argsort(np.argsort(vb[m]))
    ra = ra.astype(float)
    rb = rb.astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return "—" if den == 0 else f"{float((ra * rb).sum() / den):.3f}"


def fire_line(xs: list[dict[str, Any]], pred) -> str:
    hit = [x for x in xs if pred(x)]
    if not hit:
        return "0"
    wait = sum(1 for x in hit if x.get("wait_helps"))
    ok = sum(1 for x in hit if x.get("left_ok"))
    return f"{len(hit)}（已对 {ok} / 误杀 {wait}）"


def block(title: str, xs: list[dict[str, Any]]) -> list[str]:
    ok = [x for x in xs if x.get("left_ok")]
    wait = [x for x in xs if x.get("wait_helps")]
    both = [x for x in xs if (not x.get("left_ok")) and (not x.get("wait_helps"))]
    pair = ok + wait
    lines = [
        f"### {title}  已对/再等才对/两边都错 {len(ok)}/{len(wait)}/{len(both)}",
        "",
        "| 信号 | 已对中位 | 再等才对中位 | 两边都错中位 | AUROC 再等才对 | AUROC 已对vs再等 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 试答把握 | {med(ok, 'confidence')} | {med(wait, 'confidence')} | {med(both, 'confidence')} | "
        f"{auc(xs, lambda x: x.get('wait_helps'), 'confidence', False)} | "
        f"{auc(pair, lambda x: x.get('left_ok'), 'confidence', True)} |",
        f"| DTSR sufficiency | {med(ok, 'suff')} | {med(wait, 'suff')} | {med(both, 'suff')} | "
        f"{auc(xs, lambda x: x.get('wait_helps'), 'suff', False)} | "
        f"{auc(pair, lambda x: x.get('left_ok'), 'suff', True)} |",
        "",
        f"suff 和把握 Spearman：{spearman(xs, 'suff', 'confidence')}",
        f"论文门槛 τ=100 开火：{fire_line(xs, lambda x: fin(x.get('suff')) >= 100)}",
        f"τ≥90 开火：{fire_line(xs, lambda x: fin(x.get('suff')) >= 90)}",
        "",
    ]
    return lines


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "r1_7b"
    rows = load_rows(ROOT / f"{tag}_s42")
    lines = [
        "# DTSR 思路 sufficiency：第一扇剩窗",
        "",
        "ACL 2026 原问句：第三人称问「这段思路够不够交卷」，0–100。",
        "评委 Qwen3-4B，闭思考，贪心。轨迹是 7B 密探第一扇非高把握同答窗。",
        "正类 wait_helps = 交这扇会错、宿主再等是对的。对照是窗末试答把握。",
        "AUROC 再等才对：分数低更像必须再等（和论文「够了就停」同向）。",
        f"已打 {len(rows)}。",
        "",
    ]
    lines += block("全体", rows)
    for ds, zh in (("math-500", "MATH"), ("olympiadbench", "奥赛"), ("gpqa-diamond", "GPQA")):
        part = [x for x in rows if x.get("dataset") == ds]
        if part:
            lines += block(zh, part)
    for kind, zh in (("low", "低把握"), ("mix", "混合")):
        part = [x for x in rows if x.get("kind") == kind]
        if part:
            lines += block(zh, part)
    lines += [
        "suff 中位已对明显高于再等才对、且 AUROC 明显高于把握，才算另一条问题。",
        "和把握 Spearman 很高，就是口头自信换包装。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
