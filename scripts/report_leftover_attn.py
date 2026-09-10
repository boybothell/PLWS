#!/usr/bin/env python3
"""AUROC of leftover-window question attention and Wait layer contrast."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg

TABLE = AE / "tables/leftover_attn.md"
SCORE_DIR = AE / "results/leftover_attn"
CELLS = (
    ("7B MATH", "r1_7b_math-500"),
    ("7B GPQA", "r1_7b_gpqa-diamond"),
)
SIGS = (
    ("confidence", "把握（对照）"),
    ("lookback_ratio", "boxed 题面占比"),
    ("lb_q", "boxed 题面质量"),
    ("thought_lookback_ratio", "思路末 题面占比"),
    ("thought_lb_q", "思路末 题面质量"),
    ("q_entropy", "题面 key 熵"),
    ("attn_H", "全序列熵（对照）"),
    ("lens_rise", "boxed 末层−一半深（对照）"),
    ("wait_lookback_ratio", "Wait 题面占比"),
    ("wait_lb_q", "Wait 题面质量"),
    ("wait_q_entropy", "Wait 题面熵"),
    ("wait_lens_rise", "Wait 末层−一半深"),
    ("wait_next_logp", "Wait 下一词 logp"),
    ("wait_stop_margin", "Wait 处收口−Wait"),
)


def load_scores(stem: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(SCORE_DIR.glob(f"{stem}_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") in {"ok", "partial"}:
                rows.append(row)
    return rows


def pair(rows: list[dict[str, Any]], field: str) -> tuple[list[float], list[float]]:
    pos, neg = [], []
    for row in rows:
        val = row.get(field)
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if num != num:
            continue
        (pos if row.get("pos") else neg).append(num)
    return pos, neg


def fmt(x: float) -> str:
    return "—" if x != x else f"{x:.3f}"


def cell_block(name: str, rows: list[dict[str, Any]]) -> list[str]:
    n = len(rows)
    n_pos = sum(1 for r in rows if r.get("pos"))
    n_wait = sum(1 for r in rows if r.get("wait_pos") is not None and r.get("wait_pos") == r.get("wait_pos"))
    lines = [
        f"## {name}",
        "",
        f"齐 {n}，可停 {n_pos}，有 Wait {n_wait}。正类 = 试答对上金标 / 写完终答。",
        "",
        "| 尺子 | n+ | n− | 越大越好 | 越小越好 | 更好的一边 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    best_name, best_auc = "", 0.0
    for field, zh in SIGS:
        pos, neg = pair(rows, field)
        hi = rg.auroc(pos, neg)
        lo = rg.auroc(neg, pos)
        better = hi if hi == hi and (lo != lo or hi >= lo) else lo
        side = "大" if better == hi else "小"
        lines.append(f"| {zh} | {len(pos)} | {len(neg)} | {fmt(hi)} | {fmt(lo)} | {fmt(better)}（{side}） |")
        if better == better and better > best_auc and field != "confidence":
            best_auc = better
            best_name = zh
    go = "能过 0.70" if best_auc >= 0.70 else "过不了 0.70"
    lines.append("")
    lines.append(f"除把握外最好：{best_name or '无'} {fmt(best_auc)}，{go}。")
    lines.append("")
    return lines


def main() -> None:
    lines = [
        "# 第一扇剩窗：题面注意力 / 自然 Wait 层对比",
        "",
        "7B MATH / GPQA，只打第一扇非高把握同答窗最后一步。一次前向同时读题面注意力和 CoT 里最后一个 Wait。",
        "正类 = 试答已经对上金标或写完终答。AUROC ≥ 0.70 才谈门；贴 0.5 就停。",
        "",
    ]
    missing = []
    for name, stem in CELLS:
        rows = load_scores(stem)
        if not rows:
            missing.append(name)
            lines.append(f"## {name}")
            lines.append("")
            lines.append("还没齐。")
            lines.append("")
            continue
        lines.extend(cell_block(name, rows))
    if missing:
        lines.append(f"未齐：{', '.join(missing)}。")
        lines.append("")
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
