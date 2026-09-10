#!/usr/bin/env python3
"""草稿末尾反口：对窗/错窗守原答还是跟着诱饵走。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/samewin_challenge"
NONE = AE / "results/samewin_reprobe"
TABLE = AE / "tables/samewin_challenge.md"


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


def fmt(xs: list[dict[str, Any]], key: str) -> str:
    n = len(xs)
    if not n:
        return "—"
    hit = sum(1 for x in xs if x.get(key))
    return f"{hit}/{n} ({100 * hit / n:.0f}%)"


def block(title: str, xs: list[dict[str, Any]], none_map: dict[tuple[int, int], dict[str, Any]]) -> list[str]:
    ok = [x for x in xs if x["gold_ok"]]
    bad = [x for x in xs if not x["gold_ok"]]
    nok = [none_map[k] for x in ok if (k := (x["question_idx"], x["decision_step"])) in none_map]
    nbad = [none_map[k] for x in bad if (k := (x["question_idx"], x["decision_step"])) in none_map]
    lines = [
        f"### {title}  对/错 {len(ok)}/{len(bad)}",
        "",
        "| | 对窗守原答 | 错窗守原答 | 对窗跟诱饵 | 错窗跟诱饵 |",
        "|---|---|---|---|---|",
        f"| 不遮（对照） | {fmt(nok, 'keep')} | {fmt(nbad, 'keep')} | — | — |",
        f"| 草稿末反口 | {fmt(ok, 'keep')} | {fmt(bad, 'keep')} | {fmt(ok, 'follow')} | {fmt(bad, 'follow')} |",
        "",
    ]
    return lines


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "math-500"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rows = load_folder(ROOT / f"{dataset}_s{seed}")
    none_rows = [
        x
        for x in load_folder(NONE / f"{dataset}_s{seed}")
        if x.get("mask") == "none"
    ]
    none_map = {(int(x["question_idx"]), int(x["decision_step"])): x for x in none_rows}
    src = {}
    for x in rows:
        src[x.get("bait_src", "?")] = src.get(x.get("bait_src", "?"), 0) + 1
    lines = [
        "# 四步同答窗：草稿末尾加一句反口，7B 重探",
        "",
        "每题每种窗档第一扇。诱饵优先用本题更早的别的试答，否则改原答一个近邻。",
        "守 = 新 boxed 等于原试答。跟 = 新 boxed 等于诱饵。",
        f"已打 {len(rows)}。诱饵来源 {src}。",
        "",
    ]
    lines += block("全体", rows, none_map)
    for kind, zh in (("high", "高把握"), ("mix", "混合"), ("low", "低把握")):
        part = [x for x in rows if x.get("kind") == kind]
        if part:
            lines += block(zh, part, none_map)
    hist = [x for x in rows if x.get("bait_src") == "hist"]
    if hist:
        lines += block("只用历史试答当诱饵", hist, none_map)
    lines += [
        "对窗仍守、错窗跟着走，才是切口。两边都跟，是听最后一句。两边都守，是收口锁死。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
