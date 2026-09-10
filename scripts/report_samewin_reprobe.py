#!/usr/bin/env python3
"""遮完再探：对窗/错窗还守不守原来的 boxed。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/samewin_reprobe"
TABLE = AE / "tables/samewin_reprobe.md"
MASKS = (
    ("none", "不遮（对照）"),
    ("hide_problem", "遮题"),
    ("hide_cot", "遮草稿"),
    ("hide_last", "遮最后一步"),
)
KIND_ZH = {"all": "全体", "high": "高把握", "mix": "混合", "low": "低把握"}


def load_rows(dataset: str, seed: int) -> list[dict[str, Any]]:
    folder = ROOT / f"{dataset}_s{seed}"
    rows: list[dict[str, Any]] = []
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    rows.append(row)
    return rows


def rate(xs: list[dict[str, Any]], key: str = "keep") -> tuple[int, int, float]:
    n = len(xs)
    hit = sum(1 for x in xs if x.get(key))
    return hit, n, (hit / n if n else float("nan"))


def line(name: str, xs: list[dict[str, Any]]) -> list[str]:
    ok = [x for x in xs if x["gold_ok"]]
    bad = [x for x in xs if not x["gold_ok"]]
    out = [f"### {name}  对/错题 {len({x['question_idx'] for x in ok})}/{len({x['question_idx'] for x in bad})}", ""]
    out.append("| 遮法 | 对窗还守原答 | 错窗还守原答 | 对窗新答仍对金标 | 错窗新答变成对 |")
    out.append("|---|---|---|---|---|")
    by = defaultdict(list)
    for x in xs:
        by[x["mask"]].append(x)
    for key, zh in MASKS:
        part = by.get(key) or []
        pok = [x for x in part if x["gold_ok"]]
        pbad = [x for x in part if not x["gold_ok"]]
        ko, no, ro = rate(pok)
        kb, nb, rb = rate(pbad)
        go = rate(pok, "new_gold_ok")
        gb = rate(pbad, "new_gold_ok")
        def fmt(h, n, r):
            return "—" if n == 0 else f"{h}/{n} ({100*r:.0f}%)"
        out.append(
            f"| {zh} | {fmt(ko,no,ro)} | {fmt(kb,nb,rb)} | {fmt(*go)} | {fmt(*gb)} |"
        )
    out.append("")
    return out


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "math-500"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rows = load_rows(dataset, seed)
    lines = [
        "# 四步同答窗：遮完再用 7B 自己重探",
        "",
        "每题每种窗档各取第一扇连续 4 步同答。PUMA 试答收口，贪心 boxed。",
        "对窗 = 原试答对金标。守 = 新 boxed 等于原试答。",
        f"已打 {len(rows)} 条（题×遮法）。",
        "",
    ]
    lines += line("全体", rows)
    for kind, zh in (("high", "高把握"), ("mix", "混合"), ("low", "低把握")):
        part = [x for x in rows if x.get("kind") == kind]
        if part:
            lines += line(zh, part)
    lines += [
        "先看不遮：重探本身就会改口多少。遮法的守答率要明显低于不遮，才算干扰出了规律。",
        "对窗遮题仍守、错窗遮题不守，才是靠题。两边遮草稿都不守，只说明答案写在最后一段。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
