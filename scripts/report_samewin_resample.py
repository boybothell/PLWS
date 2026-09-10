#!/usr/bin/env python3
"""同点再抽：对窗/错窗守原答还是自己裂开。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/samewin_resample"
NONE = AE / "results/samewin_reprobe"
TABLE = AE / "tables/samewin_resample.md"


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


def fmt_win(xs: list[dict[str, Any]], key: str) -> str:
    n = len(xs)
    if not n:
        return "—"
    hit = sum(1 for x in xs if x.get(key))
    return f"{hit}/{n} ({100 * hit / n:.0f}%)"


def fmt_draw(xs: list[dict[str, Any]], key: str) -> str:
    draws = [d for x in xs for d in x.get("draws") or []]
    n = len(draws)
    if not n:
        return "—"
    hit = sum(1 for d in draws if d.get(key))
    return f"{hit}/{n} ({100 * hit / n:.0f}%)"


def uniq(xs: list[dict[str, Any]]) -> str:
    if not xs:
        return "—"
    vals = []
    for x in xs:
        answers = {str(d.get("new_answer") or "") for d in x.get("draws") or []}
        vals.append(len(answers))
    return f"{sum(vals) / len(vals):.2f}"


def block(title: str, xs: list[dict[str, Any]], none_map: dict[tuple[int, int], dict[str, Any]]) -> list[str]:
    ok = [x for x in xs if x["gold_ok"]]
    bad = [x for x in xs if not x["gold_ok"]]
    nok = [none_map[k] for x in ok if (k := (x["question_idx"], x["decision_step"])) in none_map]
    nbad = [none_map[k] for x in bad if (k := (x["question_idx"], x["decision_step"])) in none_map]
    return [
        f"### {title}  对/错 {len(ok)}/{len(bad)}",
        "",
        "| | 对窗守原答 | 错窗守原答 | 对窗裂开 | 错窗裂开 | 对窗新答仍对 | 错窗新答变成对 |",
        "|---|---|---|---|---|---|---|",
        f"| 贪心对照 | {fmt_win(nok, 'keep')} | {fmt_win(nbad, 'keep')} | — | — | {fmt_win(nok, 'new_gold_ok')} | {fmt_win(nbad, 'new_gold_ok')} |",
        f"| 单次再抽 | {fmt_draw(ok, 'keep')} | {fmt_draw(bad, 'keep')} | — | — | {fmt_draw(ok, 'new_gold_ok')} | {fmt_draw(bad, 'new_gold_ok')} |",
        f"| 四条全守/裂开 | {fmt_win(ok, 'keep_all')} | {fmt_win(bad, 'keep_all')} | {fmt_win(ok, 'split')} | {fmt_win(bad, 'split')} | — | — |",
        "",
        f"每扇不同答条数：对窗 {uniq(ok)}，错窗 {uniq(bad)}。",
        "",
    ]


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "math-500"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rows = load_folder(ROOT / f"{dataset}_s{seed}")
    none_rows = [x for x in load_folder(NONE / f"{dataset}_s{seed}") if x.get("mask") == "none"]
    none_map = {(int(x["question_idx"]), int(x["decision_step"])): x for x in none_rows}
    n_draw = rows[0].get("n_draw") if rows else 0
    temp = rows[0].get("temperature") if rows else None
    lines = [
        "# 四步同答窗：同点再抽，不改题不改草稿",
        "",
        "每题每种窗档第一扇。PUMA 试答收口，T=0.6 / top_p=0.95 / top_k=30。",
        "守 = 新 boxed 等于原试答。裂开 = 四条里至少一条不守。",
        f"已打 {len(rows)}。每扇 {n_draw} 条，温度 {temp}。",
        "",
    ]
    lines += block("全体", rows, none_map)
    for kind, zh in (("high", "高把握"), ("mix", "混合"), ("low", "低把握")):
        part = [x for x in rows if x.get("kind") == kind]
        if part:
            lines += block(zh, part, none_map)
    lines += [
        "对窗仍全守、错窗裂开，才是切口。两边都守，是收口锁死。两边都裂，是采样本身不稳。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
