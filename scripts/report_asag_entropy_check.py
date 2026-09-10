#!/usr/bin/env python3
"""Does ASAG ΔH drop at the first correct PUMA-step probe?"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
OUT = AE / "tables/asag_entropy_check.md"
SCORE_DIR = AE / "results/asag_entropy"
KINDS = (("gold", "写对金标的题，第一次试答对上金标"), ("afinal", "写对金标的题，第一次试答等于写完终答"), ("wrong", "写完就错的题，用最后一步冒充对上"))


def load_scores(stem: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(SCORE_DIR.glob(f"{stem}_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("attn_H") == row.get("attn_H"):
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], kind: str, field: str) -> dict[str, Any] | None:
    by: dict[int, dict[float, float]] = defaultdict(dict)
    for row in rows:
        if row.get("kind") != kind:
            continue
        val = row.get(field)
        if val is None or val != val:
            continue
        by[int(row["question_idx"])][float(row["frac"])] = float(val)
    deltas: dict[float, list[float]] = defaultdict(list)
    n = 0
    for frac_map in by.values():
        h1 = frac_map.get(0.2)
        if h1 is None or abs(h1) < 1e-8:
            continue
        if 1.0 not in frac_map:
            continue
        if abs(frac_map[1.0] - h1) < 1e-12 and len(frac_map) < 2:
            continue
        n += 1
        for frac, h in frac_map.items():
            if frac == 0.2:
                continue
            deltas[frac].append((h - h1) / h1)
    if n == 0:
        return None
    out = {"n": n}
    for frac in (0.4, 0.6, 0.8, 1.0):
        xs = deltas.get(frac) or []
        if not xs:
            out[f"med_{frac}"] = float("nan")
            out[f"lt01_{frac}"] = float("nan")
            out[f"n_{frac}"] = 0
            continue
        xs = sorted(xs)
        out[f"med_{frac}"] = xs[len(xs) // 2]
        out[f"lt01_{frac}"] = sum(x < -0.1 for x in xs) / len(xs)
        out[f"n_{frac}"] = len(xs)
    return out


def pct(x: float) -> str:
    return "—" if x != x else f"{100.0 * x:.0f}%"


def med(x: float) -> str:
    return "—" if x != x else f"{x:+.3f}"


def main() -> None:
    lines = [
        "# PUMA 分步上，注意力熵会不会在对上时掉下去",
        "",
        "对照 ASAG：写完是对的题里，第一次中间试答写对时记 Hf，再取这次思考长度的 20/40/60/80% 最近的 PUMA 探点。",
        "ΔH = (该点熵 − 20% 那点的熵) / 20% 那点的熵。论文说对上时超过七成 ΔH < −0.1，更早的点很少。",
        "熵 = 最后 4 层、所有头，查询窗 = 试答 boxed 和等长的最近思路。7B，密探分步。",
        "",
    ]
    for stem, title in (
        ("r1_7b_math-500_s42", "7B MATH"),
        ("r1_7b_gpqa-diamond_s42", "7B GPQA"),
    ):
        rows = load_scores(stem)
        lines += [f"## {title}", "", f"已抽出 {len(rows)} 个探点。", ""]
        print(title, f"rows={len(rows)}", flush=True)
        for field, suffix in (("attn_H", "论文加总H"), ("attn_H_mean", "按查询均值H")):
            lines += [f"### {suffix}", ""]
            lines += [
                "| 哪一类题 | 题数 | 40% 中位ΔH / 掉过−0.1 | 60% | 80% | 第一次对上 |",
                "|---|---:|---|---|---|---|",
            ]
            for kind, why in KINDS:
                rec = summarize(rows, kind, field)
                if rec is None:
                    row = f"| {why} | 0 | 未齐 | 未齐 | 未齐 | 未齐 |"
                else:
                    row = (
                        f"| {why} | {rec['n']} "
                        f"| {med(rec['med_0.4'])} / {pct(rec['lt01_0.4'])} "
                        f"| {med(rec['med_0.6'])} / {pct(rec['lt01_0.6'])} "
                        f"| {med(rec['med_0.8'])} / {pct(rec['lt01_0.8'])} "
                        f"| {med(rec['med_1.0'])} / {pct(rec['lt01_1.0'])} |"
                    )
                print(row, flush=True)
                lines.append(row)
            lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"写成 {OUT}", flush=True)


if __name__ == "__main__":
    main()
