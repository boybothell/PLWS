#!/usr/bin/env python3
"""剩窗 jump 一枪：拆没拆已对、假平台改没改口。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/leftover_jump"
TABLE = AE / "tables/leftover_jump.md"
PEAK = 0.50


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
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def peaked(row: dict[str, Any]) -> bool | None:
    ent = fin(row.get("next_ent_mean"))
    if ent != ent:
        return None
    return ent < PEAK


def line(name: str, xs: list[dict[str, Any]]) -> str:
    if not xs:
        return f"| {name} | 0 | — | — | — | — | — |"
    ok = [x for x in xs if x.get("left_ok")]
    wait = [x for x in xs if x.get("wait_helps")]
    ok_keep = sum(1 for x in ok if x.get("keep"))
    wait_chg = sum(1 for x in wait if not x.get("keep"))
    wait_gold = sum(1 for x in wait if x.get("new_gold_ok"))
    ok_hurt = sum(1 for x in ok if not x.get("new_gold_ok"))
    return (
        f"| {name} | {len(xs)} | {len(ok)}/{len(wait)} | "
        f"{ok_keep}/{len(ok) if ok else 0} ({(ok_keep / len(ok) if ok else 0):.0%}) | "
        f"{wait_chg}/{len(wait) if wait else 0} ({(wait_chg / len(wait) if wait else 0):.0%}) | "
        f"{wait_gold}/{len(wait) if wait else 0} | "
        f"{ok_hurt}/{len(ok) if ok else 0} |"
    )


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "r1_7b"
    rows = load_rows(ROOT / f"{tag}_s42")
    peak = [x for x in rows if peaked(x) is True]
    dull = [x for x in rows if peaked(x) is False]
    unknown = [x for x in rows if peaked(x) is None]
    lines = [
        "# 剩窗换路一枪（低把握同答 + ASAG jump）",
        "",
        "7B 第一扇低把握四步同答。插入 ASAG 换路句，再采样写到下一个 Wait / `</think>` / 1024 词，然后贪心探新 boxed。",
        "方法运行时只跳「下一步思路熵 < 0.50」的窗；这里全体低把握都跳了，事后按熵切开。",
        "这不是整集 Acc/token。只看：已对还守不守、再等才对改不改口。",
        f"已打 {len(rows)}。有下一步熵 {len(peak) + len(dull)}（尖 {len(peak)} / 不尖 {len(dull)}）。MATH 大多没有熵，算未知。",
        "",
        "| 切片 | 题 | 已对/再等 | 已对仍守 | 再等改口 | 再等新答变对 | 已对新答变错 |",
        "|---|---:|---|---|---|---|---|",
        line("全体低把握都跳", rows),
        line("下一步尖才跳（方法）", peak),
        line("下一步不尖（不该跳）", dull),
        line("无熵（几乎都是 MATH）", unknown),
        "",
    ]
    for ds, zh in (("math-500", "MATH"), ("olympiadbench", "奥赛"), ("gpqa-diamond", "GPQA")):
        part = [x for x in rows if x.get("dataset") == ds]
        if part:
            lines.append(f"## {zh}")
            lines.append("")
            lines.append("| 切片 | 题 | 已对/再等 | 已对仍守 | 再等改口 | 再等新答变对 | 已对新答变错 |")
            lines.append("|---|---:|---|---|---|---|---|")
            lines.append(line("全体低把握", part))
            lines.append(line("尖才跳", [x for x in part if peaked(x) is True]))
            lines.append(line("不尖", [x for x in part if peaked(x) is False]))
            lines.append("")
    lines += [
        "方法能活：尖才跳时，已对仍守要高，再等改口（尤其变对）要明显高于不跳的自然改口。",
        "已对新答变错一多，这扇门就死，不要再调 jump 句子。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
