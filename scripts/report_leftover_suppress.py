#!/usr/bin/env python3
"""剩窗压 Wait 一枪：已对还守不守、再等才对改不改口。对照 free / jump。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/leftover_suppress"
JUMP = AE / "results/leftover_jump"
TABLE = AE / "tables/leftover_suppress.md"


def load_rows(folder: Path) -> list[dict[str, Any]]:
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


def line(name: str, xs: list[dict[str, Any]]) -> str:
    if not xs:
        return f"| {name} | 0 | — | — | — | — | — | — |"
    ok = [x for x in xs if x.get("left_ok")]
    wait = [x for x in xs if x.get("wait_helps")]
    ok_keep = sum(1 for x in ok if x.get("keep"))
    wait_chg = sum(1 for x in wait if not x.get("keep"))
    wait_gold = sum(1 for x in wait if x.get("new_gold_ok"))
    ok_hurt = sum(1 for x in ok if not x.get("new_gold_ok"))
    n_wait = sum(1 for x in xs if x.get("has_wait"))
    return (
        f"| {name} | {len(xs)} | {len(ok)}/{len(wait)} | "
        f"{ok_keep}/{len(ok) if ok else 0} ({(ok_keep / len(ok) if ok else 0):.0%}) | "
        f"{wait_chg}/{len(wait) if wait else 0} ({(wait_chg / len(wait) if wait else 0):.0%}) | "
        f"{wait_gold}/{len(wait) if wait else 0} | "
        f"{ok_hurt}/{len(ok) if ok else 0} | "
        f"{n_wait}/{len(xs)} |"
    )


def pair_line(name: str, left: list[dict[str, Any]], right: dict[str, dict[str, Any]]) -> str:
    both = [x for x in left if x["uid"] in right]
    if not both:
        return f"| {name} | 0 | — | — |"
    ok = [x for x in both if x.get("left_ok")]
    wait = [x for x in both if x.get("wait_helps")]
    ok_hurt_extra = sum(
        1
        for x in ok
        if (not x.get("new_gold_ok")) and right[x["uid"]].get("new_gold_ok")
    )
    wait_save_lost = sum(
        1
        for x in wait
        if (not x.get("new_gold_ok")) and right[x["uid"]].get("new_gold_ok")
    )
    return (
        f"| {name} | {len(both)} | "
        f"{ok_hurt_extra}/{len(ok) if ok else 0} | "
        f"{wait_save_lost}/{len(wait) if wait else 0} |"
    )


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "r1_7b"
    suppress = load_rows(ROOT / f"{tag}_s42_suppress")
    free = load_rows(ROOT / f"{tag}_s42_free")
    jump = {x["uid"]: x for x in load_rows(JUMP / f"{tag}_s42")}
    free_by = {x["uid"]: x for x in free}
    lines = [
        "# 剩窗后压 Wait 一枪（7B 低把握四步同答）",
        "",
        "从剩窗前缀再写最多 1024 词，然后贪心探新 boxed。不停在 Wait。",
        "压 Wait：采样时禁止 `Wait` / `Alternatively` / `Hmm`。",
        "不压：同一预算，这些词可以出现。leftover 都是低把握，CGRS 不会压，所以不压 ≈ CGRS。",
        "jump：同一批题上已打过的 ASAG 换路句。",
        "这不是整集 Acc/token。只看：已对还守不守、再等才对改不改口。",
        f"压 Wait 已打 {len(suppress)}，不压已打 {len(free)}。",
        "",
        "| 做法 | 题 | 已对/再等 | 已对仍守 | 再等改口 | 再等新答变对 | 已对新答变错 | 续写里有 Wait |",
        "|---|---:|---|---|---|---|---|---|",
        line("压 Wait", suppress),
        line("不压（≈ CGRS）", free),
        line("同一批 jump", [jump[x["uid"]] for x in suppress if x["uid"] in jump]),
        "",
        "## 成对：压 Wait 比不压多伤了谁",
        "",
        "只看两套都打到的题。多伤已对 = 不压金标仍对、压完金标错。少救再等 = 不压能改对、压完仍错。",
        "",
        "| 对照 | 成对题 | 多伤已对 | 少救再等 |",
        "|---|---:|---|---|",
        pair_line("压 Wait vs 不压", suppress, free_by),
        pair_line("压 Wait vs jump", suppress, jump),
        "",
    ]
    for ds, zh in (("math-500", "MATH"), ("olympiadbench", "奥赛"), ("gpqa-diamond", "GPQA")):
        part = [x for x in suppress if x.get("dataset") == ds]
        fr = [x for x in free if x.get("dataset") == ds]
        if not part and not fr:
            continue
        lines += [
            f"## {zh}",
            "",
            "| 做法 | 题 | 已对/再等 | 已对仍守 | 再等改口 | 再等新答变对 | 已对新答变错 | 续写里有 Wait |",
            "|---|---:|---|---|---|---|---|---|",
            line("压 Wait", part),
            line("不压", fr),
            "",
        ]
    n_ok = sum(1 for x in suppress if x.get("left_ok"))
    n_wait = sum(1 for x in suppress if x.get("wait_helps"))
    ok_hurt = sum(1 for x in suppress if x.get("left_ok") and not x.get("new_gold_ok"))
    wait_gold = sum(1 for x in suppress if x.get("wait_helps") and x.get("new_gold_ok"))
    free_ok_hurt = sum(1 for x in free if x.get("left_ok") and not x.get("new_gold_ok"))
    free_wait_gold = sum(1 for x in free if x.get("wait_helps") and x.get("new_gold_ok"))
    lines += [
        "## 读法",
        "",
        f"压 Wait：已对变错 {ok_hurt}/{n_ok}，再等变对 {wait_gold}/{n_wait}。",
        f"不压：已对变错 {free_ok_hurt}/{sum(1 for x in free if x.get('left_ok'))}，"
        f"再等变对 {free_wait_gold}/{sum(1 for x in free if x.get('wait_helps'))}。",
        "",
        "禁词要生效：不压续写里应大量有 Wait，压完应接近 0。",
        "再等变对明显少于不压 = 挡住改口，方向 1 关。",
        "已对变错明显多于不压 = 把验算砍坏了，方向 1 也关。",
        "两头都差不多 = 短续写上压 Wait 几乎等于不压，还要看整集 token。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
