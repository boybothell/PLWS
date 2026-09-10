#!/usr/bin/env python3
"""Viewing table: Full-CoT / PUMA / PLWS plus official DEER.

Official DEER uses a different protocol (greedy, 16k, official prompt, one run).
This file is for side-by-side reading, not the canonical main table.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/reports/fullcot_puma_plws.json"
DEER = ROOT / "results/reports/deer_github_official.json"
TABLE = ROOT / "tables/firstwin_wait/fullcot_puma_plws_deer.md"
OUT = ROOT / "results/reports/fullcot_puma_plws_deer.json"

ORDER = ("MATH", "OlympiadBench", "GPQA-Diamond", "AIME24", "AIME25")
DEER_N = {
    "MATH": 500,
    "OlympiadBench": 675,
    "GPQA-Diamond": 198,
    "AIME24": 30,
    "AIME25": 30,
}


def fmt(acc: float | None, tok: float | None) -> str:
    if acc is None or tok is None:
        return "—"
    return f"{acc:.2f}% / {tok:.0f}"


def main() -> None:
    main = json.loads(MAIN.read_text())
    deer = json.loads(DEER.read_text())
    main_map = {(c["model"], c["dataset"]): c for c in main["cells"]}
    deer_map = {
        (c["model"], c["dataset"]): c
        for c in deer["cells"]
        if c.get("complete")
    }
    lines = [
        "# Full-CoT / PUMA / 窗后压 / 官方 DEER",
        "",
        "并排查看用。主表仍是 `fullcot_puma_plws.md`。",
        "前三列：sampling、约 32k、PUMA prompt、四 seed。",
        "官方 DEER：greedy、16k、官方 prompt、每集 1 次。",
        "Overall 仍是五集等权平均 Acc / 平均 token。DEER 缺一集则 Overall 不报。",
        "",
        "| 模型 | 集 | n（四seed） | Full-CoT（原始 CoT） | PUMA | 窗后压 | n（DEER） | 官方 DEER |",
        "|---|---|---:|---|---|---|---:|---|",
    ]
    rows = []
    for model in ("7B", "8B", "14B"):
        shown = {"full": [], "puma": [], "plws": [], "deer": []}
        deer_ok = True
        for dataset in ORDER:
            cell = main_map[(model, dataset)]
            dcell = deer_map.get((model, dataset))
            d_acc = dcell["acc"] if dcell else None
            d_tok = dcell["tok"] if dcell else None
            if dcell is None:
                deer_ok = False
            shown["full"].append((cell["full"]["acc"], cell["full"]["tok"]))
            shown["puma"].append((cell["puma"]["acc"], cell["puma"]["tok"]))
            shown["plws"].append((cell["plws"]["acc"], cell["plws"]["tok"]))
            if d_acc is not None and d_tok is not None:
                shown["deer"].append((d_acc, d_tok))
            row = {
                "model": model,
                "dataset": dataset,
                "n": cell["n"],
                "full": cell["full"],
                "puma": cell["puma"],
                "plws": cell["plws"],
                "deer_n": DEER_N[dataset],
                "deer": None
                if d_acc is None
                else {"acc": round(d_acc, 2), "tok": round(d_tok)},
            }
            rows.append(row)
            lines.append(
                f"| {model} | {dataset} | {cell['n']} | "
                f"{fmt(cell['full']['acc'], cell['full']['tok'])} | "
                f"{fmt(cell['puma']['acc'], cell['puma']['tok'])} | "
                f"{fmt(cell['plws']['acc'], cell['plws']['tok'])} | "
                f"{DEER_N[dataset]} | {fmt(d_acc, d_tok)} |"
            )
        if deer_ok and len(shown["deer"]) == 5:
            deer_overall = fmt(
                sum(a for a, _ in shown["deer"]) / 5,
                sum(t for _, t in shown["deer"]) / 5,
            )
        else:
            deer_overall = "—"
        lines.append(
            f"| {model} | Overall（等权） | — | "
            f"{sum(a for a, _ in shown['full']) / 5:.2f}% / {sum(t for _, t in shown['full']) / 5:.0f} | "
            f"{sum(a for a, _ in shown['puma']) / 5:.2f}% / {sum(t for _, t in shown['puma']) / 5:.0f} | "
            f"{sum(a for a, _ in shown['plws']) / 5:.2f}% / {sum(t for _, t in shown['plws']) / 5:.0f} | "
            f"— | {deer_overall} |"
        )
    TABLE.write_text("\n".join(lines) + "\n")
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "script": "scripts/report_fullcot_puma_plws_deer.py",
                "table": str(TABLE.relative_to(ROOT)),
                "note": "viewing table; official DEER protocol differs",
                "cells": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("\n".join(lines))
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
