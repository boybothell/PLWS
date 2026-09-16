#!/usr/bin/env python3
"""Post-window line share when the marker may appear anywhere on the line.

Same window split and nonempty-line denominator as report_fullcot_line_start.py.
Numerator is lines that contain the word, not only line-start hits.
Only ATP-semantic markers: pause, new path, contrast, hedge, verify, wrap.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.artifacts import load_jsonl
from plws.matrix import FIRSTWIN
from plws.paths import PLWSPaths
from report_fullcot_line_start import (
    DATASETS,
    MODELS,
    SEEDS,
    nonempty_lines,
    reasoning_path,
    split_after_thought,
    texts_from_answers,
)

TABLE = ROOT / "tables" / "firstwin_wait" / "fullcot_anywhere.md"
REPORT = ROOT / "results" / "reports" / "fullcot_anywhere.json"
XLSX = ROOT / "tables" / "firstwin_wait" / "fullcot_anywhere_feishu.xlsx"

# Pause / new path / contrast / hedge / verify / wrap. Not function words.
ATP_MARKERS = (
    ("Wait", r"wait\b"),
    ("Alternatively", r"alternatively\b"),
    ("Hmm", r"hmm+\b|hm\b"),
    ("But", r"but\b"),
    ("Let me", r"let me\b"),
    ("So", r"so\b"),
    ("Therefore", r"therefore\b"),
    ("However", r"however\b"),
    ("Maybe", r"maybe\b"),
    ("Perhaps", r"perhaps\b"),
    ("Hold on", r"hold on\b"),
    ("Actually", r"actually\b"),
    ("But let", r"but let\b"),
    ("But maybe", r"but maybe\b"),
    ("Let me check", r"let me (?:check|verify|confirm|double-check)"),
    ("Let me think", r"let me think"),
    ("Let's", r"let's\b"),
    ("double-check", r"double[- ]check\b"),
    ("another way", r"another (?:way|approach|method)\b"),
    ("Thus", r"thus\b"),
    ("Instead", r"instead\b"),
    ("等一下", r"等一下"),
    ("换一种", r"换一种|换个(?:思路|方法)"),
)
SHOW = tuple(name for name, _pat in ATP_MARKERS)
PINNED = (
    "Wait",
    "Alternatively",
    "Hmm",
    "But",
    "Let me",
    "So",
    "Therefore",
)
COL_ORDER = ("14B", "7B", "Nemotron", "4B")
NAMES = {
    "14B": "DeepSeek-R1-Distill-Qwen-14B",
    "7B": "DeepSeek-R1-Distill-Qwen-7B",
    "Nemotron": "Llama-3.1-Nemotron-Nano-8B-v1",
    "4B": "Qwen3-4B",
}
LEADING = r"^(?:[#>*\-\s]*)(?:\*\*)?"
START = {name: re.compile(LEADING + pat, re.I) for name, pat in ATP_MARKERS}
ANY = {name: re.compile(pat, re.I) for name, pat in ATP_MARKERS}


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def analyze() -> dict:
    paths = PLWSPaths.discover(__file__)
    stats = {
        zh: {
            "post_questions": 0,
            "post_lines": 0,
            "start": Counter(),
            "contain": Counter(),
        }
        for _, zh in MODELS
    }
    missing: list[str] = []
    for model, zh in MODELS:
        cell = stats[zh]
        for dataset in DATASETS:
            for seed in SEEDS:
                answers = reasoning_path(paths, model, dataset, seed)
                if answers is None:
                    missing.append(f"{model}:{dataset}:{seed}")
                    continue
                qmap = {
                    question: text for question, text in texts_from_answers(answers)
                }
                jobs_path = paths.jobs_path(
                    model, dataset, seed, FIRSTWIN, k=4, lexicon="core"
                )
                if not jobs_path.is_file():
                    continue
                for job in load_jsonl(jobs_path):
                    text = qmap.get((job.get("question") or "").strip())
                    if text is None:
                        continue
                    post = split_after_thought(job.get("thought") or "", text)
                    if post is None or not post.strip():
                        continue
                    lines = nonempty_lines(post)
                    if not lines:
                        continue
                    cell["post_questions"] += 1
                    cell["post_lines"] += len(lines)
                    for line in lines:
                        for name in SHOW:
                            if START[name].search(line):
                                cell["start"][name] += 1
                            if ANY[name].search(line):
                                cell["contain"][name] += 1
    return {"stats": stats, "missing": missing}


def _table(stats: dict, key: str) -> list[str]:
    headers = ["词", *[NAMES[zh] for zh in COL_ORDER]]
    lines = [
        f"| {' | '.join(headers)} |",
        f"|{'|'.join(['---'] + ['---:' for _ in COL_ORDER])}|",
    ]
    for name in SHOW:
        cells = [name]
        for zh in COL_ORDER:
            cells.append(f"{pct(stats[zh][key][name], stats[zh]['post_lines']):.2f}%")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_table(stats: dict) -> list[str]:
    lines = [
        "# 窗后行内含词占行比例",
        "",
        "与段首占行同一套窗后非空行。分母是第一次同答窗之后按 `\\n` 切开、去掉空行的行数。",
        "分子是这些行里**任意位置**出现该词的行数，不要求段首。",
        "只收有 ATP 语义的词：停顿、改路、转折、试探、验算、收口。虚词和公式残留不进。",
        "CORE 仍是整段硬禁，本表只是对照。",
        "",
        *_table(stats, "contain"),
        "",
        "## 对照：同一分母上的段首占行",
        "",
        *_table(stats, "start"),
        "",
        "覆盖：",
    ]
    for zh in COL_ORDER:
        cell = stats[zh]
        lines.append(
            f"- {zh}：窗后题 {cell['post_questions']}，非空行 {cell['post_lines']}"
        )
    return lines


def write_xlsx(stats: dict) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head = Font(bold=True, size=11)
    fill = PatternFill("solid", fgColor="FFE8EEF4")
    pin = PatternFill("solid", fgColor="FFF3E8")
    thin = Border(
        left=Side(style="thin", color="FFD0D0D0"),
        right=Side(style="thin", color="FFD0D0D0"),
        top=Side(style="thin", color="FFD0D0D0"),
        bottom=Side(style="thin", color="FFD0D0D0"),
    )
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    headers = ["词", *[NAMES[zh] for zh in COL_ORDER]]

    def fill_sheet(ws, key: str) -> None:
        ws.append(headers)
        for name in SHOW:
            ws.append(
                [
                    name,
                    *[
                        f"{pct(stats[zh][key][name], stats[zh]['post_lines']):.2f}%"
                        for zh in COL_ORDER
                    ],
                ]
            )
        for col, _ in enumerate(headers, 1):
            cell = ws.cell(1, col)
            cell.font = head
            cell.fill = fill
            cell.alignment = center
            cell.border = thin
        for ridx, row in enumerate(
            ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=5), 1
        ):
            for i, cell in enumerate(row):
                cell.border = thin
                cell.alignment = left if i == 0 else center
                if ridx <= len(PINNED):
                    cell.fill = pin
        ws.column_dimensions["A"].width = 16
        for i in range(2, 6):
            ws.column_dimensions[get_column_letter(i)].width = 32
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:E{ws.max_row}"

    ws = wb.active
    ws.title = "行内含词ATP"
    fill_sheet(ws, "contain")
    other = wb.create_sheet("段首占行ATP")
    fill_sheet(other, "start")
    wb.save(XLSX)


def main() -> None:
    got = analyze()
    if got["missing"]:
        raise SystemExit("missing " + ", ".join(got["missing"]))
    stats = got["stats"]
    lines = write_table(stats)
    TABLE.write_text("\n".join(lines) + "\n")
    REPORT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "script": "scripts/report_fullcot_anywhere.py",
                "denominator": "post-window nonempty lines",
                "numerator": "lines containing the word anywhere",
                "words": list(SHOW),
                "pinned": list(PINNED),
                "models": {
                    zh: {
                        "post_questions": stats[zh]["post_questions"],
                        "post_lines": stats[zh]["post_lines"],
                        "start_pct": {
                            name: pct(stats[zh]["start"][name], stats[zh]["post_lines"])
                            for name in SHOW
                        },
                        "contain_pct": {
                            name: pct(
                                stats[zh]["contain"][name], stats[zh]["post_lines"]
                            )
                            for name in SHOW
                        },
                    }
                    for zh in COL_ORDER
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    write_xlsx(stats)
    print("\n".join(lines))
    print(f"wrote {TABLE}")
    print(f"wrote {REPORT}")
    print(f"wrote {XLSX}")


if __name__ == "__main__":
    main()
