#!/usr/bin/env python3
"""Write the Feishu Excel copy of the Full-CoT / PUMA / DEER / 窗后压 table.

Each method cell bolds the highest Acc and the lowest token among Full-CoT,
PUMA, DEER, and 窗后压. Empty DEER cells are skipped.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError:  # unittest / okay-budget venv has no openpyxl
    Workbook = CellRichText = TextBlock = InlineFont = None
    Alignment = Border = Font = PatternFill = Side = get_column_letter = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "reports" / "fullcot_puma_plws.json"
OUT = ROOT / "tables" / "firstwin_wait" / "fullcot_puma_plws_feishu.xlsx"

NAMES = {
    "7B": "DeepSeek-R1-Distill-Qwen-7B",
    "Nemotron": "Llama-3.1-Nemotron-Nano-8B-v1",
    "14B": "DeepSeek-R1-Distill-Qwen-14B",
    "4B": "Qwen3-4B",
    "8B": "Qwen3-8B",
}
ORDER = ["7B", "Nemotron", "14B", "4B", "8B"]
DS_ORDER = ["MATH", "OlympiadBench", "GPQA-Diamond", "AIME24", "AIME25"]
EXTRA_DS_ORDER = ["AMC23"]
METHODS = ("full", "puma", "deer", "plws")
COMPARE = ("full", "puma", "deer", "plws")
HEADERS = ["模型", "集", "n", "Full-CoT (原始 CoT)", "PUMA", "DEER", "窗后压"]

if InlineFont is not None:
    BOLD_IF = InlineFont(b=True, sz=11)
    NORM_IF = InlineFont(b=False, sz=11)
    HEAD_FONT = Font(bold=True, size=11)
    BODY_FONT = Font(size=11)
    OVER_FILL = PatternFill("solid", fgColor="FFF3E8")
    HEAD_FILL = PatternFill("solid", fgColor="FFE8EEF4")
    THIN = Border(
        left=Side(style="thin", color="FFD0D0D0"),
        right=Side(style="thin", color="FFD0D0D0"),
        top=Side(style="thin", color="FFD0D0D0"),
        bottom=Side(style="thin", color="FFD0D0D0"),
    )
    CENTER = Alignment(horizontal="center", vertical="center")
    LEFT = Alignment(horizontal="left", vertical="center")
else:
    BOLD_IF = NORM_IF = HEAD_FONT = BODY_FONT = None
    OVER_FILL = HEAD_FILL = THIN = CENTER = LEFT = None


def acc_s(acc: float) -> str:
    return f"{acc:.2f}%"


def tok_s(tok: float) -> str:
    return str(int(round(tok)))


def rich_cell(acc: float, tok: float, acc_win: bool, tok_win: bool) -> CellRichText:
    return CellRichText(
        TextBlock(BOLD_IF if acc_win else NORM_IF, acc_s(acc)),
        TextBlock(NORM_IF, " / "),
        TextBlock(BOLD_IF if tok_win else NORM_IF, tok_s(tok)),
    )


def winners(parts: dict) -> tuple[set[str], set[str]]:
    keys = [key for key in COMPARE if parts.get(key)]
    if not keys:
        return set(), set()
    accs = {key: parts[key]["acc"] for key in keys}
    toks = {key: parts[key]["tok"] for key in keys}
    max_acc = max(accs.values())
    min_tok = min(toks.values())
    return (
        {key for key, value in accs.items() if value == max_acc},
        {key for key, value in toks.items() if value == min_tok},
    )


def method_value(parts: dict, key: str, acc_w: set[str], tok_w: set[str]):
    cell = parts.get(key)
    if cell is None:
        return "", ""
    return (
        rich_cell(cell["acc"], cell["tok"], key in acc_w, key in tok_w),
        f"{acc_s(cell['acc'])} / {tok_s(cell['tok'])}",
    )


def put_row(ws, widths, r, model, dataset, n, parts, overall=False) -> None:
    acc_w, tok_w = winners(parts)
    values = [model, dataset, n]
    shown = [model, dataset, str(n)]
    for key in METHODS:
        value, text = method_value(parts, key, acc_w, tok_w)
        values.append(value)
        shown.append(text)
    for col, value in enumerate(values, 1):
        cell = ws.cell(r, col, value)
        cell.font = BODY_FONT
        cell.border = THIN
        cell.alignment = LEFT if col <= 2 else CENTER
        if overall:
            cell.fill = OVER_FILL
        widths[col - 1] = max(widths[col - 1], len(shown[col - 1]))


def mean_cell(rows: list[tuple[float, float]]) -> dict:
    return {
        "acc": sum(acc for acc, _ in rows) / len(rows),
        "tok": sum(tok for _, tok in rows) / len(rows),
    }


def main() -> None:
    if Workbook is None:
        raise SystemExit("openpyxl is required to write the Feishu xlsx")
    payload = json.loads(SRC.read_text())
    by_model: dict[str, list] = defaultdict(list)
    for row in payload["cells"]:
        by_model[row["model"]].append(row)

    wb = Workbook()
    ws = wb.active
    ws.title = "Full-CoT PUMA DEER 窗后压"
    ws.append(HEADERS)
    for col, _ in enumerate(HEADERS, 1):
        cell = ws.cell(1, col)
        cell.font = HEAD_FONT
        cell.fill = HEAD_FILL
        cell.alignment = CENTER
        cell.border = THIN

    widths = [len(header) for header in HEADERS]
    r = 2
    for model in ORDER:
        rows = {item["dataset"]: item for item in by_model[model]}
        shown = {key: [] for key in METHODS}
        name = NAMES[model]
        for dataset in (*DS_ORDER, *EXTRA_DS_ORDER):
            item = rows.get(dataset)
            if item is None:
                continue
            put_row(ws, widths, r, name, item["dataset"], item["n"], item)
            for key in METHODS:
                if item.get(key):
                    shown[key].append((item[key]["acc"], item[key]["tok"]))
            r += 1
        overall = {
            key: mean_cell(shown[key]) if key != "deer" or len(shown[key]) == 5 else None
            for key in METHODS
        }
        put_row(ws, widths, r, name, "Overall（等权）", "—", overall, overall=True)
        r += 1

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:G{r - 1}"
    ws.row_dimensions[1].height = 22
    for i, width in enumerate(widths, 1):
        factor = 1.2 if i == 1 else 1.15
        ws.column_dimensions[get_column_letter(i)].width = min(
            48, max(10, width * factor + 2)
        )
    wb.save(OUT)
    print(f"wrote {OUT}")
    print(f"rows {r - 1}")


if __name__ == "__main__":
    main()
