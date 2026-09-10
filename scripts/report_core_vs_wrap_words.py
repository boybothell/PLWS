#!/usr/bin/env python3
"""Why CORE three words beat But / So / Therefore.

Causal proof from official Full-CoT vs CORE and wrap-sample CORE+wrap vs
CORE. Offline next-probe lifts explain the original shortlist; they are
not the proof. Function-label restart rates are listed only as a
rejected metric.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "reports" / "fullcot_puma_plws.json"
WRAP = ROOT / "results" / "reports" / "wrap_sample_vs_core.json"
LEX = ROOT / "results" / "reports" / "firstwin_postwindow_lex.json"
LINES = ROOT / "results" / "reports" / "fullcot_line_start.json"
OUT = ROOT / "results" / "reports" / "core_vs_wrap_words.json"
TABLE = ROOT / "tables" / "firstwin_wait" / "core_vs_wrap_words.md"
TZ = ZoneInfo("Asia/Shanghai")
MODELS = ("7B", "Nemotron", "14B", "4B")
LARGE = ("MATH", "OlympiadBench", "GPQA-Diamond")
WORDS = ("Wait", "Alternatively", "Hmm", "But")
CLOSERS = ("So", "Therefore")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pct(x: float | None) -> str:
    return "" if x is None else f"{x:.2f}%"


def pp(x: float | None) -> str:
    return "" if x is None else f"{x:+.2f}"


def tok(x: float | None) -> str:
    return "" if x is None else f"{x:.0f}"


def dtok(x: float | None) -> str:
    return "" if x is None else f"{x:+.0f}"


def lex_row(rows: list[dict[str, Any]], word: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("word") == word:
            return row
    return None


def official_slices(main: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in MODELS}
    for cell in main["cells"]:
        model = cell["model"]
        if model not in by_model:
            continue
        full = cell["full"]
        plws = cell["plws"]
        row = {
            "model": model,
            "dataset": cell["dataset"],
            "n": cell["n"],
            "full_acc": full["acc"],
            "core_acc": plws["acc"],
            "d_acc": plws["acc"] - full["acc"],
            "full_tok": full["tok"],
            "core_tok": plws["tok"],
            "d_tok": plws["tok"] - full["tok"],
        }
        by_model[model].append(row)
        out.append(row)
    for model, rows in by_model.items():
        for name, pred in (
            ("MATH / Olympiad / GPQA", lambda r: r["dataset"] in LARGE),
            ("五集 Overall", lambda r: True),
        ):
            picked = [r for r in rows if pred(r)]
            if not picked:
                continue
            out.append(
                {
                    "model": model,
                    "dataset": name,
                    "n": sum(r["n"] for r in picked),
                    "full_acc": mean([r["full_acc"] for r in picked]),
                    "core_acc": mean([r["core_acc"] for r in picked]),
                    "d_acc": mean([r["d_acc"] for r in picked]),
                    "full_tok": mean([r["full_tok"] for r in picked]),
                    "core_tok": mean([r["core_tok"] for r in picked]),
                    "d_tok": mean([r["d_tok"] for r in picked]),
                    "slice": True,
                }
            )
    return out


def pool_wrap(cells: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(int(c["n"]) for c in cells)
    questions = sum(int(c["n_questions"]) for c in cells)
    beat = sum(int(c["wrap_beat_core"]) for c in cells)
    lose = sum(int(c["wrap_lose_core"]) for c in cells)
    core_ok = sum(c["n"] * c["core_acc"] / 100.0 for c in cells)
    wrap_ok = sum(c["n"] * c["wrap_acc"] / 100.0 for c in cells)
    full_ok = sum(c["n"] * c["full_acc"] / 100.0 for c in cells)
    return {
        "n": n,
        "n_questions": questions,
        "full_acc": 100.0 * full_ok / n if n else None,
        "core_acc": 100.0 * core_ok / n if n else None,
        "wrap_acc": 100.0 * wrap_ok / n if n else None,
        "d_acc_wrap_minus_core": 100.0 * (beat - lose) / n if n else None,
        "d_acc_core_minus_full": (
            100.0 * (core_ok - full_ok) / n if n else None
        ),
        "core_tok": (
            sum(c["n"] * c["core_tok"] for c in cells) / n if n else None
        ),
        "wrap_tok": (
            sum(c["n"] * c["wrap_tok"] for c in cells) / n if n else None
        ),
        "d_tok_wrap_minus_core": (
            sum(c["n"] * c["d_tok_wrap_minus_core"] for c in cells) / n
            if n
            else None
        ),
        "core_cont": (
            sum(c["n"] * c["core_cont"] for c in cells) / n if n else None
        ),
        "wrap_cont": (
            sum(c["n"] * c["wrap_cont"] for c in cells) / n if n else None
        ),
        "d_cont_wrap_minus_core": (
            sum(c["n"] * c["d_cont_wrap_minus_core"] for c in cells) / n
            if n
            else None
        ),
        "wrap_beat_core": beat,
        "wrap_lose_core": lose,
    }


def pool_reason(cells: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    leaves = [leaf for cell in cells if (leaf := cell.get("by_reason", {}).get(reason))]
    n = sum(int(leaf["n"]) for leaf in leaves)
    beat = sum(int(leaf["wrap_beat_core"]) for leaf in leaves)
    lose = sum(int(leaf["wrap_lose_core"]) for leaf in leaves)
    core_ok = sum(leaf["n"] * leaf["core_acc"] / 100.0 for leaf in leaves)
    wrap_ok = sum(leaf["n"] * leaf["wrap_acc"] / 100.0 for leaf in leaves)
    return {
        "reason": reason,
        "n": n,
        "core_acc": 100.0 * core_ok / n if n else None,
        "wrap_acc": 100.0 * wrap_ok / n if n else None,
        "d_acc_wrap_minus_core": 100.0 * (beat - lose) / n if n else None,
        "core_tok": (
            sum(leaf["n"] * leaf["core_tok"] for leaf in leaves) / n if n else None
        ),
        "wrap_tok": (
            sum(leaf["n"] * leaf["wrap_tok"] for leaf in leaves) / n if n else None
        ),
        "d_tok_wrap_minus_core": (
            sum(leaf["n"] * leaf["d_tok_wrap_minus_core"] for leaf in leaves) / n
            if n
            else None
        ),
        "wrap_beat_core": beat,
        "wrap_lose_core": lose,
    }


def offline_lifts(lex: dict[str, Any]) -> list[dict[str, Any]]:
    summary = lex["summary"]
    ruin = lex["conditional_outcomes"]["correct_window_corruption"]
    save = lex["conditional_outcomes"]["wrong_window_rescue"]
    rows: list[dict[str, Any]] = []
    for word in WORDS:
        all_row = lex_row(summary, word)
        ruin_row = lex_row(ruin, word)
        save_row = lex_row(save, word)
        if not all_row:
            continue
        rows.append(
            {
                "word": word,
                "in_probe": True,
                "n": all_row["n"],
                "present": all_row["present"],
                "prevalence": all_row["prevalence"],
                "change_lift": all_row["lift"],
                "ruin_lift": None if ruin_row is None else ruin_row["corrupt_lift_vs_all"],
                "save_lift": None if save_row is None else save_row["rescue_lift_vs_all"],
            }
        )
    return rows


def closer_stats(lines: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for word in CLOSERS:
        post: list[float] = []
        plain: list[float] = []
        for model in MODELS:
            rec = lines.get("models", {}).get(model, {}).get("words", {}).get(word)
            if not rec:
                continue
            post.append(rec["post_line_pct"])
            plain.append(rec["labels"]["plain"])
        rows.append(
            {
                "word": word,
                "in_probe": False,
                "post_line_pct_models": post,
                "post_line_pct": mean(post),
                "plain_pct": mean(plain),
            }
        )
    return rows


def rejected_labels(lines: dict[str, Any]) -> list[dict[str, Any]]:
    words = ("Wait", "Alternatively", "Hmm", "But", "So", "Therefore")
    rows: list[dict[str, Any]] = []
    for word in words:
        plains = []
        for model in MODELS:
            rec = lines.get("models", {}).get(model, {}).get("words", {}).get(word)
            if rec:
                plains.append(rec["labels"]["plain"])
        rows.append({"word": word, "plain_pct": mean(plains)})
    return rows


def official_line(row: dict[str, Any]) -> str:
    return (
        f"| {row['model']} | {row['dataset']} | {row['n']} | "
        f"{pct(row['full_acc'])} | {pct(row['core_acc'])} | "
        f"{pp(row['d_acc'])} | {tok(row['full_tok'])} | "
        f"{tok(row['core_tok'])} | {dtok(row['d_tok'])} |"
    )


def wrap_line(label: str, rec: dict[str, Any]) -> str:
    return (
        f"| {label} | {rec['n']} | {rec.get('n_questions', '')} | "
        f"{pct(rec['core_acc'])} | {pct(rec['wrap_acc'])} | "
        f"{pp(rec['d_acc_wrap_minus_core'])} | {tok(rec['core_tok'])} | "
        f"{tok(rec['wrap_tok'])} | {dtok(rec['d_tok_wrap_minus_core'])} | "
        f"{rec['wrap_beat_core']} | {rec['wrap_lose_core']} |"
    )


def main() -> None:
    main_rep = load(MAIN)
    wrap_rep = load(WRAP)
    lex = load(LEX)
    lines = load(LINES)
    official = official_slices(main_rep)
    wrap_cells = wrap_rep["cells"]
    wrap_pending = wrap_rep.get("pending") or []
    wrap_all = pool_wrap(wrap_cells)
    wrap_models = {
        zh: pool_wrap([c for c in wrap_cells if c["model_zh"] == zh])
        for zh in sorted({c["model_zh"] for c in wrap_cells})
    }
    wrap_reasons = {
        reason: pool_reason(wrap_cells, reason)
        for reason in (
            "flip",
            "keep_right_long",
            "keep_right_short",
            "keep_wrong",
        )
    }
    lifts = offline_lifts(lex)
    closers = closer_stats(lines)
    rejected = rejected_labels(lines)
    generated = datetime.now(TZ).isoformat(timespec="seconds")
    payload = {
        "generated_at": generated,
        "script": "scripts/report_core_vs_wrap_words.py",
        "table": str(TABLE.relative_to(ROOT)),
        "claim": (
            "Ban Wait/Alternatively/Hmm after the first window. "
            "Do not also ban But/So/Therefore."
        ),
        "proof": "paired generation Acc/token, not line-start function labels",
        "official": official,
        "wrap": {
            "all": wrap_all,
            "by_model": wrap_models,
            "by_reason": wrap_reasons,
            "cells": [
                {
                    "model": c["model_zh"],
                    "dataset": c["dataset_zh"],
                    "n": c["n"],
                    "n_questions": c["n_questions"],
                    "core_acc": c["core_acc"],
                    "wrap_acc": c["wrap_acc"],
                    "d_acc_wrap_minus_core": c["d_acc_wrap_minus_core"],
                    "core_tok": c["core_tok"],
                    "wrap_tok": c["wrap_tok"],
                    "d_tok_wrap_minus_core": c["d_tok_wrap_minus_core"],
                    "wrap_beat_core": c["wrap_beat_core"],
                    "wrap_lose_core": c["wrap_lose_core"],
                }
                for c in wrap_cells
            ],
            "pending": wrap_pending,
        },
        "offline_lifts": lifts,
        "closers": closers,
        "rejected_plain": rejected,
        "sources": {
            "official": str(MAIN.relative_to(ROOT)),
            "wrap": str(WRAP.relative_to(ROOT)),
            "lex": str(LEX.relative_to(ROOT)),
            "line_start": str(LINES.relative_to(ROOT)),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    large_rows = [r for r in official if r.get("dataset") == "MATH / Olympiad / GPQA"]
    overall_rows = [r for r in official if r.get("dataset") == "五集 Overall"]
    cell_rows = [r for r in official if not r.get("slice")]
    large_dacc = mean([r["d_acc"] for r in large_rows])
    large_dtok = mean([r["d_tok"] for r in large_rows])
    wrap_md = wrap_all

    lines_out = [
        "# CORE 三词为什么优于 But / So / Therefore",
        "",
        "要证明的是：第一扇同答窗后该硬禁 `Wait / Alternatively / Hmm`，",
        "不该再禁 `But / So / Therefore`。",
        "**证明是同题配对生成的 Acc 和 token**，不是段首功能分类。",
        f"由 `scripts/report_core_vs_wrap_words.py` 从规范报告重算。生成 {generated}。",
        "",
        "## 结论",
        "",
        "| 干预 | 对照 | Acc | token | 处置 |",
        "|---|---|---|---|---|",
        (
            f"| 压 CORE 三词 | 官方全量 Full-CoT，"
            f"MATH/Olympiad/GPQA 等权 | "
            f"四模型平均 {pp(large_dacc)} | "
            f"四模型平均 {dtok(large_dtok)} | 留下 |"
        ),
        (
            f"| 再压 But/So/Therefore | 同前缀 wrap-sample，"
            f"已齐 {wrap_md['n']} jobs | "
            f"{pp(wrap_md['d_acc_wrap_minus_core'])} | "
            f"{dtok(wrap_md['d_tok_wrap_minus_core'])} | 丢掉 |"
        ),
        "",
        "大集上压 CORE 基本持平或略升 Acc，并砍 token。",
        "同一前缀再压收口词，纠 CORE 的题少于弄坏 CORE 的题，token 几乎不省。",
        "",
        "## 1. 因果：压 CORE（官方全量）",
        "",
        "窗后压 = 思考阶段整段硬禁 CORE 三词。对照是同一条 Full-CoT。",
        "数字来自 [`fullcot_puma_plws.md`](fullcot_puma_plws.md)，四 seed。",
        "Overall 是数据集等权，不是题数加权。",
        "",
        "### 1.1 三个大集（MATH / Olympiad / GPQA）",
        "",
        "AIME 窗早、错锁多，是已知例外，先从大集看 CORE 该不该留下。",
        "",
        "| 模型 | 切片 | n | Full Acc | CORE Acc | ΔAcc | Full tok | CORE tok | Δtok |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in large_rows:
        lines_out.append(official_line(row))
    lines_out.extend(
        [
            "",
            "### 1.2 五集 Overall",
            "",
            "AIME 把 Overall Acc 拉下来；token 仍短。正式叙事仍以大集为准。",
            "",
            "| 模型 | 切片 | n | Full Acc | CORE Acc | ΔAcc | Full tok | CORE tok | Δtok |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overall_rows:
        lines_out.append(official_line(row))
    lines_out.extend(
        [
            "",
            "### 1.3 逐格",
            "",
            "| 模型 | 集 | n | Full Acc | CORE Acc | ΔAcc | Full tok | CORE tok | Δtok |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cell_rows:
        lines_out.append(official_line(row))

    lines_out.extend(
        [
            "",
            "## 2. 因果：再压 But / So / Therefore（同前缀）",
            "",
            "词表 `core_plus_but_so_therefore`。前缀、题、窗与 CORE 相同。",
            "样本是翻转题加对照，**不是官方全量**，不进主表。",
            "Olympiad 未齐，不进本表。明细见",
            "[`wrap_sample_vs_core.md`](wrap_sample_vs_core.md)。",
            "",
            "### 2.1 已齐合计",
            "",
            "| 切片 | jobs | 题 | CORE Acc | +wrap Acc | ΔAcc | CORE tok | +wrap tok | Δtok | wrap 纠 CORE | wrap 损 CORE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            wrap_line("已齐合计", wrap_md),
        ]
    )
    for zh, rec in wrap_models.items():
        lines_out.append(wrap_line(zh, rec))
    lines_out.extend(
        [
            "",
            "纠少损多：wrap 比 CORE 对的题更少。token 只少一百出头，",
            "不是“再砍一截过思考”。",
            "",
            "### 2.2 按集",
            "",
            "| 切片 | jobs | 题 | CORE Acc | +wrap Acc | ΔAcc | CORE tok | +wrap tok | Δtok | wrap 纠 CORE | wrap 损 CORE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in wrap_cells:
        lines_out.append(wrap_line(f"{cell['model_zh']} {cell['dataset_zh']}", cell))
    lines_out.extend(
        [
            "",
            "只有 4B AIME25 明显赢（翻转层）。其余已齐集持平或掉。",
            "",
            "### 2.3 按抽样原因",
            "",
            "`keep_right_long` 是好锁长续写：CORE 已经对了，再禁收口词最容易验坏。",
            "",
            "| 原因 | jobs | CORE Acc | +wrap Acc | ΔAcc | CORE tok | +wrap tok | Δtok | wrap 纠 CORE | wrap 损 CORE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for reason, rec in wrap_reasons.items():
        if rec["n"] == 0:
            continue
        lines_out.append(
            f"| {reason} | {rec['n']} | {pct(rec['core_acc'])} | "
            f"{pct(rec['wrap_acc'])} | {pp(rec['d_acc_wrap_minus_core'])} | "
            f"{tok(rec['core_tok'])} | {tok(rec['wrap_tok'])} | "
            f"{dtok(rec['d_tok_wrap_minus_core'])} | "
            f"{rec['wrap_beat_core']} | {rec['wrap_lose_core']} |"
        )
    if wrap_pending:
        lines_out.extend(["", "未齐（不进表）：", ""])
        for rec in wrap_pending:
            lines_out.append(
                f"- {rec['model_zh']} {rec['dataset_zh']}: "
                f"{rec['scored']}/{rec['jobs']}"
            )

    lines_out.extend(
        [
            "",
            "## 3. 离线同口径：当初为什么短名单是这三词",
            "",
            "这是选词理由，**不是禁词因果**。口径：第一扇窗后到下一密探试答的新正文，",
            "词出现在任意一行开头。lift = P(下一试答换答\\|有该词) − P(换答\\|无该词)。",
            "覆盖 22423 条对齐轨迹，见 [`postwindow_lex.md`](postwindow_lex.md)。",
            "",
            "| 词 | 覆盖 | 换答 Lift | 好锁验坏 Lift | 错锁救回 Lift | 定位 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    roles = {
        "Wait": "主候选：高覆盖，更偏验坏",
        "Alternatively": "高覆盖；换答/救回为负，靠去一词看 token",
        "Hmm": "低频，验坏最尖",
        "But": "高频阴性对照：几乎不换答，救回为负",
    }
    for row in lifts:
        ruin = "" if row["ruin_lift"] is None else f"{row['ruin_lift'] * 100:+.1f}"
        save = "" if row["save_lift"] is None else f"{row['save_lift'] * 100:+.1f}"
        lines_out.append(
            f"| {row['word']} | {100.0 * row['prevalence']:.1f}% | "
            f"{row['change_lift'] * 100:+.1f} | {ruin} | {save} | "
            f"{roles[row['word']]} |"
        )
    for row in closers:
        post = "" if row["post_line_pct"] is None else f"{row['post_line_pct']:.1f}%"
        lines_out.append(
            f"| {row['word']} | 窗后占行 {post} | — | — | — | "
            f"未进密探候选；段首分析定为收口 |"
        )
    lines_out.extend(
        [
            "",
            "`So` / `Therefore` 当时就没有放进密探候选表，因为它们是把同一条算完，",
            "不是过思考重启口。第 2 节的 GPU 扩词已经否了再禁它们。",
            "",
            "## 4. 不能当证明的东西",
            "",
            "段首正文分类的 `plain` 是残差标签：既不是 verify / revise / newpath，",
            "就记成“同一条往下写”。Wait 和 But 在这一列上分不开；",
            "So / Therefore 的高 `plain` 正是收口，不是“没信息”。",
            "所以 **不能用重启比例或 plain 证明 CORE 优于 But/So/Therefore**。",
            "",
            "| 词 | 四模型窗后段首 plain 平均 |",
            "|---|---:|",
        ]
    )
    for row in rejected:
        plain = "" if row["plain_pct"] is None else f"{row['plain_pct']:.1f}%"
        lines_out.append(f"| {row['word']} | {plain} |")
    lines_out.extend(
        [
            "",
            "CORE 仍是整段硬禁，不是只禁段首。",
            "",
        ]
    )
    TABLE.write_text("\n".join(lines_out))
    print(f"wrote {OUT}")
    print(f"wrote {TABLE}")
    print(
        f"large-set mean dAcc={large_dacc:+.2f} dTok={large_dtok:+.0f}; "
        f"wrap n={wrap_md['n']} dAcc={wrap_md['d_acc_wrap_minus_core']:+.2f} "
        f"dTok={wrap_md['d_tok_wrap_minus_core']:+.0f} "
        f"beat={wrap_md['wrap_beat_core']} lose={wrap_md['wrap_lose_core']}"
    )


if __name__ == "__main__":
    main()
