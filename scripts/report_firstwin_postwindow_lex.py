#!/usr/bin/env python3
"""Describe reflection words immediately after the first k=4 answer lock.

This is an offline association analysis over existing dense-probe traces.  It
does not estimate the causal effect of suppressing a word.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plws.paths import PLWSPaths
from plws.window import Observation, first_same_answer_window

K = 4
MSS = 10
MAIN_DATASETS = {
    "math-500",
    "olympiadbench",
    "gpqa-diamond",
    "aime24",
    "aime25",
}
DATASET_NAMES = {
    "math-500": "MATH",
    "olympiadbench": "Olympiad",
    "gpqa-diamond": "GPQA",
    "aime24": "AIME24",
    "aime25": "AIME25",
}
MODEL_NAMES = {
    "r1_7b": "7B",
    "nemotron_8b": "8B",
    "r1_14b": "14B",
    "r1_32b": "32B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_30b_a3b": "Qwen3-30B",
}

# Match semantic groups, not tokenizer spellings.  ``line_start`` below adds
# the start-of-line anchor; boundaries here avoid matching e.g. "somewhat".
CANDIDATES: dict[str, str] = {
    "Wait": r"wait\b",
    "Alternatively": r"alternatively\b",
    "Hmm": r"hm+\b|huh\b",
    "However": r"however\b",
    "But": r"but\b",
    "Actually": r"actually\b",
    "Let me": r"let\s+me\b",
    "Maybe": r"maybe\b",
    "Perhaps": r"perhaps\b",
    "Hold on": r"hold\s+on\b",
    "double-check": r"double[- ]check\b",
    "another way": r"another\s+(?:way|approach|method)\b",
    "等一下": r"等一下",
    "换一种": r"换一种|换个(?:思路|方法)",
    "不过/但是": r"不过|但是|然而",
    "实际上": r"实际上",
    "让我": r"让我",
}
EXPOSURE_SETS: dict[str, tuple[str, ...]] = {
    "CORE（三词任一）": ("Wait", "Alternatively", "Hmm"),
    "CORE−Wait": ("Alternatively", "Hmm"),
    "CORE−Alternatively": ("Wait", "Hmm"),
    "CORE−Hmm": ("Wait", "Alternatively"),
    "SAFE新增词任一": (
        "However",
        "Maybe",
        "Perhaps",
        "Hold on",
        "double-check",
        "another way",
        "换一种",
    ),
    "Let me": ("Let me",),
    "But": ("But",),
}
WORD_CONCLUSIONS = {
    "Wait": "主候选；高覆盖，同时增加验坏和少量救回，净效应必须受控生成确认。",
    "Alternatively": "高覆盖但不促进即时换答；保留去一词组，检验其 token 贡献。",
    "Hmm": "低频但高验坏、低救回；当前最像精准压制候选。",
    "However": "极低频；表面 Lift 大但样本不足，不单独进入主消融。",
    "But": "高频但接近零换答 Lift；作为频率匹配阴性对照。",
    "Actually": "极低频且估计不稳定，不据此选词。",
    "Let me": "促进换答但几乎不提高真实救回；作为挑战 CORE 的新增候选。",
    "Maybe": "极低频且估计不稳定，只随 SAFE 合并复核。",
    "Perhaps": "极低频且估计不稳定，只随 SAFE 合并复核。",
    "Hold on": "仅 3 例，无法判断，只随 SAFE 合并复核。",
    "double-check": "当前口径零命中，无离线证据支持单独压制。",
    "another way": "总体较少且不促进即时换答；错锁条件样本不足，只随 SAFE 复核。",
    "等一下": "当前英文轨迹零命中；保留为 Wait 的中文兼容变体。",
    "换一种": "当前英文轨迹零命中；只作为跨语言兼容变体。",
    "不过/但是": "当前英文轨迹零命中，不进入本轮英文模型消融。",
    "实际上": "当前英文轨迹零命中，不进入本轮英文模型消融。",
    "让我": "当前英文轨迹零命中，不进入本轮英文模型消融。",
}
LEADING_MARKUP = r"[ \t]*(?:[-*#>]+[ \t]*)?"
LINE_START_PATTERNS = {
    name: re.compile(rf"(?:^|\n){LEADING_MARKUP}(?:{pattern})", re.I)
    for name, pattern in CANDIDATES.items()
}
SPAN_START_PATTERNS = {
    name: re.compile(rf"^{LEADING_MARKUP}(?:{pattern})", re.I)
    for name, pattern in CANDIDATES.items()
}

REPORT = ROOT / "results" / "reports" / "firstwin_postwindow_lex.json"
TABLE = ROOT / "tables" / "firstwin_wait" / "postwindow_lex.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


_WS = re.compile(r"\s+")
_LATEX = re.compile(r"\\(?:left|right|cdot|times|dfrac|tfrac|frac|mathrm|text)")


def norm(value: Any) -> str:
    text = _WS.sub("", str(value or "").strip().lower())
    text = text.replace("dfrac", "frac").replace("tfrac", "frac")
    return _LATEX.sub("", text).replace("\\", "")


def same(left: Any, right: Any) -> bool:
    a, b = norm(left), norm(right)
    return bool(a) and a == b


def usable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = []
    for row in sorted(rows, key=lambda item: int(item["stopped_len"])):
        if not str(row.get("final_answer") or ""):
            continue
        if not math.isfinite(finite(row.get("confidence"))):
            continue
        usable.append(row)
    return usable


def span_of(old: str, new: str) -> str | None:
    """Return newly generated text, tolerating a shared-prefix truncation."""

    if not old or not new:
        return None
    if new.startswith(old):
        return new[len(old) :]
    head = old[:80]
    if not head or not new.startswith(head):
        return None
    common = 0
    for left, right in zip(old, new):
        if left != right:
            break
        common += 1
    return new[common:]


def discover_cells() -> dict[tuple[str, str, int], Path]:
    dense_root = ROOT / "results" / "upstream" / "dense_trials"
    cells: dict[tuple[str, str, int], tuple[int, Path]] = {}
    for model_dir in sorted(dense_root.glob("dense_G_*")):
        model = model_dir.name.removeprefix("dense_G_")
        for path in sorted(model_dir.glob("*/**/dense_puma/trial_answers.json")):
            relative = path.relative_to(model_dir).parts
            dataset = relative[0]
            if dataset not in MAIN_DATASETS:
                continue
            seeded = len(relative) == 4 and relative[1].startswith("seed_")
            seed = int(relative[1].removeprefix("seed_")) if seeded else 42
            priority = int(seeded)
            key = (model, dataset, seed)
            if key not in cells or priority > cells[key][0]:
                cells[key] = (priority, path)
    return {key: value[1] for key, value in cells.items()}


def answer_is_correct(answer: Any, info: dict[str, Any] | None) -> bool | None:
    if not info:
        return None
    ground_truth = info.get("ground_truth")
    if ground_truth not in (None, "") and same(answer, ground_truth):
        return True
    original = info.get("original_answer")
    if original not in (None, "") and same(answer, original):
        return bool(info.get("original_correct"))
    return False


def analyze_cell(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    trial_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    official_path = paths.puma_statistics_path(model, dataset, seed)
    official = {}
    if official_path.is_file():
        official = {
            int(row["question_idx"]): row for row in load_json(official_path)
        }

    by_question: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in load_json(trial_path):
        by_question[int(row["question_idx"])].append(row)

    records: list[dict[str, Any]] = []
    counts = {
        "questions": len(by_question),
        "windows": 0,
        "has_next": 0,
        "aligned": 0,
        "known_correctness": 0,
    }
    for question_idx, trial_rows in sorted(by_question.items()):
        rows = usable_rows(trial_rows)
        observations = [
            Observation(
                step=int(row["stopped_len"]),
                answer=row.get("final_answer"),
                confidence=finite(row.get("confidence")),
            )
            for row in rows
        ]
        window = first_same_answer_window(
            observations, k=K, mss=MSS, answers_equal=same
        )
        if window is None:
            continue
        counts["windows"] += 1
        end = next(
            index
            for index, row in enumerate(rows)
            if int(row["stopped_len"]) == window.end_step
        )
        if end + 1 >= len(rows):
            continue
        counts["has_next"] += 1
        old_prefix = str(rows[end].get("reasoning_prefix") or "")
        next_row = rows[end + 1]
        new_prefix = str(next_row.get("reasoning_prefix") or "")
        span = span_of(old_prefix, new_prefix)
        if span is None:
            continue
        counts["aligned"] += 1
        window_correct = answer_is_correct(
            window.answer, official.get(question_idx)
        )
        next_answer_correct = answer_is_correct(
            next_row.get("final_answer"), official.get(question_idx)
        )
        counts["known_correctness"] += int(window_correct is not None)
        records.append(
            {
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "question_idx": question_idx,
                "window_step": window.end_step,
                "window_correct": window_correct,
                "next_answer_correct": next_answer_correct,
                "next_answer_changed": not same(
                    next_row.get("final_answer"), window.answer
                ),
                "span_chars": len(span),
                "line_start": {
                    name: bool(pattern.search(span))
                    for name, pattern in LINE_START_PATTERNS.items()
                },
                "span_start": {
                    name: bool(pattern.search(span.lstrip(" \n\t")))
                    for name, pattern in SPAN_START_PATTERNS.items()
                },
            }
        )
    return records, {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "trial_path": str(trial_path.relative_to(ROOT)),
        "official_available": official_path.is_file(),
        **counts,
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    exposure: str = "line_start",
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    output = []
    for word in names or list(CANDIDATES):
        members = EXPOSURE_SETS.get(word, (word,))
        present = [
            row for row in rows if any(row[exposure][member] for member in members)
        ]
        absent = [
            row
            for row in rows
            if not any(row[exposure][member] for member in members)
        ]
        changed_present = sum(row["next_answer_changed"] for row in present)
        changed_absent = sum(row["next_answer_changed"] for row in absent)
        rate_present = changed_present / len(present) if present else None
        rate_absent = changed_absent / len(absent) if absent else None
        lift = (
            rate_present - rate_absent
            if rate_present is not None and rate_absent is not None
            else None
        )
        output.append(
            {
                "word": word,
                "n": len(rows),
                "present": len(present),
                "prevalence": len(present) / len(rows) if rows else None,
                "changed_present": changed_present,
                "change_rate_present": rate_present,
                "changed_absent": changed_absent,
                "change_rate_absent": rate_absent,
                "lift": lift,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -(row["present"]),
            -(row["lift"] if row["lift"] is not None else -999),
            row["word"],
        ),
    )


def core_combinations(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        present = [
            word
            for word in ("Wait", "Alternatively", "Hmm")
            if row["line_start"][word]
        ]
        counts[" + ".join(present) if present else "三词均无"] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def conditional_outcome_summary(
    rows: list[dict[str, Any]],
    *,
    window_correct: bool,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Summarize immediate change and actual corruption/rescue by exposure."""

    eligible = [
        row
        for row in rows
        if row["window_correct"] is window_correct
        and row["next_answer_correct"] is not None
    ]
    outcome_key = "corrupt" if window_correct else "rescue"

    def outcome(row: dict[str, Any]) -> bool:
        return (
            row["next_answer_correct"] is False
            if window_correct
            else row["next_answer_correct"] is True
        )

    base_change = (
        sum(row["next_answer_changed"] for row in eligible) / len(eligible)
        if eligible
        else None
    )
    base_outcome = (
        sum(outcome(row) for row in eligible) / len(eligible)
        if eligible
        else None
    )
    output = []
    for word in names or list(CANDIDATES):
        members = EXPOSURE_SETS.get(word, (word,))
        present = [
            row
            for row in eligible
            if any(row["line_start"][member] for member in members)
        ]
        absent = [
            row
            for row in eligible
            if not any(row["line_start"][member] for member in members)
        ]
        change_present = (
            sum(row["next_answer_changed"] for row in present) / len(present)
            if present
            else None
        )
        outcome_present = (
            sum(outcome(row) for row in present) / len(present)
            if present
            else None
        )
        outcome_absent = (
            sum(outcome(row) for row in absent) / len(absent)
            if absent
            else None
        )
        output.append(
            {
                "word": word,
                "n": len(eligible),
                "present": len(present),
                "base_change_rate": base_change,
                "change_rate_present": change_present,
                "change_lift_vs_all": (
                    change_present - base_change
                    if change_present is not None and base_change is not None
                    else None
                ),
                f"base_{outcome_key}_rate": base_outcome,
                f"{outcome_key}_rate_present": outcome_present,
                f"{outcome_key}_lift_vs_all": (
                    outcome_present - base_outcome
                    if outcome_present is not None and base_outcome is not None
                    else None
                ),
                f"{outcome_key}_rate_absent": outcome_absent,
                f"{outcome_key}_contrast_vs_absent": (
                    outcome_present - outcome_absent
                    if outcome_present is not None and outcome_absent is not None
                    else None
                ),
            }
        )
    return sorted(output, key=lambda row: (-row["present"], row["word"]))


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def fmt_pp(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:+.1f}"


def summary_table(summary: list[dict[str, Any]], min_present: int = 0) -> list[str]:
    lines = [
        "| 词组 | 段首题数 | 占比 | 之后下一试答换答 | 不含该词 | lift (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["present"] < min_present:
            continue
        lines.append(
            f"| {row['word']} | {row['present']} | {fmt_pct(row['prevalence'])} "
            f"| {fmt_pct(row['change_rate_present'])} "
            f"| {fmt_pct(row['change_rate_absent'])} | {fmt_pp(row['lift'])} |"
        )
    return lines


def conditional_table(
    summary: list[dict[str, Any]],
    *,
    outcome: str,
    min_present: int = 5,
) -> list[str]:
    zh = "验坏" if outcome == "corrupt" else "救回"
    lines = [
        f"| 词组 | 段首题数 | 换答率 | 换答 Lift/总体 (pp) | {zh}率 "
        f"| 条件总体{zh}率 | {zh} Lift/总体 (pp) | {zh} Δ/无该词 (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["present"] < min_present:
            continue
        lines.append(
            f"| {row['word']} | {row['present']} "
            f"| {fmt_pct(row['change_rate_present'])} "
            f"| {fmt_pp(row['change_lift_vs_all'])} "
            f"| {fmt_pct(row[f'{outcome}_rate_present'])} "
            f"| {fmt_pct(row[f'base_{outcome}_rate'])} "
            f"| {fmt_pp(row[f'{outcome}_lift_vs_all'])} "
            f"| {fmt_pp(row[f'{outcome}_contrast_vs_absent'])} |"
        )
    return lines


def all_word_conclusion_table(
    base_summary: list[dict[str, Any]],
    corrupt_summary: list[dict[str, Any]],
    rescue_summary: list[dict[str, Any]],
) -> list[str]:
    base = {row["word"]: row for row in base_summary}
    corrupt = {row["word"]: row for row in corrupt_summary}
    rescue = {row["word"]: row for row in rescue_summary}
    lines = [
        "| 词组 | 段首 n / 覆盖 | 换答 Lift | 验坏 Lift | 救回 Lift | 当前结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for word in CANDIDATES:
        overall = base[word]
        bad = corrupt[word]
        good = rescue[word]
        conditional_reliable = min(bad["present"], good["present"]) >= 100
        bad_lift = (
            fmt_pp(bad["corrupt_lift_vs_all"])
            if conditional_reliable
            else "样本不足"
        )
        good_lift = (
            fmt_pp(good["rescue_lift_vs_all"])
            if conditional_reliable
            else "样本不足"
        )
        lines.append(
            f"| {word} | {overall['present']} / {fmt_pct(overall['prevalence'])} "
            f"| {fmt_pp(overall['lift'])} | {bad_lift} | {good_lift} "
            f"| {WORD_CONCLUSIONS[word]} |"
        )
    return lines


def grouped_summaries(
    rows: list[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    values = sorted({str(row[key]) for row in rows})
    return {
        value: summarize([row for row in rows if str(row[key]) == value])
        for value in values
    }


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    all_rows: list[dict[str, Any]] = []
    cells = []
    for (model, dataset, seed), trial_path in sorted(discover_cells().items()):
        try:
            rows, cell = analyze_cell(
                paths, model, dataset, seed, trial_path
            )
        except (json.JSONDecodeError, OSError, ValueError) as error:
            print(
                f"skip {model} {dataset} s{seed}: {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        all_rows.extend(rows)
        cells.append(cell)
        print(
            f"{model} {dataset} s{seed}: q={cell['questions']} "
            f"win={cell['windows']} next={cell['has_next']} "
            f"align={cell['aligned']}",
            flush=True,
        )

    correct = [row for row in all_rows if row["window_correct"] is True]
    wrong = [row for row in all_rows if row["window_correct"] is False]
    corrupt_summary = conditional_outcome_summary(
        all_rows, window_correct=True
    )
    rescue_summary = conditional_outcome_summary(
        all_rows, window_correct=False
    )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "scripts/report_firstwin_postwindow_lex.py",
        "definition": {
            "window": "first contiguous k=4 same-answer window ending at step >=10",
            "text": "new reasoning text between window-end probe and next usable probe",
            "exposure": "candidate appears at the start of any line in that text",
            "outcome": "next usable probe answer differs from window answer",
            "lift": "P(change | word) - P(change | no word); descriptive, not causal",
        },
        "candidates": list(CANDIDATES),
        "exposure_sets": EXPOSURE_SETS,
        "cells": cells,
        "n_aligned": len(all_rows),
        "summary": summarize(all_rows),
        "summary_sets": summarize(all_rows, names=list(EXPOSURE_SETS)),
        "summary_span_start": summarize(all_rows, exposure="span_start"),
        "core_combinations": {
            "all": core_combinations(all_rows),
            "correct": core_combinations(correct),
            "wrong": core_combinations(wrong),
        },
        "conditional_outcomes": {
            "correct_window_corruption": corrupt_summary,
            "wrong_window_rescue": rescue_summary,
        },
        "by_window_correctness": {
            "correct": summarize(correct),
            "wrong": summarize(wrong),
            "correct_sets": summarize(correct, names=list(EXPOSURE_SETS)),
            "wrong_sets": summarize(wrong, names=list(EXPOSURE_SETS)),
        },
        "by_model": grouped_summaries(all_rows, "model"),
        "by_dataset": grouped_summaries(all_rows, "dataset"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    n_questions = sum(cell["questions"] for cell in cells)
    n_windows = sum(cell["windows"] for cell in cells)
    n_next = sum(cell["has_next"] for cell in cells)
    lines = [
        "# 第一扇同答窗后的反思词",
        "",
        "零 GPU 描述性分析。第一扇连续 4 步同答窗（末步 ≥10）之后，取到下一次可用密探试答前新写的正文。",
        "“段首”指这段正文任意一行开头；“换答”指紧邻下一次密探试答与窗内答案不同。",
        "lift = P(下一试答换答 | 该词段首出现) − P(换答 | 该词未段首出现)。这是相关性，不是禁词的因果收益。",
        "",
        "原始 Full-CoT 全文/窗后段首密度、以及各模型当前最值得看的词，见",
        "[`fullcot_line_start.md`](fullcot_line_start.md)。CORE 仍是整段硬禁，不是只禁段首。",
        "",
        f"覆盖 {len(cells)} 个现有 cell、{n_questions} 条密探题；第一扇窗 {n_windows}，"
        f"有下一探针 {n_next}，前缀对齐 {len(all_rows)}（{fmt_pct(len(all_rows) / n_next if n_next else None)}）。",
        "",
        "## 组合覆盖",
        "",
        *summary_table(report["summary_sets"]),
        "",
        "三词段首共现（同一密探间隔可以出现多个段首）："
        + "；".join(
            f"{name} {count}"
            for name, count in report["core_combinations"]["all"].items()
        )
        + "。",
        "",
        "## 全部可用轨迹",
        "",
        *summary_table(report["summary"]),
        "",
        "## 窗时已对",
        "",
        f"n={len(correct)}。",
        "",
        *summary_table(report["by_window_correctness"]["correct"], min_present=5),
        "",
        "## 窗时已错",
        "",
        f"n={len(wrong)}。",
        "",
        *summary_table(report["by_window_correctness"]["wrong"], min_present=5),
        "",
        "## 好锁验坏 vs 错锁救回",
        "",
        "这里不再把“换了答案”直接当成验坏或救回：验坏要求下一试答实际错误，"
        "救回要求下一试答实际正确。`Lift/总体` 使用本节条件总体作为基线，"
        "`Δ/无该词` 则直接比较有词与无词样本。",
        "",
        "### 窗时已对：好锁验坏",
        "",
        *conditional_table(corrupt_summary, outcome="corrupt"),
        "",
        "### 窗时已错：错锁救回",
        "",
        *conditional_table(rescue_summary, outcome="rescue"),
        "",
        "## 按模型",
        "",
    ]
    for model, summary in report["by_model"].items():
        model_rows = [row for row in all_rows if row["model"] == model]
        lines += [
            f"### {MODEL_NAMES.get(model, model)}",
            "",
            f"n={len(model_rows)}；仅列段首出现至少 10 题的词。",
            "",
            *summary_table(summary, min_present=10),
            "",
        ]
    lines += [
        "## 按数据集",
        "",
    ]
    for dataset, summary in report["by_dataset"].items():
        dataset_rows = [row for row in all_rows if row["dataset"] == dataset]
        lines += [
            f"### {DATASET_NAMES.get(dataset, dataset)}",
            "",
            f"n={len(dataset_rows)}；仅列段首出现至少 10 题的词。",
            "",
            *summary_table(summary, min_present=10),
            "",
        ]
    corrupt_by = {row["word"]: row for row in corrupt_summary}
    rescue_by = {row["word"]: row for row in rescue_summary}
    lines += [
        "## 候选词逐项结论",
        "",
        "换答 Lift 为有词与无词之差；验坏/救回 Lift 为相对各自条件总体之差。"
        "验坏与救回两栏要求好锁、错锁中各至少 100 个命中，否则标为样本不足。",
        "",
        *all_word_conclusion_table(
            report["summary"], corrupt_summary, rescue_summary
        ),
        "",
        "## 条件结果读法",
        "",
        "- `Wait`：好锁验坏 Lift/总体 "
        f"{fmt_pp(corrupt_by['Wait']['corrupt_lift_vs_all'])}pp，"
        "错锁救回 Lift/总体 "
        f"{fmt_pp(rescue_by['Wait']['rescue_lift_vs_all'])}pp；"
        "同时放大两类变化，但更偏向验坏，不是纯负面词。",
        "- `Hmm`：好锁验坏 Lift/总体 "
        f"{fmt_pp(corrupt_by['Hmm']['corrupt_lift_vs_all'])}pp，"
        "错锁救回 Lift/总体 "
        f"{fmt_pp(rescue_by['Hmm']['rescue_lift_vs_all'])}pp；"
        "当前描述上最像“高验坏、低救回”的压制候选，但样本较少。",
        "- `Alternatively`：好锁验坏 Lift/总体 "
        f"{fmt_pp(corrupt_by['Alternatively']['corrupt_lift_vs_all'])}pp，"
        "错锁救回 Lift/总体 "
        f"{fmt_pp(rescue_by['Alternatively']['rescue_lift_vs_all'])}pp；"
        "高覆盖但不像即时有效自纠错词，是否省 token 仍需生成消融。",
        "- `Let me`：错锁后的换答 Lift/总体 "
        f"{fmt_pp(rescue_by['Let me']['change_lift_vs_all'])}pp，"
        "但真实救回 Lift/总体仅 "
        f"{fmt_pp(rescue_by['Let me']['rescue_lift_vs_all'])}pp；"
        "当前数据不支持“说 Let me 就真的自我纠错”，多数换答没有立刻变对。",
        "",
        "对挑选压词而言，理想候选是“好锁验坏 Lift 高、错锁真实救回 Lift 低”；"
        "这与评价词本身是否有益时的方向相反。两种 Lift 都是观察相关，不能替代受控禁词实验。",
        "",
    ]
    lines += [
        "## 解释限制",
        "",
        "- 一段可能同时命中多个词，单词 lift 尚未控制共现、模型和数据集构成。",
        "- 密探间隔内的段首词先于下一次试答，但不等于禁掉该词就会避免换答。",
        "- Qwen3 密探覆盖仍在补；不同模型的 n 不齐，跨模型总表不能直接当冻结词表依据。",
        f"- 完整分层数字见 `{REPORT.relative_to(ROOT)}`。",
        "",
    ]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines))
    print(f"wrote {REPORT}", flush=True)
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
