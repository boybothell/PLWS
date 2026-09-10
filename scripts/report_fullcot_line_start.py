#!/usr/bin/env python3
"""Full-CoT line-start stats, including the span after the first same-answer window.

This is an offline description of original Full-CoT reasoning.  It does not
change CORE, which remains a full-thought hard ban of Wait / Alternatively / Hmm.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plws.artifacts import load_jsonl
from plws.inputs import same_answer
from plws.matrix import FIRSTWIN
from plws.paths import PLWSPaths

REPORT = ROOT / "results" / "reports" / "fullcot_line_start.json"
TABLE = ROOT / "tables" / "firstwin_wait" / "fullcot_line_start.md"

MODELS = [
    ("r1_7b", "7B"),
    ("nemotron_8b", "Nemotron"),
    ("r1_14b", "14B"),
    ("qwen3_4b", "4B"),
]
DATASETS = (
    "math-500",
    "olympiadbench",
    "gpqa-diamond",
    "aime24",
    "aime25",
)
SEEDS = (42, 0, 1, 123)
MARKERS = [
    ("Wait", r"wait\b"),
    ("Alternatively", r"alternatively\b"),
    ("Hmm", r"hmm+\b|hm\b"),
    ("Let me", r"let me\b"),
    ("But", r"but\b"),
    ("But let", r"but let\b"),
    ("But maybe", r"but maybe\b"),
    ("However", r"however\b"),
    ("Maybe", r"maybe\b"),
    ("So", r"so\b"),
    ("Therefore", r"therefore\b"),
    ("Let me check", r"let me (?:check|verify|confirm|double-check)"),
    ("Let me think", r"let me think"),
]
WORTHWHILE = {
    "7B": ("Wait", "But", "Alternatively", "Hmm"),
    "Nemotron": ("Alternatively", "Wait", "But"),
    "14B": ("Wait", "But", "Alternatively", "Hmm"),
    "4B": ("But", "Alternatively", "Let me", "Wait"),
}
WRAP_WORDS = ("So", "Therefore")
LEADING = r"(?m)^(?:[#>*\-\s]*)(?:\*\*)?"
COMPILED = [
    (name, re.compile(LEADING + f"({pattern})", re.I)) for name, pattern in MARKERS
]
VERIFY = re.compile(
    r"\b(check|verify|confirm|double[- ]check|make sure|recalculat|recompute|"
    r"plug (?:it |this )?back|sanity|does that (?:make sense|work)|"
    r"is this (?:right|correct)|yes,? that(?:'s| is) (?:correct|right)|"
    r"this (?:confirms|matches|checks out)|"
    r"let me (?:check|verify|confirm|double-check|recheck|make sure)|"
    r"same as (?:original|before|the original))\b",
    re.I,
)
NEWPATH = re.compile(
    r"\b(another (?:way|approach|method|idea)|different (?:way|approach|method)|"
    r"instead|try (?:another|a different)|from scratch|start over|new approach)\b",
    re.I,
)
REVISE = re.compile(
    r"\b(wrong|mistake|incorrect|doesn't work|does not work|can't be|cannot be|"
    r"I (?:erred|messed|was wrong)|that's not right|not correct|contradict)\b",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pct(part: int | float, whole: int | float) -> float | None:
    if not whole:
        return None
    return 100.0 * part / whole


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}%"


def reasoning_path(paths: PLWSPaths, model: str, dataset: str, seed: int) -> Path | None:
    folder = paths.puma_output_dir(model, dataset, seed)
    if model == "r1_7b" and dataset == "math-500" and seed == 42:
        official = (
            paths.results / "baselines" / "official" / "math500_official" / "puma_ds7b"
        )
        if (official / "answers.json").is_file():
            folder = official
    candidate = folder / "answers.json"
    return candidate if candidate.is_file() else None


def texts_from_answers(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else data.get("rows", [])
    out: list[tuple[str, str]] = []
    for row in rows:
        text = row.get("reasoning") or ""
        if not text and row.get("generated_text"):
            generated = row["generated_text"]
            if "<think>" in generated:
                text = generated.split("<think>", 1)[1].split("</think>", 1)[0]
            else:
                text = generated.split("</think>", 1)[0]
        if text.strip():
            out.append(((row.get("question") or "").strip(), text))
    return out


def split_after_thought(thought: str, reasoning: str) -> str | None:
    if not thought or not reasoning:
        return None
    if reasoning.startswith(thought):
        return reasoning[len(thought) :]
    shared = 0
    for left, right in zip(thought, reasoning):
        if left != right:
            break
        shared += 1
    if shared < 80:
        return None
    return reasoning[shared:]


def nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def count_markers(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in COMPILED}


def paragraph_after(text: str, match: re.Match[str]) -> str:
    end = text.find("\n\n", match.end())
    if end < 0:
        end = min(len(text), match.end() + 350)
    else:
        end = min(end, match.end() + 400)
    return text[match.start() : end]


def primary_label(text: str) -> str:
    if REVISE.search(text):
        return "revise"
    if NEWPATH.search(text):
        return "newpath"
    if VERIFY.search(text):
        return "verify"
    return "plain"


def lock_bucket(job: dict[str, Any]) -> str:
    lock_ok = bool(job.get("left_ok"))
    final_ok = bool(job.get("orig_ok"))
    same = same_answer(job.get("old_answer"), job.get("original"))
    if lock_ok and same:
        return "ok_eq"
    if lock_ok and not same:
        return "ok_ruin"
    if (not lock_ok) and final_ok:
        return "bad_save"
    if (not lock_ok) and same:
        return "bad_eq"
    return "bad_other"


def empty_model() -> dict[str, Any]:
    return {
        "full_questions": 0,
        "full_lines": 0,
        "full_qhit": Counter(),
        "full_lhit": Counter(),
        "window_jobs": 0,
        "window_aligned": 0,
        "post_questions": 0,
        "post_lines": 0,
        "pre_lines": 0,
        "post_qhit": Counter(),
        "post_lhit": Counter(),
        "pre_lhit": Counter(),
        "buckets": Counter(),
        "word_q": {name: Counter() for name, _ in MARKERS},
        "word_lab": {name: Counter() for name, _ in MARKERS},
        "word_lab_ok_eq": {name: Counter() for name, _ in MARKERS},
    }


def analyze() -> dict[str, Any]:
    paths = PLWSPaths.discover(__file__)
    stats = {zh: empty_model() for _, zh in MODELS}
    missing_cells: list[str] = []

    for model, zh in MODELS:
        model_stats = stats[zh]
        for dataset in DATASETS:
            for seed in SEEDS:
                answers = reasoning_path(paths, model, dataset, seed)
                if answers is None:
                    missing_cells.append(f"{model}:{dataset}:{seed}")
                    continue
                pairs = texts_from_answers(answers)
                qmap = {question: text for question, text in pairs}
                for _, text in pairs:
                    model_stats["full_questions"] += 1
                    lines = nonempty_lines(text)
                    model_stats["full_lines"] += len(lines)
                    counts = count_markers(text)
                    for name, hits in counts.items():
                        if hits:
                            model_stats["full_qhit"][name] += 1
                            model_stats["full_lhit"][name] += hits

                jobs_path = paths.jobs_path(
                    model, dataset, seed, FIRSTWIN, k=4, lexicon="core"
                )
                if not jobs_path.is_file():
                    continue
                for job in load_jsonl(jobs_path):
                    model_stats["window_jobs"] += 1
                    text = qmap.get((job.get("question") or "").strip())
                    if text is None:
                        continue
                    post = split_after_thought(job.get("thought") or "", text)
                    if post is None:
                        continue
                    model_stats["window_aligned"] += 1
                    if not post.strip():
                        continue
                    model_stats["post_questions"] += 1
                    thought = job.get("thought") or ""
                    model_stats["pre_lines"] += len(nonempty_lines(thought))
                    model_stats["post_lines"] += len(nonempty_lines(post))
                    bucket = lock_bucket(job)
                    model_stats["buckets"][bucket] += 1
                    pre_counts = count_markers(thought)
                    post_counts = count_markers(post)
                    for name, hits in pre_counts.items():
                        model_stats["pre_lhit"][name] += hits
                    for name, hits in post_counts.items():
                        if hits:
                            model_stats["post_qhit"][name] += 1
                            model_stats["post_lhit"][name] += hits
                            model_stats["word_q"][name][bucket] += 1
                            matches = list(
                                next(p for n, p in COMPILED if n == name).finditer(post)
                            )
                            for match in matches[:3]:
                                label = primary_label(paragraph_after(post, match))
                                model_stats["word_lab"][name][label] += 1
                                if bucket == "ok_eq":
                                    model_stats["word_lab_ok_eq"][name][label] += 1

    models_out: dict[str, Any] = {}
    for _, zh in MODELS:
        raw = stats[zh]
        n_full = raw["full_questions"]
        n_lines = raw["full_lines"]
        n_post = raw["post_questions"]
        n_post_lines = raw["post_lines"]
        n_pre_lines = raw["pre_lines"]
        words = {}
        for name, _ in MARKERS:
            post_q = raw["post_qhit"][name]
            lab = raw["word_lab"][name]
            lab_n = sum(lab.values())
            ok_lab = raw["word_lab_ok_eq"][name]
            ok_n = sum(ok_lab.values())
            q_buckets = raw["word_q"][name]
            q_n = sum(q_buckets.values())
            words[name] = {
                "full_question_pct": pct(raw["full_qhit"][name], n_full),
                "full_line_pct": pct(raw["full_lhit"][name], n_lines),
                "post_question_pct": pct(raw["post_qhit"][name], n_post),
                "post_line_pct": pct(raw["post_lhit"][name], n_post_lines),
                "pre_line_pct": pct(raw["pre_lhit"][name], n_pre_lines),
                "post_questions": post_q,
                "lock_eq_final_pct": pct(
                    q_buckets["ok_eq"] + q_buckets["bad_eq"], q_n
                ),
                "ok_eq_pct": pct(q_buckets["ok_eq"], q_n),
                "bad_save_pct": pct(q_buckets["bad_save"], q_n),
                "ok_ruin_pct": pct(q_buckets["ok_ruin"], q_n),
                "labels": {
                    key: pct(lab[key], lab_n)
                    for key in ("verify", "revise", "newpath", "plain")
                },
                "ok_eq_labels": {
                    key: pct(ok_lab[key], ok_n)
                    for key in ("verify", "revise", "newpath", "plain")
                },
            }
        buckets = raw["buckets"]
        models_out[zh] = {
            "full_questions": n_full,
            "full_lines": n_lines,
            "window_jobs": raw["window_jobs"],
            "window_aligned": raw["window_aligned"],
            "post_questions": n_post,
            "post_lines": n_post_lines,
            "pre_lines": n_pre_lines,
            "buckets": {
                "ok_eq_pct": pct(buckets["ok_eq"], n_post),
                "ok_ruin_pct": pct(buckets["ok_ruin"], n_post),
                "bad_save_pct": pct(buckets["bad_save"], n_post),
                "bad_eq_pct": pct(buckets["bad_eq"], n_post),
                "bad_other_pct": pct(buckets["bad_other"], n_post),
                "lock_eq_final_pct": pct(
                    buckets["ok_eq"] + buckets["bad_eq"], n_post
                ),
                "lock_ok_pct": pct(buckets["ok_eq"] + buckets["ok_ruin"], n_post),
                "counts": dict(buckets),
            },
            "words": words,
            "worthwhile": list(WORTHWHILE[zh]),
        }

    return {
        "generated_at": utc_now(),
        "script": "scripts/report_fullcot_line_start.py",
        "table": str(TABLE.relative_to(ROOT)),
        "scope": {
            "models": [zh for _, zh in MODELS],
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "source": "PUMA answers.json reasoning; 7B MATH seed 42 uses official puma_ds7b",
            "window": "firstwin k=4, end step >= 10, leftover = Full-CoT after job.thought",
            "line_start": "nonempty line, optional markdown/bullet prefix",
            "core_note": "CORE remains a full-thought hard ban; line-start is analysis only",
        },
        "missing_cells": missing_cells,
        "models": models_out,
    }


def why_word(model: str, word: str) -> str:
    notes = {
        ("7B", "Wait"): "主换段词；窗后比窗前更密；好锁时下一试答很少换",
        ("7B", "But"): "窗后第二密的重启口；换答 Lift 接近 0",
        ("7B", "Alternatively"): "覆盖高，但不促进换答",
        ("7B", "Hmm"): "低频；验坏 Lift 高、救回低",
        ("14B", "Wait"): "主换段词，窗后更密",
        ("14B", "But"): "窗后第二密的重启口",
        ("14B", "Alternatively"): "覆盖高，换答弱",
        ("14B", "Hmm"): "同 7B，量小、偏验坏",
        ("Nemotron", "Alternatively"): "它的主换段词；窗后还升了",
        ("Nemotron", "Wait"): "覆盖几乎每题；窗后段落里验算口气约四分之一",
        ("Nemotron", "But"): "和前两个同量级",
        ("4B", "But"): "最密；常见形状是 But let me check",
        ("4B", "Alternatively"): "几乎每题、窗后还升；很多是同一条再写一遍",
        ("4B", "Let me"): "4B 才密；check / think again 族",
        ("4B", "Wait"): "题题都有，但占行只有 7B 的约三分之一",
    }
    return notes.get((model, word), "窗后段首密，且不像 So / Therefore 那种收口")


def render(report: dict[str, Any]) -> str:
    models = report["models"]
    names = [name for name, _ in MARKERS]
    lines = [
        "# 原始 Full-CoT 段首词",
        "",
        "零 GPU 描述性分析。看的是 PUMA `answers.json` 里的原始 `reasoning`，",
        "不是 CORE 压制后的续写。CORE 仍是思考阶段整段硬禁 `Wait / Alternatively / Hmm`，",
        "不是只禁段首。本表只记录段首比例和窗后行为，不改跑法。",
        "",
        "7B MATH seed 42 用官方 `puma_ds7b`。四模型各五集四 seed。8B / 30B 未齐，不进本表。",
        "",
        "## 四个比例",
        "",
        "- **全文占行**：整道 Full-CoT 里，非空行中从这个词开头的行占多少。看密不密。",
        "- **全文题**：至少有一行从这个词开头的题占多少。看广不广。",
        "- **窗后占行**：第一次同答窗之后的续写里，这个词当段首的行占多少。看锁完之后还密不密。",
        "- **窗后题**：有第一扇窗且对得上正文的题里，窗后至少有一行从这个词开头的占多少。看锁完之后还广不广。",
        "",
        "覆盖高、占行低 = 题题都用、但不当主换段词。覆盖和占行都高 = 又广又密。",
        "",
        "## 覆盖",
        "",
        "| 模型 | 全文题 | 全文非空行 | 有窗且对上窗后正文 | 窗后非空行 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, zh in MODELS:
        row = models[zh]
        lines.append(
            f"| {zh} | {row['full_questions']} | {row['full_lines']} | "
            f"{row['post_questions']} | {row['post_lines']} |"
        )

    lines += [
        "",
        "## 全文段首",
        "",
        "| 段首词 | 7B题 | Nemo | 14B | 4B | 7B占行 | Nemo | 14B | 4B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        q = [fmt_pct(models[zh]["words"][name]["full_question_pct"]) for _, zh in MODELS]
        l = [fmt_pct(models[zh]["words"][name]["full_line_pct"], 2) for _, zh in MODELS]
        lines.append(
            f"| {name} | {q[0]} | {q[1]} | {q[2]} | {q[3]} | {l[0]} | {l[1]} | {l[2]} | {l[3]} |"
        )

    lines += [
        "",
        "## 窗后段首",
        "",
        "| 段首词 | 7B题 | Nemo | 14B | 4B | 7B占行 | Nemo | 14B | 4B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        q = [fmt_pct(models[zh]["words"][name]["post_question_pct"]) for _, zh in MODELS]
        l = [fmt_pct(models[zh]["words"][name]["post_line_pct"], 2) for _, zh in MODELS]
        lines.append(
            f"| {name} | {q[0]} | {q[1]} | {q[2]} | {q[3]} | {l[0]} | {l[1]} | {l[2]} | {l[3]} |"
        )

    lines += [
        "",
        "## 窗后整段答案有没有变",
        "",
        "有窗、对得上正文的题。好锁改口很少；约六成终答仍是窗里那个答案。",
        "",
        "| 模型 | 锁=终答 | 好锁且不变 | 好锁后改口 | 错锁救回 | 错锁坚持 | 错锁另错 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, zh in MODELS:
        b = models[zh]["buckets"]
        lines.append(
            f"| {zh} | {fmt_pct(b['lock_eq_final_pct'])} | {fmt_pct(b['ok_eq_pct'])} | "
            f"{fmt_pct(b['ok_ruin_pct'])} | {fmt_pct(b['bad_save_pct'])} | "
            f"{fmt_pct(b['bad_eq_pct'])} | {fmt_pct(b['bad_other_pct'])} |"
        )

    lines += [
        "",
        "## 窗后高频段首是不是验算空转",
        "",
        "不是整段都能叫无意义验算。`So` / `Therefore` 主要是把同一条算完；",
        "`Wait` / `Alternatively` / `But` 看起来像重开，多数仍改不了锁答，只有一小部分真纠错。",
        "标签互斥：revise > newpath > verify > plain。每题每个词最多看 3 段。",
        "",
        "| 模型 | 段首 | 好锁不变 | 错锁救回 | verify | revise | newpath | plain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, zh in MODELS:
        for name in ("Wait", "Alternatively", "But", "Let me", "So", "Therefore"):
            word = models[zh]["words"][name]
            if not word["post_questions"]:
                continue
            lab = word["labels"]
            lines.append(
                f"| {zh} | {name} | {fmt_pct(word['ok_eq_pct'])} | "
                f"{fmt_pct(word['bad_save_pct'])} | {fmt_pct(lab['verify'])} | "
                f"{fmt_pct(lab['revise'])} | {fmt_pct(lab['newpath'])} | "
                f"{fmt_pct(lab['plain'])} |"
            )

    lines += [
        "",
        "## 各模型当前最值得看的词",
        "",
        "按窗后段首密度排，且排除 `So` / `Therefore` 收口。这是分析候选，不是新词表，也不是把 CORE 改成只禁段首。",
        "",
    ]
    for _, zh in MODELS:
        lines.append(f"### {zh}")
        lines.append("")
        lines.append("| 词 | 全文占行 | 全文题 | 窗后占行 | 窗后题 | 定位 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for name in models[zh]["worthwhile"]:
            word = models[zh]["words"][name]
            lines.append(
                f"| **{name}** | {fmt_pct(word['full_line_pct'], 2)} | "
                f"{fmt_pct(word['full_question_pct'])} | {fmt_pct(word['post_line_pct'], 2)} | "
                f"{fmt_pct(word['post_question_pct'])} | {why_word(zh, name)} |"
            )
        wrap = ", ".join(
            f"`{name}` 窗后占行 {fmt_pct(models[zh]['words'][name]['post_line_pct'], 2)}"
            for name in WRAP_WORDS
        )
        lines.append("")
        lines.append(f"不列入收口：{wrap}。")
        lines.append("")

    lines += [
        "## 读法",
        "",
        "- 7B / 14B：先看 `Wait`，`But` 第二档，`Alternatively` 第三档。",
        "- Nemotron：`Alternatively` ≈ `Wait` ≈ `But`，没有单独一个断层第一。",
        "- 4B：先看 `But` 和 `Alternatively`，`Let me` 比 `Wait` 更像它的过思考口。",
        "- CORE 已经全禁 `Wait / Alternatively / Hmm`。4B 窗后最密的两个（`But`、`Let me`）不在 CORE 里。",
        "- 窗后这些重启词多数改不了锁答，但不能说基本上都是无意义验算；错锁仍有一小截靠后面写对。",
        "",
        "密探间隔里的换答 Lift 仍见 [`postwindow_lex.md`](postwindow_lex.md)。",
        "机器报告：`results/reports/fullcot_line_start.json`。",
        "",
    ]
    return "\n".join(lines)


def json_ready(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    report = analyze()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2) + "\n")
    TABLE.write_text(render(report))
    print(f"wrote {TABLE.relative_to(ROOT)}")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    for _, zh in MODELS:
        row = report["models"][zh]
        words = ", ".join(row["worthwhile"])
        print(
            f"{zh}: full={row['full_questions']} post={row['post_questions']} worthwhile={words}"
        )


if __name__ == "__main__":
    main()
