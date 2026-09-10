#!/usr/bin/env python3
"""4B verbalized P(A) along 7B k=4 same-answer windows: 对窗 vs 错窗."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg  # noqa: E402

ROOT = AE / "results/hc_verbal/olympiadbench_s42"
TABLE = AE / "tables/hc_verbal_oly.md"
OK = {"ok", "high_ok"}
WRONG = {"wait", "both_wrong", "high_wrong"}
SCORE_RE = re.compile(r"(?i)\b([A-D])\s*:\s*(\d{1,3})\b")
LETTERS = "ABCD"


def med(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[len(xs) // 2]


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ROOT.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def dist_of(row: dict[str, Any], *, strict: bool) -> dict[str, float] | None:
    used = [LETTERS[i] for i in range(len(row.get("cands") or []))] + ["D"]
    raw = row.get("scores") if row.get("status") == "ok" else None
    if raw is None:
        found: dict[str, int] = {}
        for letter, value in SCORE_RE.findall(str(row.get("text") or "")):
            n = int(value)
            if 0 <= n <= 100:
                found[letter.upper()] = n
        if any(letter not in found for letter in used):
            return None
        raw = {letter: found[letter] for letter in used}
    total = sum(int(raw[letter]) for letter in used)
    if strict:
        if total not in {99, 100, 101}:
            return None
        return {letter: int(raw[letter]) / 100.0 for letter in used}
    if total <= 0:
        return None
    return {letter: int(raw[letter]) / total for letter in used}


def windows(rows: list[dict[str, Any]], *, strict: bool) -> list[dict[str, Any]]:
    by: dict[tuple[int, int], list[tuple[int, float, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") not in {"ok", "parse_fail"}:
            continue
        dist = dist_of(row, strict=strict)
        if dist is None:
            continue
        by[(int(row["question_idx"]), int(row["win_step"]))].append(
            (int(row["rel_step"]), dist["A"], str(row.get("cls") or ""))
        )
    out = []
    for (qid, step), seq in by.items():
        got = {rel: (p, cls) for rel, p, cls in seq}
        if set(got) != {0, 1, 2, 3}:
            continue
        ps = [got[i][0] for i in range(4)]
        out.append(
            {
                "qid": qid,
                "win_step": step,
                "cls": got[0][1],
                "ps": ps,
                "min_p": min(ps),
            }
        )
    return out


def persist(xs: list[dict[str, Any]], tau: float) -> float:
    if not xs:
        return float("nan")
    return sum(1 for x in xs if all(p >= tau for p in x["ps"])) / len(xs)


def slice_line(name: str, xs: list[dict[str, Any]]) -> str:
    if not xs:
        return f"| {name} | 0 | — | — | — | — | — | — | — |"
    cols = [med([x["ps"][i] for x in xs]) for i in range(4)]
    return (
        f"| {name} | {len(xs)} | "
        + " | ".join(f"{c:.2f}" for c in cols)
        + f" | {med([x['min_p'] for x in xs]):.2f} | "
        f"{100.0 * persist(xs, 0.70):.0f}% | {100.0 * persist(xs, 0.80):.0f}% |"
    )


def block(title: str, wins: list[dict[str, Any]]) -> list[str]:
    pos = [w for w in wins if w["cls"] in OK]
    neg = [w for w in wins if w["cls"] in WRONG]
    left_ok = [w for w in wins if w["cls"] == "ok"]
    wait = [w for w in wins if w["cls"] == "wait"]
    return [
        f"## {title}",
        "",
        f"齐四步的窗 {len(wins)}。",
        "",
        "| 切片 | 窗 | 步1 P(A) | 步2 | 步3 | 步4 | 四步最低 | 四步≥0.70 | 四步≥0.80 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        slice_line("对窗（已对 / 高把握对）", pos),
        slice_line("错窗（再等 / 两边错 / 高把握错）", neg),
        slice_line("剩窗已对", left_ok),
        slice_line("再等才对", wait),
        slice_line("两边都错", [w for w in wins if w["cls"] == "both_wrong"]),
        slice_line("高把握对", [w for w in wins if w["cls"] == "high_ok"]),
        slice_line("高把握错", [w for w in wins if w["cls"] == "high_wrong"]),
        "",
        f"min P(A) AUROC（对 vs 错） {rg.auroc([w['min_p'] for w in pos], [w['min_p'] for w in neg]):.3f}。",
        f"剩窗已对 vs 再等 {rg.auroc([w['min_p'] for w in left_ok], [w['min_p'] for w in wait]):.3f}。",
        "",
    ]


def main() -> None:
    rows = load_rows()
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") == "parse_fail"]
    long = [r for r in rows if r.get("status") == "too_long"]
    lines = [
        "# 4B 口头摊分：7B 奥赛四步同答对窗 / 错窗",
        "",
        "Qwen3-4B 闭思考。提示只有题目、该步 7B 思路、A/B/C 试答原文和 D. None of the above。",
        "生成 `A: int` 行，要求四数和为 100。同一扇窗四步 A/B/C 固定。",
        f"前向 {len(rows)}，严格有效 {len(ok)}，解析失败 {len(fail)}，超长 {len(long)}。",
        "",
        "假说：对窗四步都把高分扣在 A；错窗扣不住。",
        "",
    ]
    lines += block("严格：四数和为 100", windows(rows, strict=True))
    lines += block("对照：字母齐且各在 0–100，按和归一（4B 常写成独立打分）", windows(rows, strict=False))
    lines += [
        "严格协议下对窗四步≥0.80 高于错窗，但剩窗样本被解析失败削得很薄。",
        "归一对照若对/错仍分开，假说还活；高把握错也连续高锁 A，就还是承诺。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
