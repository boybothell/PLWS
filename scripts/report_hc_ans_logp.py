#!/usr/bin/env python3
"""4B answer-span logp on 7B Olympiad k=4 same-answer windows."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg  # noqa: E402

ROOT = AE / "results/hc_ans_logp/olympiadbench_s42"
TABLE = AE / "tables/hc_ans_logp_oly.md"
OK = {"ok", "high_ok"}
WRONG = {"wait", "both_wrong", "high_wrong"}


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


def windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        if len(row.get("cands") or []) < 2:
            continue
        by[(int(row["question_idx"]), int(row["win_step"]))].append(row)
    out = []
    for (qid, step), seq in by.items():
        seq = sorted(seq, key=lambda x: int(x["rel_step"]))
        if [int(x["rel_step"]) for x in seq] != [0, 1, 2, 3]:
            continue
        ps = [float(x["p_self"]) for x in seq]
        picks = [int(x["pick"]) for x in seq]
        out.append(
            {
                "qid": qid,
                "cls": seq[0].get("cls"),
                "ps": ps,
                "min_p": min(ps),
                "picks": picks,
                "all_self": all(p == 0 for p in picks),
                "persist80": all(p >= 0.80 for p in ps),
            }
        )
    return out


def persist(xs: list[dict[str, Any]], tau: float) -> float:
    if not xs:
        return float("nan")
    return sum(1 for x in xs if all(p >= tau for p in x["ps"])) / len(xs)


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "—"
    return f"{n}（{100.0 * n / d:.0f}%）"


def slice_prob(name: str, xs: list[dict[str, Any]]) -> str:
    if not xs:
        return f"| {name} | 0 | — | — | — | — | — | — | — |"
    cols = [med([x["ps"][i] for x in xs]) for i in range(4)]
    return (
        f"| {name} | {len(xs)} | "
        + " | ".join(f"{c:.2f}" for c in cols)
        + f" | {med([x['min_p'] for x in xs]):.2f} | "
        f"{100.0 * persist(xs, 0.70):.0f}% | {100.0 * persist(xs, 0.80):.0f}% |"
    )


def slice_vote(name: str, xs: list[dict[str, Any]]) -> str:
    if not xs:
        return f"| {name} | 0 | — | — | — | — | — | — | — |"
    n = len(xs)
    all_self = sum(1 for x in xs if x["all_self"])
    both = sum(1 for x in xs if x["all_self"] and x["persist80"])
    same = sum(1 for x in xs if len(set(x["picks"])) == 1)
    dest = Counter(x["picks"][0] for x in xs if len(set(x["picks"])) == 1)
    dest_s = " / ".join(
        f"{'本窗' if i == 0 else f'对照{i}'} {dest[i]}" for i in sorted(dest) if dest[i]
    )
    step = []
    for i in range(4):
        rate = 100.0 * sum(1 for x in xs if x["picks"][i] == 0) / n
        step.append(f"{rate:.0f}% / {med([x['ps'][i] for x in xs]):.2f}")
    return (
        f"| {name} | {n} | {pct(all_self, n)} | {pct(both, n)} | "
        f"{100.0 * same / n:.0f}% | {dest_s or '—'} | "
        + " | ".join(step)
        + " |"
    )


def main() -> None:
    rows = load_rows()
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") in {"logp_fail", "empty_target"}]
    long = [r for r in rows if r.get("status") == "too_long"]
    wins = windows(rows)
    pos = [w for w in wins if w["cls"] in OK]
    neg = [w for w in wins if w["cls"] in WRONG]
    left_ok = [w for w in wins if w["cls"] == "ok"]
    wait = [w for w in wins if w["cls"] == "wait"]
    slices = [
        ("对窗（已对 / 高把握对）", pos),
        ("错窗（再等 / 两边错 / 高把握错）", neg),
        ("剩窗已对", left_ok),
        ("再等才对", wait),
        ("两边都错", [w for w in wins if w["cls"] == "both_wrong"]),
        ("高把握对", [w for w in wins if w["cls"] == "high_ok"]),
        ("高把握错", [w for w in wins if w["cls"] == "high_wrong"]),
    ]
    lines = [
        "# 4B 答案 token 质量：每个候选单独接在同一句后面",
        "",
        "不标 A/B/C，不让模型报数字。每个候选各自接在 `Final answer:` 后 teacher-force。",
        "分数是这段答案 token 的平均 logp，再在本步候选里做 softmax。本窗答永远是第 0 个。",
        "只有至少两个候选的齐四步窗进表。",
        f"前向 {len(rows)}，有效 {len(ok)}，抽不出质量 {len(fail)}，超长 {len(long)}。"
        f"齐四步且有对照答的窗 {len(wins)}。",
        "",
        "相对质量 = 本窗答在候选里的 softmax 质量。越大越像 4B 会写出这个答。",
        "",
        "| 切片 | 窗 | 步1 相对质量 | 步2 | 步3 | 步4 | 四步最低 | 四步≥0.70 | 四步≥0.80 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, xs in slices:
        lines.append(slice_prob(name, xs))
    lines += [
        "",
        f"四步最低相对质量 AUROC（对 vs 错） {rg.auroc([w['min_p'] for w in pos], [w['min_p'] for w in neg]):.3f}。",
        f"剩窗已对 vs 再等 {rg.auroc([w['min_p'] for w in left_ok], [w['min_p'] for w in wait]):.3f}。",
        "",
        "投票：每步看平均 logp 最大的那个候选。本窗答 = 第 0 个。",
        "",
        "| 切片 | 窗 | 四步都压本窗 | 且相对质量≥0.80 | 四步同选 | 同选去向 | 步1 压本窗 / 中位质量 | 步2 | 步3 | 步4 |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for name, xs in slices:
        lines.append(slice_vote(name, xs))
    lines.append("")
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
