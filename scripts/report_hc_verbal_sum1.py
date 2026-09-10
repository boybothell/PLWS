#!/usr/bin/env python3
"""Dict-format verbalized distribution (sum to 1) on 7B k=4 windows."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg  # noqa: E402

ROOT = AE / "results/hc_verbal_sum1/olympiadbench_s42"
TABLE = AE / "tables/hc_verbal_sum1_oly.md"
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
        by[(int(row["question_idx"]), int(row["win_step"]))].append(row)
    out = []
    for (qid, step), seq in by.items():
        seq = sorted(seq, key=lambda x: int(x["rel_step"]))
        if [int(x["rel_step"]) for x in seq] != [0, 1, 2, 3]:
            continue
        ps = [float(x["p_self"]) for x in seq]
        out.append({"qid": qid, "cls": seq[0].get("cls"), "ps": ps, "min_p": min(ps)})
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


def main() -> None:
    rows = load_rows()
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") == "parse_fail"]
    long = [r for r in rows if r.get("status") == "too_long"]
    wins = windows(rows)
    pos = [w for w in wins if w["cls"] in OK]
    neg = [w for w in wins if w["cls"] in WRONG]
    left_ok = [w for w in wins if w["cls"] == "ok"]
    wait = [w for w in wins if w["cls"] == "wait"]
    lines = [
        "# 4B 口头分布（dict，和必须为 1）：7B 奥赛四步同答",
        "",
        "旧摊分约束 + 规范 dict。0.0–1.0，必须和为 1.0（±0.02），不事后归一。",
        f"前向 {len(rows)}，有效 {len(ok)}，解析/和不合法 {len(fail)}，超长 {len(long)}。"
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
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
