#!/usr/bin/env python3
"""4B 开思考闭集投票：连续同选且概率高，对照奥赛剩窗。"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg  # noqa: E402
import report_leftover_after as after  # noqa: E402

ROOT = AE / "results/hc_think/olympiadbench_s42"
TABLE = AE / "tables/hc_think_oly.md"
NONE = "__NONE__"


def fin(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ROOT.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    rows.sort(key=lambda x: (int(x["question_idx"]), int(x["decision_step"])))
    return rows


def locked_runs(seq: list[dict[str, Any]], k: int, tau: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(seq):
        cur = seq[i]
        pick = cur.get("pick")
        if pick == NONE or not pick:
            i += 1
            continue
        j = i
        ps = []
        while j < len(seq) and seq[j].get("pick") != NONE and rg.same(seq[j].get("pick"), pick):
            p = fin(seq[j].get("pick_p"))
            ps.append(p)
            j += 1
        if j - i >= k and all(p == p and p >= tau for p in ps[: j - i]):
            out.append(
                {
                    "start": i,
                    "end": j - 1,
                    "n": j - i,
                    "pick": pick,
                    "min_p": min(ps),
                    "step": int(seq[j - 1]["decision_step"]),
                    "trial": seq[j - 1].get("trial_answer"),
                    "same_trial": bool(rg.same(pick, seq[j - 1].get("trial_answer"))),
                }
            )
            i = j
        else:
            i += 1
    return out


def line(name: str, n_q: int, fires: list[dict[str, Any]], left: dict[int, dict[str, Any]]) -> str:
    if not fires:
        return f"| {name} | {n_q} | 0 | — | — | — |"
    hit = [f for f in fires if int(f["qid"]) in left]
    ok = sum(1 for f in hit if left[int(f["qid"])].get("left_ok"))
    wait = sum(1 for f in hit if left[int(f["qid"])].get("wait_helps"))
    return (
        f"| {name} | {n_q} | {len(fires)} | {len(hit)} | "
        f"{ok}/{len(hit) if hit else 0} | {wait}/{len(hit) if hit else 0} |"
    )


def main() -> None:
    rows = load_rows()
    left_rows = after.load_left("r1_7b", "olympiadbench", 42)
    left = {int(x["question_idx"]): x for x in left_rows}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[int(row["question_idx"])].append(row)
    n_q = len(by)
    pick_trial = sum(1 for r in rows if r.get("pick_is_trial"))
    pick_none = sum(1 for r in rows if r.get("pick_is_none"))
    closed = sum(1 for r in rows if r.get("closed"))
    lines = [
        "# 4B 开思考闭集投票（奥赛 7B 每步试答）",
        "",
        "Qwen3-4B `enable_thinking=True`。提示只有题目 + 到该步出现过的试答（1-4，含「以上都不对」）。",
        "不带 7B 思路。思考最多 256 词，再读 1/2/3/4 的归一化概率，取最高作为选择。",
        f"已打 {len(rows)} 步 / {n_q} 题。选当前试答 {pick_trial}/{len(rows)}，选都不对 {pick_none}/{len(rows)}，思考收口 {closed}/{len(rows)}。",
        "",
        "| 连续同选 | 题 | 至少一扇锁 | 锁在剩窗题 | 其中剩窗已对 | 其中再等才对 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for k, tau in ((2, 0.80), (3, 0.80), (4, 0.80), (4, 0.90), (4, 0.95)):
        fires = []
        for qid, seq in by.items():
            runs = locked_runs(seq, k, tau)
            if runs:
                first = runs[0]
                first["qid"] = qid
                fires.append(first)
        lines.append(line(f"k={k} 且 min p≥{tau:.2f}", n_q, fires, left))
    # leftover-step snapshot
    left_hit = []
    for rec in left_rows:
        qid = int(rec["question_idx"])
        seq = by.get(qid) or []
        at = next((r for r in seq if int(r["decision_step"]) == int(rec["left_step"])), None)
        if at:
            left_hit.append({**rec, "pick": at.get("pick"), "pick_p": at.get("pick_p"), "pick_is_trial": at.get("pick_is_trial")})
    if left_hit:
        ok = [x for x in left_hit if x.get("left_ok")]
        wait = [x for x in left_hit if x.get("wait_helps")]
        lines += [
            "",
            "## 剩窗当步 4B 选了谁",
            "",
            f"接到剩窗当步 {len(left_hit)}。已对 {len(ok)}，再等才对 {len(wait)}。",
            f"已对仍选当前试答 {sum(1 for x in ok if x.get('pick_is_trial'))}/{len(ok)}，"
            f"中位概率 {sorted(fin(x['pick_p']) for x in ok if fin(x['pick_p'])==fin(x['pick_p']))[len(ok)//2] if ok else float('nan'):.2f}。",
            f"再等仍选当前试答 {sum(1 for x in wait if x.get('pick_is_trial'))}/{len(wait)}。",
            "",
        ]
    lines += [
        "方法能活：连续同选高概率时，已对明显多于再等，且低 FPR 优于把握连答。",
        "两边都锁当前试答，就还是承诺，不要再改编号。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
