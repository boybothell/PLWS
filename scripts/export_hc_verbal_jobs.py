#!/usr/bin/env python3
"""Export 7B Olympiad k=4 same-answer windows for 4B verbalized distributions.

Each leftover window and the first high window (if distinct) emit 4 steps.
A/B/C are answer strings only; D is always None of the above. No CoT in the
job file — the scorer loads reasoning_prefix from the dense trials.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
import report_first_lock_room as room  # noqa: E402
import report_k4_hyps as hy  # noqa: E402
import report_k4_second_lock as sl  # noqa: E402
import report_leftover_after as after  # noqa: E402

OUT = AE / "results/hc_verbal/olympiadbench_s42/jobs.jsonl"
MODEL = "r1_7b"
DATASET = "olympiadbench"
SEED = 42


def nearest_others(rows: list[dict[str, Any]], win_ans: str, win_step: int, k: int = 2) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for row in rows:
        ans = str(row.get("final_answer") or "")
        if not ans or rg.same(ans, win_ans):
            continue
        step = int(row["stopped_len"])
        scored.append((abs(step - win_step), step, ans))
    scored.sort()
    out: list[str] = []
    for _dist, _step, ans in scored:
        if any(rg.same(ans, old) for old in out):
            continue
        out.append(ans)
        if len(out) >= k:
            break
    return out


def window_rows(rows: list[dict[str, Any]], win: dict[str, Any]) -> list[dict[str, Any]]:
    end = int(win["end"])
    return rows[end + 1 - rg.K : end + 1]


def pack_win(
    *,
    qi: int,
    question: str,
    rows: list[dict[str, Any]],
    win: dict[str, Any],
    cls: str,
    source: str,
) -> list[dict[str, Any]]:
    ans = str(win["ans"] or "")
    step = int(win["step"])
    others = nearest_others(rows, ans, step, 2)
    cands = [ans] + others
    jobs = []
    for rel, row in enumerate(window_rows(rows, win)):
        jobs.append(
            {
                "question_idx": qi,
                "win_step": step,
                "win_kind": win["kind"],
                "source": source,
                "cls": cls,
                "rel_step": rel,
                "decision_step": int(row["stopped_len"]),
                "trial_answer": ans,
                "geo_conf": row.get("confidence"),
                "question": question,
                "cands": cands,
            }
        )
    return jobs


def main() -> None:
    trials_path = room.dense_trial_path(MODEL, DATASET, SEED)
    official_path = dd.puma_stat_path(MODEL, DATASET, SEED)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(official_path)}
        if official_path.is_file()
        else {}
    )
    gp = low.gpath(MODEL, DATASET, SEED)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    left_map = {int(x["question_idx"]): x for x in after.load_left(MODEL, DATASET, SEED)}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        grouped[int(row["question_idx"])].append(row)

    jobs: list[dict[str, Any]] = []
    n_left = n_high = 0
    for qi, trials in grouped.items():
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        question = str(rows[0].get("question") or trials[0].get("question") or "")
        if not question:
            continue
        wins = sl.same_windows(rows)
        if not wins:
            continue
        info = official.get(qi) or {}
        g = gmap.get(qi) or {}
        last = max(trials, key=lambda x: int(x["stopped_len"]))
        gt = info.get("ground_truth") or g.get("ground_truth")
        original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
        orig_ok = bool(info.get("original_correct")) if "original_correct" in info else bool(
            low.credit(original, gt, original, True)
        )
        left = next((w for w in wins if hy.leftover(w)), None)
        high = next((w for w in wins if w["kind"] == "high"), None)
        rec = left_map.get(qi)
        if left is not None:
            if rec:
                if rec.get("wait_helps"):
                    cls = "wait"
                elif rec.get("left_ok"):
                    cls = "ok"
                else:
                    cls = "both_wrong"
            else:
                cls = "ok" if low.credit(left["ans"], gt, original, orig_ok) else "both_wrong"
            jobs.extend(
                pack_win(qi=qi, question=question, rows=rows, win=left, cls=cls, source="leftover")
            )
            n_left += 1
        if high is not None and (left is None or int(high["step"]) != int(left["step"])):
            high_ok = bool(low.credit(high["ans"], gt, original, orig_ok))
            jobs.extend(
                pack_win(
                    qi=qi,
                    question=question,
                    rows=rows,
                    win=high,
                    cls="high_ok" if high_ok else "high_wrong",
                    source="high",
                )
            )
            n_high += 1

    jobs.sort(key=lambda x: (x["question_idx"], x["win_step"], x["rel_step"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
    n_q = len({j["question_idx"] for j in jobs})
    print(
        f"wrote {OUT} steps={len(jobs)} q={n_q} leftover_wins={n_left} high_wins={n_high}",
        flush=True,
    )


if __name__ == "__main__":
    main()
