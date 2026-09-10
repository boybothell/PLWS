#!/usr/bin/env python3
"""7B 一粒种子：等到第一扇 High 或 Mix 再压。第一扇 Low 先不压。

无 High/Mix 的题不导出，整集贴官方。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402
import report_dense_k4_lowconf_ceiling as low  # noqa: E402
import report_first_lock_room as room  # noqa: E402
import report_k4_second_lock as sl  # noqa: E402

DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
HM = ("high", "mix")


def first_hm(wins: list[dict]) -> dict | None:
    for win in wins:
        if win.get("kind") in HM:
            return win
    return None


def build_jobs(model: str, seed: int, datasets: tuple[str, ...] = DATASETS) -> list[dict]:
    jobs: list[dict] = []
    stats: Counter[str] = Counter()
    for dataset in datasets:
        trials_path = room.dense_trial_path(model, dataset, seed=seed)
        if not trials_path.is_file():
            continue
        official_path = dd.puma_stat_path(model, dataset, seed)
        official = (
            {int(r["question_idx"]): r for r in dd.load_json(official_path)}
            if official_path.is_file()
            else {}
        )
        gp = low.gpath(model, dataset, seed)
        gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
        by: dict[int, list] = {}
        for row in dd.load_json(trials_path):
            by.setdefault(int(row["question_idx"]), []).append(row)
        for qi in sorted(set(by) | set(official) | set(gmap)):
            trials = by.get(qi)
            if not trials:
                continue
            rows = rg.usable_rows(trials)
            if not rows:
                continue
            wins = sl.same_windows(rows)
            if not wins:
                stats["no_window"] += 1
                continue
            first = wins[0]
            win = first_hm(wins)
            if win is None:
                stats["low_only"] += 1
                continue
            left_row = rows[win["end"]]
            thought = str(left_row.get("reasoning_prefix") or "")
            question = str(left_row.get("question") or "")
            if not thought or not question:
                stats["no_prefix"] += 1
                continue
            info = official.get(qi) or {}
            g = gmap.get(qi) or {}
            last = max(trials, key=lambda x: int(x["stopped_len"]))
            gt = info.get("ground_truth") or g.get("ground_truth")
            original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
            orig_ok = (
                bool(info.get("original_correct"))
                if "original_correct" in info
                else bool(low.credit(original, gt, original, True))
            )
            high_after = next(
                (w for w in wins if w["kind"] == "high" and w["step"] > win["step"]),
                None,
            )
            delayed = first.get("kind") == "low"
            stats["first_hm" if not delayed else "delayed_hm"] += 1
            jobs.append(
                {
                    "uid": f"{model}:{dataset}:{seed}:{qi}",
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "question_idx": int(qi),
                    "left_step": int(win["step"]),
                    "kind": win["kind"],
                    "gate": "first_hm",
                    "first_kind": first.get("kind"),
                    "first_step": int(first["step"]),
                    "delayed": delayed,
                    "confidence": win.get("c"),
                    "left_ok": bool(low.credit(win["ans"], gt, original, orig_ok)),
                    "wait_helps": False,
                    "will_change": bool(
                        high_after is not None and not rg.same(win["ans"], high_after["ans"])
                    ),
                    "never_high": high_after is None,
                    "same_as_high": bool(
                        high_after is not None and rg.same(win["ans"], high_after["ans"])
                    ),
                    "host_ok": orig_ok,
                    "old_answer": str(win.get("ans") or ""),
                    "question": question,
                    "thought": thought,
                    "gt": gt,
                    "original": original,
                    "orig_ok": orig_ok,
                }
            )
    print(f"stats {dict(stats)} jobs={len(jobs)} {dict(Counter(j['dataset'] for j in jobs))}", flush=True)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=AE / "results/first_hm_gate/jobs/r1_7b_s42.jsonl",
    )
    args = parser.parse_args()
    jobs = build_jobs(args.model_tag, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(f"wrote {len(jobs)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
