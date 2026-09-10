#!/usr/bin/env python3
"""第一扇 High 之前已经出现过 Mix 的题：切到第一扇 High 再压。

旧 High leftover 只收「第一扇就是 High」。
first-HM 在第一扇 Mix 就压，不会等到后面的 High。
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


def first_high_after_mix(wins: list[dict]) -> dict | None:
    """第一扇 High 之前已经有 Mix。先 High 再 Mix 的题不算。"""
    mix = next((w for w in wins if w.get("kind") == "mix"), None)
    high = next((w for w in wins if w.get("kind") == "high"), None)
    if mix is None or high is None:
        return None
    if int(mix["step"]) >= int(high["step"]):
        return None
    return {**high, "mix_step": int(mix["step"]), "mix_ans": mix.get("ans")}


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
            win = first_high_after_mix(wins)
            if win is None:
                stats["no_mix_then_high"] += 1
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
            first = wins[0]
            mix = next(w for w in wins if w.get("kind") == "mix")
            stats[f"first_{first.get('kind')}"] += 1
            jobs.append(
                {
                    "uid": f"{model}:{dataset}:{seed}:{qi}",
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "question_idx": int(qi),
                    "left_step": int(win["step"]),
                    "kind": "high",
                    "gate": "mix_then_high",
                    "first_kind": first.get("kind"),
                    "first_step": int(first["step"]),
                    "mix_step": int(mix["step"]),
                    "mix_ok": bool(low.credit(mix.get("ans"), gt, original, orig_ok)),
                    "delayed": first.get("kind") != "high",
                    "confidence": win.get("c"),
                    "left_ok": bool(low.credit(win["ans"], gt, original, orig_ok)),
                    "wait_helps": False,
                    "will_change": not rg.same(mix.get("ans"), win["ans"]),
                    "never_high": False,
                    "same_as_high": bool(rg.same(mix.get("ans"), win["ans"])),
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
        default=AE / "results/mix_then_high/jobs/r1_7b_s42.jsonl",
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
