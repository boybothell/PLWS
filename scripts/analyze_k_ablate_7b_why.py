#!/usr/bin/env python3
"""Decompose why 7B leftover Acc peaks at k=4 versus k=5/6."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.paths import PLWSPaths  # noqa: E402
from report_fullcot_puma_plws import load_json  # noqa: E402
from report_k_ablate_7b import (  # noqa: E402
    DATASETS,
    KS,
    MODEL,
    SEEDS,
    load_k_cell_scores,
    load_k_jobs,
)

OUT = ROOT / "results" / "reports" / "k_ablate_7b_why.json"


def pct(num: float, den: int) -> float | None:
    if den <= 0:
        return None
    return 100.0 * num / den


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    rows: list[dict] = []
    for dataset, dszh, _expect_n in DATASETS:
        for seed in SEEDS:
            official = {
                int(row["question_idx"]): row
                for row in load_json(paths.puma_statistics_path(MODEL, dataset, seed))
            }
            jobs = {k: load_k_jobs(paths, dataset, seed, k) for k in KS}
            scores = {k: load_k_cell_scores(paths, dataset, seed, k) for k in KS}
            for qid, info in official.items():
                item = {
                    "dataset": dszh,
                    "seed": seed,
                    "qid": int(qid),
                    "full_ok": bool(info.get("original_correct")),
                }
                for k in KS:
                    rec = scores[k].get((dataset, int(qid)))
                    job = jobs[k].get(int(qid))
                    if rec is None:
                        item[f"w{k}"] = False
                        item[f"ok{k}"] = item["full_ok"]
                        continue
                    item[f"w{k}"] = True
                    item[f"ok{k}"] = bool(rec.get("new_gold_ok"))
                    item[f"left_ok{k}"] = bool(rec.get("left_ok"))
                    left = rec.get("left_step")
                    if left is None and job is not None:
                        left = job.get("left_step")
                    item[f"left{k}"] = int(left) if left is not None else None
                    item[f"will_change{k}"] = bool(rec.get("will_change"))
                    item[f"ans{k}"] = str(rec.get("new_answer") or "")
                    item[f"lock_ans{k}"] = str(rec.get("old_answer") or "")
                rows.append(item)

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    by_ds["ALL"] = rows

    per_k = {}
    for dszh, group in by_ds.items():
        n = len(group)
        per_k[dszh] = {}
        for k in KS:
            win = [r for r in group if r[f"w{k}"]]
            un = [r for r in group if not r[f"w{k}"]]
            flips_up = sum(1 for r in win if r[f"ok{k}"] and not r["full_ok"])
            flips_down = sum(1 for r in win if (not r[f"ok{k}"]) and r["full_ok"])
            left_ok = [r for r in win if r.get(f"left_ok{k}") is True]
            left_wrong = [r for r in win if r.get(f"left_ok{k}") is False]
            rescue = sum(
                1
                for r in win
                if r.get(f"left_ok{k}") is False and r[f"ok{k}"]
            )
            ruin = sum(
                1
                for r in win
                if r.get(f"left_ok{k}") is True and not r[f"ok{k}"]
            )
            steps = [r[f"left{k}"] for r in win if r.get(f"left{k}") is not None]
            per_k[dszh][f"k{k}"] = {
                "n": n,
                "windowed": len(win),
                "window_rate": pct(len(win), n),
                "acc": pct(sum(1 for r in group if r[f"ok{k}"]), n),
                "full_acc": pct(sum(1 for r in group if r["full_ok"]), n),
                "flips_up": flips_up,
                "flips_down": flips_down,
                "net_flips": flips_up - flips_down,
                "windowed_acc": pct(sum(1 for r in win if r[f"ok{k}"]), len(win)),
                "windowed_full_acc": pct(
                    sum(1 for r in win if r["full_ok"]), len(win)
                ),
                "unwindowed_acc": pct(sum(1 for r in un if r["full_ok"]), len(un)),
                "left_ok_rate": pct(len(left_ok), len(win)),
                "lock_then_rescue": rescue,
                "lock_then_ruin": ruin,
                "lock_wrong": len(left_wrong),
                "median_left_step": median(steps) if steps else None,
                "mean_left_step": mean(steps),
            }

    def vs_k4(group: list[dict], k: int) -> dict:
        shared = [r for r in group if r["w4"] and r[f"w{k}"]]
        drop = [r for r in group if r["w4"] and not r[f"w{k}"]]
        extra = [r for r in group if (not r["w4"]) and r[f"w{k}"]]
        same_lock = [
            r
            for r in shared
            if r.get("left4") is not None
            and r.get(f"left{k}") is not None
            and r["left4"] == r[f"left{k}"]
        ]
        later = [
            r
            for r in shared
            if r.get("left4") is not None
            and r.get(f"left{k}") is not None
            and r[f"left{k}"] > r["left4"]
        ]
        earlier = [
            r
            for r in shared
            if r.get("left4") is not None
            and r.get(f"left{k}") is not None
            and r[f"left{k}"] < r["left4"]
        ]

        def delta(items: list[dict]) -> dict:
            if not items:
                return {
                    "n": 0,
                    "k4_ok": 0,
                    "kk_ok": 0,
                    "full_ok": 0,
                    "k4_only": 0,
                    "kk_only": 0,
                    "net_vs_k4": 0,
                }
            k4_ok = sum(1 for r in items if r["ok4"])
            kk_ok = sum(1 for r in items if r[f"ok{k}"])
            return {
                "n": len(items),
                "k4_ok": k4_ok,
                "kk_ok": kk_ok,
                "full_ok": sum(1 for r in items if r["full_ok"]),
                "k4_only": sum(1 for r in items if r["ok4"] and not r[f"ok{k}"]),
                "kk_only": sum(1 for r in items if r[f"ok{k}"] and not r["ok4"]),
                "net_vs_k4": kk_ok - k4_ok,
                "k4_acc": pct(k4_ok, len(items)),
                "kk_acc": pct(kk_ok, len(items)),
                "full_acc": pct(sum(1 for r in items if r["full_ok"]), len(items)),
            }

        later_lock_wrong_now = sum(
            1
            for r in later
            if r.get("left_ok4") is True and r.get(f"left_ok{k}") is False
        )
        later_lock_right_now = sum(
            1
            for r in later
            if r.get("left_ok4") is False and r.get(f"left_ok{k}") is True
        )
        later_ans_change = sum(
            1 for r in later if r.get("ans4", "") != r.get(f"ans{k}", "")
        )
        later_step_delta = [
            r[f"left{k}"] - r["left4"]
            for r in later
            if r.get("left4") is not None and r.get(f"left{k}") is not None
        ]
        drop_k4_rescued = sum(
            1 for r in drop if r["ok4"] and not r["full_ok"]
        )
        drop_k4_ruined = sum(
            1 for r in drop if (not r["ok4"]) and r["full_ok"]
        )
        return {
            "shared": {
                **delta(shared),
                "same_lock": delta(same_lock),
                "later_lock": {
                    **delta(later),
                    "lock_became_wrong": later_lock_wrong_now,
                    "lock_became_right": later_lock_right_now,
                    "final_answer_changed": later_ans_change,
                    "median_extra_steps": (
                        median(later_step_delta) if later_step_delta else None
                    ),
                    "mean_extra_steps": mean(later_step_delta),
                },
                "earlier_lock": delta(earlier),
            },
            "dropout": {
                **delta(drop),
                "k4_rescued_lost": drop_k4_rescued,
                "k4_ruined_recovered": drop_k4_ruined,
                "net_if_keep_fullcot": drop_k4_ruined - drop_k4_rescued,
            },
            "extra_window": delta(extra),
        }

    compare = {}
    for dszh, group in by_ds.items():
        compare[dszh] = {f"k{k}": vs_k4(group, k) for k in KS if k != 4}

    # monotonicity: windowed(k+1) subset windowed(k)?
    mono = {}
    for dszh, group in by_ds.items():
        mono[dszh] = {}
        for a, b in zip(KS, KS[1:]):
            only_b = sum(1 for r in group if r[f"w{b}"] and not r[f"w{a}"])
            only_a = sum(1 for r in group if r[f"w{a}"] and not r[f"w{b}"])
            both = sum(1 for r in group if r[f"w{a}"] and r[f"w{b}"])
            mono[dszh][f"k{a}_to_k{b}"] = {
                "both": both,
                "lost": only_a,
                "gained": only_b,
            }

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "scripts/analyze_k_ablate_7b_why.py",
        "n_questions": len(rows),
        "per_k": per_k,
        "vs_k4": compare,
        "window_monotonicity": mono,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")

    def show(dszh: str) -> None:
        print(f"\n===== {dszh} =====")
        for k in KS:
            cell = per_k[dszh][f"k{k}"]
            print(
                f"k={k} win={cell['windowed']} rate={cell['window_rate']:.1f}% "
                f"acc={cell['acc']:.2f} net_flip={cell['net_flips']:+d} "
                f"(up {cell['flips_up']} down {cell['flips_down']}) "
                f"left_ok={cell['left_ok_rate']:.1f}% "
                f"rescue={cell['lock_then_rescue']} ruin={cell['lock_then_ruin']} "
                f"med_step={cell['median_left_step']}"
            )
        for k in (2, 3, 5, 6):
            c = compare[dszh][f"k{k}"]
            s = c["shared"]
            d = c["dropout"]
            print(
                f"  vs k=4 | k={k} shared n={s['n']} net={s['net_vs_k4']:+d} "
                f"(same_lock {s['same_lock']['n']} net={s['same_lock']['net_vs_k4']:+d}; "
                f"later {s['later_lock']['n']} net={s['later_lock']['net_vs_k4']:+d} "
                f"lock_wrong+{s['later_lock']['lock_became_wrong']} "
                f"lock_right+{s['later_lock']['lock_became_right']}) "
                f"drop n={d['n']} net={d['net_vs_k4']:+d} "
                f"lost_rescue={d['k4_rescued_lost']} "
                f"recovered_ruin={d['k4_ruined_recovered']} "
                f"extra={c['extra_window']['n']}/{c['extra_window']['net_vs_k4']:+d}"
            )

    for name in ["ALL", "MATH", "OlympiadBench", "GPQA-Diamond", "AIME24", "AIME25"]:
        show(name)


if __name__ == "__main__":
    main()
