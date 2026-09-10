#!/usr/bin/env python3
"""Why k=5/6 lose Acc vs k=4: classify every disagreeing question."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.inputs import same_answer  # noqa: E402
from plws.paths import PLWSPaths  # noqa: E402
from report_fullcot_puma_plws import (  # noqa: E402
    load_json,
    load_trials_by_question,
)
from report_k_ablate_7b import (  # noqa: E402
    DATASETS,
    MODEL,
    SEEDS,
    load_k_cell_scores,
    load_k_jobs,
)

OUT = ROOT / "results" / "reports" / "k_ablate_7b_k56_loss.json"
WAIT_RE = re.compile(r"(?m)^(?:[#>*\-\s]*)(?:\*\*)?(wait\b)", re.I)


def wait_starts(text: str) -> int:
    return len(WAIT_RE.findall(text or ""))


def prefix_at(trials: list[dict], step: int) -> str:
    for row in trials:
        if int(row.get("stopped_len") or 0) == int(step):
            return str(row.get("reasoning_prefix") or "")
    return ""


def classify(row: dict, k: int) -> str:
    if not row["ok4"] or row[f"ok{k}"]:
        raise ValueError("not a k4-win k-loss")
    if row["w4"] and not row[f"w{k}"]:
        return "dropout_back_to_wrong_fullcot"
    if (not row["w4"]) and row[f"w{k}"]:
        return "new_window_ruined_fullcot"
    if not row["w4"] and not row[f"w{k}"]:
        return "both_fullcot_impossible"
    l4, lk = row.get("left4"), row.get(f"left{k}")
    same_step = l4 is not None and lk is not None and l4 == lk
    same_lock = same_answer(row.get("lock_ans4"), row.get(f"lock_ans{k}"))
    later = l4 is not None and lk is not None and lk > l4
    lok4 = bool(row.get("left_ok4"))
    lokk = bool(row.get(f"left_ok{k}"))
    if same_step and same_lock:
        return "same_lock_later_write_differs"
    if later and same_lock and lok4 and lokk:
        return "later_same_correct_lock_then_ruined"
    if later and lok4 and not lokk:
        return "later_lock_revised_to_wrong"
    if later and (not lok4) and (not lokk):
        return "lost_rescue_of_wrong_lock"
    if later and (not lok4) and lokk:
        return "later_lock_became_right_but_final_wrong"
    if later and same_lock and lok4 and not lokk:
        return "later_lock_revised_to_wrong"
    return "other"


def reverse_classify(row: dict, k: int) -> str:
    if not row[f"ok{k}"] or row["ok4"]:
        raise ValueError("not a k-win k4-loss")
    if row["w4"] and not row[f"w{k}"]:
        return "dropout"
    if (not row["w4"]) and row[f"w{k}"]:
        return "new_window_helped"
    l4, lk = row.get("left4"), row.get(f"left{k}")
    same_step = l4 is not None and lk is not None and l4 == lk
    same_lock = same_answer(row.get("lock_ans4"), row.get(f"lock_ans{k}"))
    later = l4 is not None and lk is not None and lk > l4
    lok4 = bool(row.get("left_ok4"))
    lokk = bool(row.get(f"left_ok{k}"))
    if same_step and same_lock:
        return "same_lock_later_write_differs"
    if later and same_lock and lok4 and lokk:
        return "later_same_lock_we_ruined_k4"
    if later and (not lok4) and lokk:
        return "later_lock_revised_to_right"
    if later and lok4 and not lokk:
        return "later_lock_worse_but_k_final_ok"
    if later and (not lok4) and (not lokk):
        return "both_wrong_lock_k_rescued"
    return "other"


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    rows: list[dict] = []
    trials_cache: dict[tuple[str, int], dict[int, list]] = {}
    ds_map = {zh: en for en, zh, _ in DATASETS}

    for dataset, dszh, _n in DATASETS:
        for seed in SEEDS:
            official = {
                int(row["question_idx"]): row
                for row in load_json(paths.puma_statistics_path(MODEL, dataset, seed))
            }
            jobs = {k: load_k_jobs(paths, dataset, seed, k) for k in (4, 5, 6)}
            scores = {k: load_k_cell_scores(paths, dataset, seed, k) for k in (4, 5, 6)}
            tpath = paths.dense_trial_path(MODEL, dataset, seed)
            trials_cache[(dataset, seed)] = load_trials_by_question(tpath)
            for qid, info in official.items():
                item = {
                    "dataset": dszh,
                    "ds": dataset,
                    "seed": seed,
                    "qid": int(qid),
                    "full_ok": bool(info.get("original_correct")),
                }
                for k in (4, 5, 6):
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
                    item[f"ans{k}"] = str(rec.get("new_answer") or "")
                    item[f"lock_ans{k}"] = str(rec.get("old_answer") or "")
                rows.append(item)

    payload: dict = {"n": len(rows)}
    for k in (5, 6):
        lose = [r for r in rows if r["ok4"] and not r[f"ok{k}"]]
        gain = [r for r in rows if r[f"ok{k}"] and not r["ok4"]]
        lose_c = Counter(classify(r, k) for r in lose)
        gain_c = Counter(reverse_classify(r, k) for r in gain)
        by_ds_lose = defaultdict(Counter)
        by_ds_gain = defaultdict(Counter)
        wait_between = []
        for r in lose:
            by_ds_lose[r["dataset"]][classify(r, k)] += 1
            l4, lk = r.get("left4"), r.get(f"left{k}")
            if r.get("w4") and r.get(f"w{k}") and l4 is not None and lk is not None and lk > l4:
                trials = trials_cache[(r["ds"], r["seed"])].get(r["qid"], [])
                p4 = prefix_at(trials, l4)
                pk = prefix_at(trials, lk)
                extra = pk[len(p4) :] if pk.startswith(p4) else pk
                wait_between.append(
                    {
                        "kind": classify(r, k),
                        "ds": r["dataset"],
                        "extra_steps": lk - l4,
                        "wait": wait_starts(extra),
                    }
                )
        for r in gain:
            by_ds_gain[r["dataset"]][reverse_classify(r, k)] += 1
        later = [x for x in wait_between]
        payload[f"vs_k{k}"] = {
            "k4_only": len(lose),
            "kk_only": len(gain),
            "net": len(gain) - len(lose),
            "lose_reasons": dict(lose_c),
            "gain_reasons": dict(gain_c),
            "lose_by_dataset": {ds: dict(c) for ds, c in by_ds_lose.items()},
            "gain_by_dataset": {ds: dict(c) for ds, c in by_ds_gain.items()},
            "later_extra_steps_mean": (
                sum(x["extra_steps"] for x in later) / len(later) if later else None
            ),
            "later_wait_mean": (
                sum(x["wait"] for x in later) / len(later) if later else None
            ),
            "later_wait_by_kind": {
                kind: {
                    "n": sum(1 for x in later if x["kind"] == kind),
                    "mean_wait": (
                        sum(x["wait"] for x in later if x["kind"] == kind)
                        / max(1, sum(1 for x in later if x["kind"] == kind))
                    ),
                    "mean_steps": (
                        sum(x["extra_steps"] for x in later if x["kind"] == kind)
                        / max(1, sum(1 for x in later if x["kind"] == kind))
                    ),
                }
                for kind in sorted({x["kind"] for x in later})
            },
        }

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")
    for k in (5, 6):
        block = payload[f"vs_k{k}"]
        print(f"\n===== k=4 vs k={k}  official n={payload['n']} =====")
        print(f"k4 only {block['k4_only']}  k{k} only {block['kk_only']}  net {block['net']:+d}")
        print("lose", block["lose_reasons"])
        print("gain", block["gain_reasons"])
        print("later wait", block["later_wait_by_kind"])
        print("by ds lose", block["lose_by_dataset"])


if __name__ == "__main__":
    main()
