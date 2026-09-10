#!/usr/bin/env python3
"""CoT-level reason k=4 is the first settled same-answer run."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_leftover_suppress_jobs import same_windows_k  # noqa: E402
from plws.inputs import (  # noqa: E402
    answer_credit,
    same_answer,
    usable_trial_rows,
)
from plws.paths import PLWSPaths  # noqa: E402
from report_fullcot_puma_plws import load_json, load_trials_by_question  # noqa: E402
from report_k_ablate_7b import DATASETS, KS, MODEL, SEEDS  # noqa: E402

OUT = ROOT / "results" / "reports" / "k_ablate_7b_cot.json"
WAIT_RE = re.compile(r"(?m)^(?:[#>*\-\s]*)(?:\*\*)?(wait\b)", re.I)
MSS = 10


def wait_starts(text: str) -> int:
    return len(WAIT_RE.findall(text or ""))


def first_streak(rows: list[dict]) -> dict | None:
    n = len(rows)
    for i in range(n):
        if int(rows[i]["stopped_len"]) < MSS:
            continue
        ans = rows[i].get("final_answer")
        if not ans:
            continue
        j = i
        while (
            j + 1 < n
            and int(rows[j + 1]["stopped_len"]) == int(rows[j]["stopped_len"]) + 1
            and same_answer(ans, rows[j + 1].get("final_answer"))
        ):
            j += 1
        if j - i + 1 >= 2:
            return {
                "start": int(rows[i]["stopped_len"]),
                "end": int(rows[j]["stopped_len"]),
                "length": j - i + 1,
                "ans": str(ans),
            }
    return None


def lock_on_run(win: dict | None, run: dict | None) -> bool:
    if win is None or run is None:
        return False
    return run["start"] <= int(win["step"]) <= run["end"] and same_answer(
        win["ans"], run["ans"]
    )


def row_at_step(rows: list[dict], step: int) -> dict | None:
    for row in rows:
        if int(row["stopped_len"]) == int(step):
            return row
    return None


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    all_q: list[dict] = []
    for dataset, dszh, _n in DATASETS:
        for seed in SEEDS:
            official = {
                int(row["question_idx"]): row
                for row in load_json(paths.puma_statistics_path(MODEL, dataset, seed))
            }
            trials = load_trials_by_question(paths.dense_trial_path(MODEL, dataset, seed))
            for qid, info in official.items():
                rows = usable_trial_rows(trials.get(int(qid), []))
                run = first_streak(rows)
                wins = {k: (same_windows_k(rows, k) or [None])[0] for k in KS}
                # same_windows_k returns [] not falsy in a useful way
                wins = {}
                for k in KS:
                    found = same_windows_k(rows, k)
                    wins[k] = found[0] if found else None
                item = {
                    "ds": dszh,
                    "seed": seed,
                    "qid": int(qid),
                    "full_ok": bool(info.get("original_correct")),
                    "full_ans": str(info.get("original_answer") or ""),
                    "gt": info.get("ground_truth"),
                    "run_len": run["length"] if run else 0,
                    "run_start": run["start"] if run else None,
                    "run_end": run["end"] if run else None,
                    "run_ans": run["ans"] if run else "",
                    "run_eq_full": bool(
                        run and same_answer(run["ans"], info.get("original_answer"))
                    ),
                    "run_ok": bool(
                        run
                        and answer_credit(
                            run["ans"],
                            info.get("ground_truth"),
                            info.get("original_answer"),
                            bool(info.get("original_correct")),
                        )
                    ),
                }
                last = rows[-1] if rows else None
                full_wait = wait_starts(
                    str((last or {}).get("reasoning_prefix") or "")
                )
                item["full_wait"] = full_wait
                for k in KS:
                    win = wins[k]
                    item[f"w{k}"] = win is not None
                    if win is None:
                        continue
                    item[f"lock{k}"] = int(win["step"])
                    item[f"lock_ans{k}"] = str(win["ans"] or "")
                    item[f"on_first{k}"] = lock_on_run(win, run)
                    prefix = str(
                        (row_at_step(rows, win["step"]) or {}).get("reasoning_prefix")
                        or ""
                    )
                    item[f"wait_before{k}"] = wait_starts(prefix)
                    item[f"wait_after{k}"] = max(0, full_wait - wait_starts(prefix))
                all_q.append(item)

    have_run = [q for q in all_q if q["run_len"] >= 2]
    inter = [q for q in all_q if all(q.get(f"w{k}") for k in KS)]
    hist = Counter(q["run_len"] if q["run_len"] <= 8 else 9 for q in have_run)
    by_len: dict[int, dict] = {}
    for length in range(2, 10):
        chunk = [
            q
            for q in have_run
            if (q["run_len"] == length if length < 9 else q["run_len"] >= 9)
        ]
        if not chunk:
            continue
        by_len[length] = {
            "n": len(chunk),
            "run_ok": sum(1 for q in chunk if q["run_ok"]),
            "run_eq_full": sum(1 for q in chunk if q["run_eq_full"]),
            "full_ok": sum(1 for q in chunk if q["full_ok"]),
        }

    def lock_story(group: list[dict]) -> dict:
        out = {}
        for k in KS:
            locked = [q for q in group if q.get(f"w{k}")]
            on_first = [q for q in locked if q.get(f"on_first{k}")]
            later = [q for q in locked if q.get(f"on_first{k}") is False]
            later_changed = [
                q
                for q in later
                if q.get("run_ans")
                and q.get(f"lock_ans{k}")
                and not same_answer(q["run_ans"], q[f"lock_ans{k}"])
            ]
            out[f"k{k}"] = {
                "n": len(locked),
                "on_first_run": len(on_first),
                "later_run": len(later),
                "later_answer_changed": len(later_changed),
                "mean_wait_before": (
                    sum(q[f"wait_before{k}"] for q in locked) / len(locked)
                    if locked
                    else None
                ),
                "mean_wait_after": (
                    sum(q[f"wait_after{k}"] for q in locked) / len(locked)
                    if locked
                    else None
                ),
            }
        return out

    # first-run length vs whether k=4/5/6 can use it
    use = {}
    for k in KS:
        use[f"k{k}"] = {}
        for length in range(2, 9):
            chunk = [q for q in have_run if q["run_len"] == length]
            if not chunk:
                continue
            use[f"k{k}"][str(length)] = {
                "n": len(chunk),
                "windowed": sum(1 for q in chunk if q.get(f"w{k}")),
                "on_first": sum(1 for q in chunk if q.get(f"on_first{k}")),
            }

    payload = {
        "n_official": len(all_q),
        "n_with_first_run": len(have_run),
        "n_intersection": len(inter),
        "first_run_hist": {str(k): hist[k] for k in sorted(hist)},
        "first_run_quality": {
            str(k): {
                **v,
                "run_ok_rate": 100.0 * v["run_ok"] / v["n"],
                "run_eq_full_rate": 100.0 * v["run_eq_full"] / v["n"],
            }
            for k, v in by_len.items()
        },
        "locks_all_with_run": lock_story(have_run),
        "locks_intersection": lock_story(inter),
        "first_run_usable_by_k": use,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")
    print("first-run length hist (2..8, 9=9+)", dict(sorted(hist.items())))
    print("\nfirst-run quality")
    for length, v in sorted(by_len.items()):
        label = str(length) if length < 9 else "9+"
        print(
            f"  L={label:3} n={v['n']:4}  "
            f"run_ok={100*v['run_ok']/v['n']:.1f}%  "
            f"eq_full={100*v['run_eq_full']/v['n']:.1f}%"
        )
    print("\nlock on first run? (all with a first run)")
    for k in KS:
        s = payload["locks_all_with_run"][f"k{k}"]
        print(
            f"  k={k} win={s['n']} on_first={s['on_first_run']} "
            f"later={s['later_run']} later_changed={s['later_answer_changed']} "
            f"wait_before={s['mean_wait_before']:.2f} wait_after={s['mean_wait_after']:.2f}"
        )
    print("\nlock on first run? (4802 intersection)")
    for k in KS:
        s = payload["locks_intersection"][f"k{k}"]
        print(
            f"  k={k} on_first={s['on_first_run']} later={s['later_run']} "
            f"later_changed={s['later_answer_changed']} "
            f"wait_before={s['mean_wait_before']:.2f} wait_after={s['mean_wait_after']:.2f}"
        )


if __name__ == "__main__":
    main()
