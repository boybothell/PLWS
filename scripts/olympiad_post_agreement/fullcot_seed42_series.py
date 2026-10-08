#!/usr/bin/env python3
"""Full-CoT accuracy after the k=4 window, R1-7B seed 42.

Denominator is the fixed set of windowed questions. t=0 is the trial at
left_step. A finished question keeps its last trial answer.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if "PLWS_ROOT" in __import__("os").environ:
    ROOT = Path(__import__("os").environ["PLWS_ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tmp" / "PUMA" / "puma"))

from plws.grading import grade_many  # noqa: E402

OUT = ROOT / "tmp/plws_step_probe/count_bias/rho_0p98/r1_7b/fullcot_seed42_series.json"
CELLS = {
    "gpqa-diamond": ROOT
    / "results/upstream/dense_trials/dense_G_r1_7b/gpqa-diamond/dense_puma/trial_answers.json",
    "olympiadbench": ROOT
    / "results/upstream/dense_trials/dense_G_r1_7b/olympiadbench/dense_puma/trial_answers.json",
    "aime25": ROOT
    / "results/upstream/dense_trials/dense_G_r1_7b/aime25/seed_42/dense_puma/trial_answers.json",
}
JOBS = ROOT / "results/runs/plws/window_first/k_4/lexicon_core/r1_7b"
STATS = ROOT / "tmp/incoming_main/results/baselines/puma/puma_offline_r1_7b"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def curve(flags: list[list[bool]]) -> list[int]:
    n = len(flags)
    tmax = max(len(row) for row in flags)
    counts = []
    for step in range(tmax):
        counts.append(
            sum(row[step] if step < len(row) else row[-1] for row in flags)
        )
    if n == 0:
        raise RuntimeError("no windowed questions")
    return counts


def one(dataset: str, trials_path: Path) -> dict:
    jobs = load_jsonl(JOBS / dataset / "seed_42/jobs/firstwin.jsonl")
    print(f"[{dataset}] load trials", flush=True)
    trials = json.loads(trials_path.read_text())
    by_question: dict[int, dict[int, str]] = defaultdict(dict)
    for row in trials:
        by_question[int(row["question_idx"])][int(row["stopped_len"])] = row.get(
            "final_answer"
        )
    del trials
    pairs: list[tuple[str, str]] = []
    spans: list[int] = []
    for job in jobs:
        index = int(job["question_idx"])
        left = int(job["left_step"])
        steps = by_question.get(index)
        if not steps:
            raise SystemExit(f"{dataset} missing trials for {job['uid']}")
        post = sorted(step for step in steps if step >= left)
        if not post or post[0] != left or post != list(range(post[0], post[-1] + 1)):
            raise SystemExit(
                f"{dataset} gap uid={job['uid']} left={left} npost={len(post)}"
            )
        spans.append(len(post))
        gold = job["gt"]
        for step in post:
            pairs.append((steps[step], gold))
    print(f"[{dataset}] grade pairs={len(pairs)} questions={len(jobs)}", flush=True)
    graded = grade_many(pairs, workers=8, chunksize=16)
    flags: list[list[bool]] = []
    cursor = 0
    mismatch = 0
    for job, npost in zip(jobs, spans):
        row_flags: list[bool] = []
        for ok, error in graded[cursor : cursor + npost]:
            if error:
                raise SystemExit(f"{dataset} grade {job['uid']}: {error}")
            row_flags.append(bool(ok))
        cursor += npost
        if row_flags[0] != bool(job.get("left_ok")):
            mismatch += 1
        flags.append(row_flags)
    if cursor != len(graded):
        raise SystemExit(f"{dataset} grade cursor {cursor} != {len(graded)}")
    counts = curve(flags)
    stats = json.loads((STATS / dataset / "statistics.json").read_text())
    by_index = {int(row["question_idx"]): row for row in stats}
    window = {int(job["question_idx"]) for job in jobs}
    final_correct = sum(bool(row["original_correct"]) for row in stats)
    window_orig = sum(bool(by_index[index]["original_correct"]) for index in window)
    n = len(flags)
    return {
        "n": n,
        "n_all": len(stats),
        "final_correct": final_correct,
        "final_acc": 100.0 * final_correct / len(stats),
        "window_orig_correct": window_orig,
        "window_orig_acc": 100.0 * window_orig / n,
        "left_ok_mismatch": mismatch,
        "t0_correct": counts[0],
        "end_correct": counts[-1],
        "steps": len(counts),
        "counts": counts,
        "acc": [100.0 * count / n for count in counts],
        "trials": str(trials_path.relative_to(ROOT)),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file():
        payload = json.loads(OUT.read_text())
    else:
        payload = {
            "model": "r1_7b",
            "seed": 42,
            "denominator": "windowed questions, fixed; finished traces keep the last trial",
            "datasets": {},
        }
    for dataset, path in CELLS.items():
        if dataset in payload["datasets"] and payload["datasets"][dataset].get("counts"):
            print(f"[{dataset}] cached", flush=True)
            continue
        payload["datasets"][dataset] = one(dataset, path)
        OUT.write_text(json.dumps(payload))
        row = payload["datasets"][dataset]
        print(
            f"[{dataset}] n={row['n']} t0={row['t0_correct']}/{row['n']} "
            f"end={row['end_correct']}/{row['n']} steps={row['steps']} "
            f"final={row['final_correct']}/{row['n_all']} "
            f"left_ok_mismatch={row['left_ok_mismatch']}",
            flush=True,
        )
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
