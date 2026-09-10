#!/usr/bin/env python3
"""Compute independent G from dense (every-step) trial answers.

G = earliest stopped_len whose trial ≈ Full-CoT model_answer or ≈ GT.

Source: results/cross_probe_gate_r1_7b/<ds>/dense_puma/{trial_answers,answers}.json
Output: results/dense_G_r1_7b/<ds>/per_sample.json

This breaks the circular definition where G was taken from a sparse probe's
non-skip trials (which forces that probe's hit_G = 100%).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
PUMA = Path("/mnt/d/lsj/visual-latent-tts/repos/PUMA")
sys.path.insert(0, str(AE))
sys.path.insert(0, str(PUMA))

from attn_early_exit.answers import answers_equal  # noqa: E402

_WS = re.compile(r"\s+")


def _fast_eq(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    x = _WS.sub("", str(a).strip().lower())
    y = _WS.sub("", str(b).strip().lower())
    return bool(x) and x == y


def _eq(a: str | None, b: str | None) -> bool:
    if _fast_eq(a, b):
        return True
    return answers_equal(a, b)


def _one_question(args: tuple) -> dict:
    qi, A, gt, trials = args
    trials = sorted(trials, key=lambda x: x[0])
    G = None
    n_checked = 0
    for step, fa in trials:
        n_checked += 1
        if _eq(fa, A) or _eq(fa, gt):
            G = int(step)
            break
    n_steps = max((t[0] for t in trials), default=0)
    return {
        "question_idx": int(qi),
        "G": G,
        "A_final": A,
        "ground_truth": gt,
        "n_steps": int(n_steps),
        "n_trials_checked": n_checked,
        "source": "dense_puma",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--dense-root",
        type=Path,
        default=None,
        help="default: results/cross_probe_gate_r1_7b/<ds>/dense_puma",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: results/dense_G_r1_7b/<ds>/per_sample.json",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dense = args.dense_root or (
        AE / "results" / "cross_probe_gate_r1_7b" / args.dataset / "dense_puma"
    )
    out = args.out or (
        AE / "results" / "dense_G_r1_7b" / args.dataset / "per_sample.json"
    )

    answers = json.loads((dense / "answers.json").read_text())
    trials_raw = json.loads((dense / "trial_answers.json").read_text())
    by: dict[int, list[tuple[int, str | None]]] = defaultdict(list)
    for e in trials_raw:
        if not e.get("success", True):
            continue
        if e.get("skipped"):
            continue
        qi = int(e["question_idx"])
        by[qi].append((int(e["stopped_len"]), e.get("final_answer")))

    jobs = []
    for i, ans in enumerate(answers):
        qi = i + 1
        A = ans.get("model_answer")
        gt = ans.get("ground_truth_answer")
        jobs.append((qi, A, gt, by.get(qi, [])))

    t0 = time.time()
    rows: list[dict] = []
    if args.workers <= 1:
        for j in jobs:
            rows.append(_one_question(j))
            if len(rows) % 50 == 0:
                print(f"  {args.dataset} {len(rows)}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one_question, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % 50 == 0:
                    print(f"  {args.dataset} {done}/{len(jobs)}", flush=True)

    rows.sort(key=lambda r: int(r["question_idx"]))
    n_g = sum(1 for r in rows if r["G"] is not None)
    n_g1 = sum(1 for r in rows if r["G"] == 1)
    n_g2 = sum(1 for r in rows if r["G"] is not None and int(r["G"]) >= 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": args.dataset,
        "dense_root": str(dense),
        "definition": "earliest dense trial ≈ Full-CoT model_answer or ≈ GT",
        "n": len(rows),
        "n_with_G": n_g,
        "n_G1": n_g1,
        "n_G_ge2": n_g2,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"WROTE {out} {meta}", flush=True)


if __name__ == "__main__":
    main()
