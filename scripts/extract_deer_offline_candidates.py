#!/usr/bin/env python3
"""Extract DEER-style stop candidates from trial answers (offline).

DEER rule (vllm_deer.py): stop at first probe with conf > λ
(default λ=0.95, policy=avg2 / geometric).

Probe schedule must match online Wait ATP (see build_deer_wait_probe_questions.py):
  at most max_judge_steps Wait ends — NOT every separate_steps paragraph.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-answers", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()

    trials = json.loads(args.trial_answers.read_text())
    by_q: dict[int, list] = defaultdict(list)
    for t in trials:
        by_q[int(t["question_idx"])].append(t)

    cands = []
    for qi in sorted(by_q):
        arr = sorted(by_q[qi], key=lambda x: int(x["stopped_len"]))
        n = int(arr[0]["original_len_reasoning_steps"])
        traj = []
        stop = None
        stop_conf = None
        tokens_trial = 0
        n_gen = 0
        for t in arr:
            conf = t.get("confidence")
            if conf is None or t.get("skipped"):
                traj.append(None)
                continue
            conf = float(conf)
            traj.append(conf)
            n_gen += 1
            tokens_trial += int(t.get("count_generated_tokens") or t.get("count_answer_tokens") or 0)
            # online vllm_deer uses strict '>' (not '>=')
            if stop is None and conf > args.threshold:
                stop = int(t["stopped_len"])
                stop_conf = conf
                break

        # trial tokens up to and including stop (or all if full)
        tokens_trial = 0
        n_gen = 0
        conf_traj = []
        for t in arr:
            sl = int(t["stopped_len"])
            conf = t.get("confidence")
            if conf is None or t.get("skipped"):
                conf_traj.append(None)
                continue
            conf_traj.append(float(conf))
            if stop is not None and sl > stop:
                break
            n_gen += 1
            # PUMA extract_final_candidates sums count_answer_tokens (boxed span).
            # Do not use count_generated_tokens (usually max_trial_tokens=30).
            if "count_answer_tokens" in t and t["count_answer_tokens"] is not None:
                tokens_trial += int(t["count_answer_tokens"])
            else:
                tokens_trial += int(t.get("count_generated_tokens") or 0)

        if stop is None:
            stop = n
            reason = "full_reasoning"
            # last available conf if any
            for c in reversed(conf_traj):
                if c is not None:
                    stop_conf = c
                    break
        else:
            reason = "deer_conf"

        # final_answer from stopping trial (for fallback)
        final_answer = ""
        for t in arr:
            if int(t["stopped_len"]) == stop and t.get("final_answer"):
                final_answer = t["final_answer"]
                break

        cands.append({
            "question_idx": qi,
            "stopped_len": stop,
            "original_len_reasoning_steps": n,
            "question": arr[0].get("question", ""),
            "skipped": False,
            "skip_reason": None,
            "similarity": None,
            "success": True,
            "generated_trial_answers": n_gen,
            "tokens_trial_answers": tokens_trial,
            "tokens_trial_answers_online": tokens_trial,
            "stop_reason": reason,
            "stop_confidence": stop_conf,
            "stop_threshold": args.threshold,
            "consecutive_confidences": None,
            "confidence_trajectory": conf_traj,
            "step_similarities": None,
            "final_answer": final_answer,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cands, indent=2) + "\n")
    n_early = sum(1 for c in cands if c["stop_reason"] != "full_reasoning")
    print(f"wrote {args.output} n={len(cands)} early_stop={n_early} full={len(cands)-n_early} λ={args.threshold}")


if __name__ == "__main__":
    main()
