#!/usr/bin/env python3
"""Evaluate zero-call trajectory signals derived from dense trial histories."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import (  # noqa: E402
    DATASETS,
    _fast_eq,
    auc,
    finite,
    load_refs,
    tpr_at_fpr,
)
from score_confcal_v1 import load_jobs  # noqa: E402


def entropy(probabilities: list[float]) -> float:
    return float(-sum(p * math.log(p) for p in probabilities if p > 0.0))


def answer_entropy(history: list[str]) -> float:
    """Normalized entropy of answer clusters within a causal history."""
    clusters: list[tuple[str, int]] = []
    for answer in history:
        for index, (representative, count) in enumerate(clusters):
            if _fast_eq(answer, representative):
                clusters[index] = (representative, count + 1)
                break
        else:
            clusters.append((answer, 1))
    if len(clusters) <= 1:
        return 0.0
    total = sum(count for _, count in clusters)
    return entropy([count / total for _, count in clusters]) / math.log(len(clusters))


def attach_signals(
    trials: dict[int, dict[int, dict[str, Any]]],
    *,
    decay: float,
    window: int,
    high_threshold: float,
    switch_decay: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qi, step_map in trials.items():
        prior_answer = ""
        ema = float("nan")
        confidences: list[float] = []
        answers: list[str] = []
        for step, trial in sorted(step_map.items()):
            confidence = finite(trial.get("confidence"))
            answer = str(trial.get("final_answer") or "")
            changed = bool(prior_answer and not _fast_eq(answer, prior_answer))
            if not math.isfinite(ema):
                ema = confidence
            elif changed:
                # A new proposed answer invalidates much of the evidence for
                # the preceding answer, while retaining a small causal memory.
                ema = switch_decay * ema + (1.0 - switch_decay) * confidence
            else:
                ema = decay * confidence + (1.0 - decay) * ema
            confidences.append(confidence)
            answers.append(answer)
            recent = confidences[-window:]
            robust = min(recent) if recent else float("nan")
            smoothness = -float(np.mean(np.abs(np.diff(recent)))) if len(recent) > 1 else 0.0
            stable_high = float(min(value - high_threshold for value in recent)) if recent else float("nan")
            rows.append(
                {
                    "question_idx": qi,
                    "decision_step": step,
                    "answer": answer,
                    "geo_conf": confidence,
                    "traj_ema": ema,
                    "traj_window_min": robust,
                    "traj_smoothness": smoothness,
                    "traj_stable_high": stable_high,
                    "traj_answer_stability": 1.0 - answer_entropy(answers[-window:]),
                    "answer_changed": int(changed),
                }
            )
            prior_answer = answer
    return rows


def label_rows(rows: list[dict[str, Any]], dataset: str) -> None:
    a_final, gold = load_refs(dataset)
    first_g: dict[int, int] = {}
    for row in sorted(rows, key=lambda item: (item["question_idx"], item["decision_step"])):
        qi = row["question_idx"]
        is_g = _fast_eq(row["answer"], a_final.get(qi)) or _fast_eq(row["answer"], gold.get(qi))
        row["is_g"] = int(is_g)
        if is_g and qi not in first_g:
            first_g[qi] = row["decision_step"]
    for row in rows:
        row["is_first_g"] = int(row["is_g"] and first_g.get(row["question_idx"]) == row["decision_step"])


def evaluate(rows: list[dict[str, Any]], signal: str) -> dict[str, float | int]:
    valid = [row for row in rows if math.isfinite(row[signal])]
    labels = np.asarray([row["is_g"] for row in valid], dtype=int)
    scores = np.asarray([row[signal] for row in valid], dtype=float)
    first = [row for row in valid if row["is_first_g"] or not row["is_g"]]
    return {
        "n": len(valid),
        "n_g": int(labels.sum()),
        "auroc": auc(labels, scores),
        "tpr_fpr1": tpr_at_fpr(labels, scores, 0.01),
        "tpr_fpr5": tpr_at_fpr(labels, scores, 0.05),
        "first_g_auroc": auc(
            np.asarray([row["is_first_g"] for row in first], dtype=int),
            np.asarray([row[signal] for row in first], dtype=float),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay", type=float, default=0.5)
    parser.add_argument("--switch-decay", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--high-threshold", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=AE / "results/confcal_judge/v2/trajectory_analysis.json")
    args = parser.parse_args()

    if not 0.0 <= args.decay <= 1.0 or not 0.0 <= args.switch_decay <= 1.0:
        raise ValueError("decay values must be in [0, 1]")
    if args.window < 1:
        raise ValueError("window must be positive")

    report: dict[str, Any] = {"config": vars(args), "datasets": {}}
    for dataset in args.datasets:
        _, trials = load_jobs(dataset, args.seed)
        rows = attach_signals(
            trials,
            decay=args.decay,
            window=args.window,
            high_threshold=args.high_threshold,
            switch_decay=args.switch_decay,
        )
        label_rows(rows, dataset)
        signals = (
            "geo_conf",
            "traj_ema",
            "traj_window_min",
            "traj_smoothness",
            "traj_stable_high",
            "traj_answer_stability",
        )
        report["datasets"][dataset] = {signal: evaluate(rows, signal) for signal in signals}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("| dataset | signal | AUROC | TPR@1% | TPR@5% | first-G AUROC |")
    print("|---|---|---:|---:|---:|---:|")
    for dataset, block in report["datasets"].items():
        for signal, metrics in block.items():
            print(
                f"| {dataset} | {signal} | {metrics['auroc']:.3f} | "
                f"{metrics['tpr_fpr1']:.3f} | {metrics['tpr_fpr5']:.3f} | {metrics['first_g_auroc']:.3f} |"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
