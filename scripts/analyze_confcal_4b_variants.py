#!/usr/bin/env python3
"""Evaluate frozen-Qwen4B confidence readouts on the full MATH-500 set.

No cal/held split.  Scores are evaluated on actual dense trajectories.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE))
from attn_early_exit.answers import answers_equal  # noqa: E402

TRIALS = AE / "results/dense_G_r1_7b/math-500/dense_puma/trial_answers.json"
ANSWERS = AE / "results/dense_G_r1_7b/math-500/dense_puma/answers.json"
PER_SAMPLE = AE / "results/dense_G_r1_7b/math-500/per_sample.json"
ROOT = AE / "results/confcal_judge/qwen4b_variants"
SCORE_DIRS = (ROOT / "math-500_s42", ROOT / "math-500_s42_cal")
POLICY = ROOT / "selected_policy.json"
TABLE = AE / "tables/probe_confcal_4b_variants.md"


def all_qids() -> set[int]:
    return {int(row["question_idx"]) for row in json.loads(PER_SAMPLE.read_text())}


def load_scores() -> dict[tuple[int, int, str], dict[str, Any]]:
    rows = {}
    for path in SCORE_DIRS:
        if not path.is_dir():
            continue
        for shard in sorted(path.glob("scores*.jsonl")):
            with shard.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        rows[(int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))] = row
    return rows


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def signal_values(row: dict[str, Any]) -> dict[str, float]:
    b = row.get("B_yesno") or {}
    eyn = row.get("EAGLE_yn") or {}
    three = row.get("B_3way") or {}
    ten = row.get("EAGLE_10bin") or {}
    steer = row.get("SteerConf") or {}
    cmp_cme = row.get("CMP_CME") or {}
    correct = safe_float(three.get("Correct"))
    incorrect = safe_float(three.get("Incorrect"))
    return {
        "B_final": safe_float(b.get("yes")),
        "B_3way_correct": correct / (correct + incorrect)
        if math.isfinite(correct) and math.isfinite(incorrect) and correct + incorrect > 0
        else float("nan"),
        "B_3way_evidence": 1.0 - safe_float(three.get("Insufficient")),
        "EAGLE_yn": safe_float(eyn.get("yes")),
        "EAGLE_10bin_final": safe_float(ten.get("final_expected")),
        "EAGLE_10bin": safe_float(ten.get("eagle_expected")),
        "SteerConf_mean": safe_float(steer.get("mean")),
        "SteerConf_min": safe_float(steer.get("min")),
        "SteerConf_stable": safe_float(steer.get("mean")) * (1.0 - safe_float(steer.get("spread"))),
        # Lower cross-model surprise/entropy means more confidence.
        "CMP_neg": -safe_float(cmp_cme.get("cmp")),
        "CME_neg": -safe_float(cmp_cme.get("cme")),
    }


def refs() -> tuple[dict[int, str], dict[int, str]]:
    af = {int(r["question_idx"]): r.get("A_final") for r in json.loads(PER_SAMPLE.read_text())}
    gt = {}
    for i, row in enumerate(json.loads(ANSWERS.read_text())):
        gt[int(row.get("question_idx") or i + 1)] = row.get("ground_truth_answer")
    return af, gt


def build_trajectories() -> tuple[dict[int, list[dict]], dict[int, str], dict[int, str]]:
    qids = all_qids()
    scores = load_scores()
    af, gt = refs()
    answer_cache: dict[tuple[int, str], tuple[bool, bool]] = {}
    out: dict[int, list[dict]] = defaultdict(list)
    for trial in json.loads(TRIALS.read_text()):
        qi = int(trial["question_idx"])
        if qi not in qids:
            continue
        ans = str(trial.get("final_answer") or "")
        key = (qi, ans)
        if key not in answer_cache:
            answer_cache[key] = (
                bool(answers_equal(ans, af.get(qi))),
                bool(answers_equal(ans, gt.get(qi))),
            )
        is_af, is_gt = answer_cache[key]
        score = scores.get((qi, int(trial["stopped_len"]), ans))
        signals = signal_values(score) if score else {}
        out[qi].append(
            {
                "step": int(trial["stopped_len"]),
                "answer": ans,
                "g": is_af or is_gt,
                "gt": is_gt,
                "reasoning_tokens": int(trial.get("count_reasoning_tokens") or 0),
                "answer_tokens": int(trial.get("count_answer_tokens") or 0),
                "scores": signals,
            }
        )
    for items in out.values():
        items.sort(key=lambda x: x["step"])
    return dict(out), af, gt


def stop(items: list[dict], signal: str, tau: float, k: int, eps: float) -> tuple[dict, str]:
    usable = [x for x in items if math.isfinite(x["scores"].get(signal, float("nan")))]
    for i, first in enumerate(usable):
        score = first["scores"][signal]
        if score < tau:
            continue
        window = usable[i : i + k]
        if len(window) < k or window[-1]["step"] < 10:
            continue
        if all(
            x["answer"] == first["answer"] and x["scores"][signal] >= score - eps
            for x in window[1:]
        ):
            return window[-1], "confidence_consecutive"
    return items[-1], "full_reasoning"


def evaluate(trajectories: dict[int, list[dict]], signal: str, tau: float, k: int, eps: float) -> dict[str, Any]:
    correct = delivered_g = early = false_stop = first_g_stop = 0
    toks, steps, g_delay, first_g_seen = [], [], [], 0
    for items in trajectories.values():
        chosen, reason = stop(items, signal, tau, k, eps)
        correct += int(chosen["gt"])
        delivered_g += int(chosen["g"])
        early += int(reason == "confidence_consecutive")
        false_stop += int(reason == "confidence_consecutive" and not chosen["g"])
        steps.append(chosen["step"])
        toks.append(
            chosen["reasoning_tokens"]
            + sum(x["answer_tokens"] for x in items if x["step"] <= chosen["step"])
        )
        first_g = next((x for x in items if x["g"]), None)
        if first_g:
            first_g_seen += 1
            g_delay.append(chosen["step"] - first_g["step"])
            first_g_stop += int(chosen["step"] == first_g["step"] and chosen["g"])
    n = len(trajectories)
    return {
        "signal": signal,
        "tau": tau,
        "k": k,
        "eps": eps,
        "n": n,
        "acc": correct / n,
        "acc_n": correct,
        "g_delivery": delivered_g / n,
        "early_rate": early / n,
        "false_stops": false_stop,
        "mean_step": float(np.mean(steps)),
        "mean_tokens_no_judge": float(np.mean(toks)),
        "mean_delay_from_first_g": float(np.mean(g_delay)) if g_delay else float("nan"),
        "first_g_coverage": first_g_seen / n,
        "first_g_recall": first_g_stop / first_g_seen if first_g_seen else float("nan"),
    }


def auc_rows(trajectories: dict[int, list[dict]], signal: str) -> float:
    labels, values = [], []
    for items in trajectories.values():
        for item in items:
            value = item["scores"].get(signal, float("nan"))
            if math.isfinite(value):
                labels.append(int(item["g"]))
                values.append(value)
    if not labels or min(labels) == max(labels):
        return float("nan")
    return float(roc_auc_score(labels, values))


def threshold_grid(trajectories: dict[int, list[dict]], signal: str) -> list[float]:
    values = [
        item["scores"].get(signal, float("nan"))
        for items in trajectories.values()
        for item in items
    ]
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if not len(finite):
        return []
    return sorted(set(float(x) for x in np.quantile(finite, [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.94, 0.96, 0.98, 0.99])))


def markdown(rows: list[dict], title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| signal | τ | k | Acc | G-delivery | first-G recall | early | false stop | mean step | tok(no judge) | delay from first G |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['signal']}` | {r['tau']:.4g} | {r['k']} | {r['acc']:.3f} ({r['acc_n']}/{r['n']}) | "
            f"{r['g_delivery']:.3f} | {r['first_g_recall']:.1%} | {r['early_rate']:.1%} | {r['false_stops']} | "
            f"{r['mean_step']:.1f} | {r['mean_tokens_no_judge']:.0f} | {r['mean_delay_from_first_g']:.1f} |"
        )
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=POLICY)
    args = parser.parse_args()
    trajectories, _, _ = build_trajectories()
    baseline = evaluate(trajectories, "B_final", 0.98, 2, 0.03)
    lines = [
        "# Frozen Qwen3-4B confidence variants",
        "",
        "Full MATH-500, no cal/held split.",
        "Stop protocol: PUMA-style `mss=10`, same answer for k consecutive dense trials, and score-drop ε.",
        "Token counts exclude sidecar inference; they are used only to rank fixed-policy solver savings.",
        "",
    ]

    records = []
    signals = sorted(
        {
            name
            for items in trajectories.values()
            for item in items
            for name in item["scores"]
        }
    )
    aucs = {name: auc_rows(trajectories, name) for name in signals}
    for signal in signals:
        for tau in threshold_grid(trajectories, signal):
            for k in (1, 2):
                records.append(evaluate(trajectories, signal, tau, k, 0.03))
    feasible = [
        row
        for row in records
        if (
            row["false_stops"] <= baseline["false_stops"]
            and row["acc"] >= baseline["acc"]
            and row["first_g_recall"] >= baseline["first_g_recall"]
        )
    ]
    if not feasible:
        raise RuntimeError("no variant meets the baseline safety constraint")
    selected = min(
        feasible,
        key=lambda row: (
            row["mean_tokens_no_judge"],
            row["mean_step"],
            -row["g_delivery"],
            -row["acc"],
        ),
    )
    payload = {
        "selection_split": "all",
        "baseline": baseline,
        "selected": selected,
        "signal_auc": aucs,
        "selection_rule": "false_stops <= baseline, acc >= baseline, and first-G recall >= baseline; minimize no-judge tokens",
    }
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    args.policy.write_text(json.dumps(payload, indent=2) + "\n")
    lines += [
        "## Full-set operating points",
        "",
        f"Baseline is `B_final τ=.98 k=2 ε=.03`; selected policy is `{selected['signal']}`.",
        "",
        "| signal | AUROC(G) |",
        "|---|---:|",
        *[f"| `{name}` | {value:.3f} |" for name, value in sorted(aucs.items())],
        "",
    ]
    lines += markdown([baseline, selected], "Baseline and selected variant")
    print(f"wrote policy {args.policy}")
    print(json.dumps({"n": baseline["n"], "baseline": baseline, "selected": selected}, indent=2))

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
