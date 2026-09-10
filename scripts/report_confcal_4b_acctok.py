#!/usr/bin/env python3
"""Acc/Tok snapshot: Full-CoT, official PUMA, geo-conf and 4B trial-as-final."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/confcal_judge/qwen4b_variants"
_WS = re.compile(r"\s+")
_LATEX = re.compile(r"\\(left|right|cdot|times|dfrac|tfrac|frac|mathrm|text)")

DATASETS: list[tuple[str, int, Path, Path]] = [
    ("math-500", 42, AE / "results/math500_official/puma_ds7b/statistics.json", AE / "results/dense_G_r1_7b/math-500"),
    ("olympiadbench", 42, AE / "results/puma_offline_r1_7b/olympiadbench/statistics.json", AE / "results/dense_G_r1_7b/olympiadbench"),
    ("gpqa-diamond", 42, AE / "results/puma_offline_r1_7b/gpqa-diamond/statistics.json", AE / "results/dense_G_r1_7b/gpqa-diamond"),
    ("aime24", 42, AE / "results/puma_offline_r1_7b/aime24/statistics.json", AE / "results/dense_G_r1_7b/aime24/seed_42"),
    ("aime24", 0, AE / "results/puma_offline_r1_7b_s0/aime24/statistics.json", AE / "results/dense_G_r1_7b/aime24/seed_0"),
    ("aime24", 1, AE / "results/puma_offline_r1_7b_s1/aime24/statistics.json", AE / "results/dense_G_r1_7b/aime24/seed_1"),
    ("aime24", 123, AE / "results/puma_offline_r1_7b_s123/aime24/statistics.json", AE / "results/dense_G_r1_7b/aime24/seed_123"),
    ("aime25", 42, AE / "results/puma_offline_r1_7b/aime25/statistics.json", AE / "results/dense_G_r1_7b/aime25/seed_42"),
    ("aime25", 0, AE / "results/puma_offline_r1_7b_s0/aime25/statistics.json", AE / "results/dense_G_r1_7b/aime25/seed_0"),
    ("aime25", 1, AE / "results/puma_offline_r1_7b_s1/aime25/statistics.json", AE / "results/dense_G_r1_7b/aime25/seed_1"),
    ("aime25", 123, AE / "results/puma_offline_r1_7b_s123/aime25/statistics.json", AE / "results/dense_G_r1_7b/aime25/seed_123"),
]


def norm(text: Any) -> str:
    s = _WS.sub("", str(text or "").strip().lower())
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = _LATEX.sub("", s)
    return s.replace("\\", "")


def same(a: Any, b: Any) -> bool:
    x, y = norm(a), norm(b)
    return bool(x) and x == y


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def signal_values(row: dict[str, Any] | None, geo: float) -> dict[str, float]:
    out = {"geo_conf": geo}
    if not row:
        return out
    b = row.get("B_yesno") or {}
    eyn = row.get("EAGLE_yn") or {}
    three = row.get("B_3way") or {}
    ten = row.get("EAGLE_10bin") or {}
    correct = safe_float(three.get("Correct"))
    incorrect = safe_float(three.get("Incorrect"))
    out["B_final"] = safe_float(b.get("yes"))
    out["B_3way_correct"] = (
        correct / (correct + incorrect)
        if math.isfinite(correct) and math.isfinite(incorrect) and correct + incorrect > 0
        else float("nan")
    )
    out["EAGLE_yn"] = safe_float(eyn.get("yes"))
    out["EAGLE_10bin"] = safe_float(ten.get("eagle_expected"))
    return out


def load_scores(dataset: str, seed: int) -> dict[tuple[int, int, str], dict[str, Any]]:
    rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    dirs = [ROOT / f"{dataset}_s{seed}"]
    if dataset == "math-500":
        dirs.append(ROOT / f"{dataset}_s{seed}_cal")
    for path in dirs:
        if not path.is_dir():
            continue
        for shard in sorted(path.glob("scores*.jsonl")):
            with shard.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("status") != "ok":
                        continue
                    key = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
                    rows[key] = row
    return rows


def official_summary(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text())
    n = len(rows)
    orig = sum(int(r["original_correct"]) for r in rows)
    comp = sum(int(r["compressed_correct"]) for r in rows)
    early = sum(int(r.get("stop_reason") != "full_reasoning") for r in rows)
    return {
        "n": n,
        "full_acc": orig / n,
        "full_acc_n": orig,
        "full_tok": sum(int(r["original_tokens"]) for r in rows) / n,
        "puma_acc": comp / n,
        "puma_acc_n": comp,
        "puma_tok": sum(int(r["compressed_tokens"]) + int(r.get("tokens_trial_answers") or 0) for r in rows) / n,
        "puma_early": early / n,
        "by_q": {
            int(r["question_idx"]): {
                "gt": r.get("ground_truth"),
                "a_final": r.get("original_answer"),
                "original_correct": bool(r["original_correct"]),
            }
            for r in rows
        },
    }


def dense_trials_path(dense_root: Path) -> Path:
    return dense_root / "dense_puma" / "trial_answers.json"


def load_meta(dense_root: Path) -> dict[int, dict[str, Any]]:
    rows = json.loads((dense_root / "per_sample.json").read_text())
    return {int(r["question_idx"]): r for r in rows}


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


def evaluate(items_by_q: dict[int, list[dict]], signal: str) -> dict[str, Any]:
    correct = delivered_g = early = false_stop = 0
    toks, delays = [], []
    first_g_seen = 0
    scored_q = 0
    for items in items_by_q.values():
        if any(math.isfinite(x["scores"].get(signal, float("nan"))) for x in items):
            scored_q += 1
        chosen, reason = stop(items, signal, 0.98, 2, 0.03)
        correct += int(chosen["gt"])
        delivered_g += int(chosen["g"])
        early += int(reason == "confidence_consecutive")
        false_stop += int(reason == "confidence_consecutive" and not chosen["g"])
        toks.append(
            chosen["reasoning_tokens"]
            + sum(x["answer_tokens"] for x in items if x["step"] <= chosen["step"])
        )
        first_g = next((x for x in items if x["g"]), None)
        if first_g:
            first_g_seen += 1
            delays.append(chosen["step"] - first_g["step"])
    n = len(items_by_q)
    return {
        "n": n,
        "scored_q": scored_q,
        "acc": correct / n,
        "acc_n": correct,
        "g_delivery": delivered_g / n,
        "early": early / n,
        "false_stops": false_stop,
        "tok": sum(toks) / n,
        "delay": sum(delays) / len(delays) if delays else float("nan"),
        "first_g_cov": first_g_seen / n,
    }


def build_trajectories(dataset: str, seed: int, official: dict[str, Any], dense_root: Path) -> dict[int, list[dict]]:
    scores = load_scores(dataset, seed)
    meta = load_meta(dense_root)
    by_q: dict[int, list[dict]] = defaultdict(list)
    with dense_trials_path(dense_root).open() as handle:
        trials = json.load(handle)
    for trial in trials:
        qi = int(trial["question_idx"])
        info = official["by_q"].get(qi) or {}
        sample = meta.get(qi) or {}
        ans = str(trial.get("final_answer") or "")
        a_final = info.get("a_final") or sample.get("A_final")
        gt = info.get("gt") or sample.get("ground_truth")
        orig_ok = bool(info.get("original_correct"))
        g_step = sample.get("G")
        is_gt = same(ans, gt) or (same(ans, a_final) and orig_ok)
        is_g = is_gt or same(ans, a_final) or (g_step is not None and int(trial["stopped_len"]) == int(g_step))
        score = scores.get((qi, int(trial["stopped_len"]), ans))
        by_q[qi].append(
            {
                "step": int(trial["stopped_len"]),
                "answer": ans,
                "gt": is_gt,
                "g": is_g,
                "reasoning_tokens": int(trial.get("count_reasoning_tokens") or 0),
                "answer_tokens": int(trial.get("count_answer_tokens") or 0),
                "scores": signal_values(score, safe_float(trial.get("confidence"))),
            }
        )
    for items in by_q.values():
        items.sort(key=lambda x: x["step"])
        first_g_ans = next((x["answer"] for x in items if x["g"]), None)
        if first_g_ans is None:
            continue
        for item in items:
            if same(item["answer"], first_g_ans):
                item["g"] = True
    return dict(by_q)


def fmt_row(name: str, acc: float, acc_n: int, n: int, tok: float, extra: str = "") -> str:
    return f"| {name} | {acc:.1%} ({acc_n}/{n}) | {tok:.0f} | {extra} |"


def main() -> None:
    want = [x for x in sys.argv[1:]]
    lines = [
        "# Acc/Tok snapshot (trial-as-final vs official PUMA)",
        "",
        "Gate: τ=0.98, k=2, ε=0.03, mss=10. 4B/geo-conf Tok = reasoning_at_stop + trial answers ≤ stop.",
        "Official PUMA Tok = compressed + trial answers (regen + RD). Full-CoT Tok = original tokens.",
        "Acc uses official `original_correct` when the submitted trial matches A_final, else latex-normalized GT match.",
        "",
    ]
    for dataset, seed, stat_path, dense_root in DATASETS:
        key = f"{dataset}_s{seed}"
        if want and key not in want and dataset not in want:
            continue
        score_dir = ROOT / f"{dataset}_s{seed}"
        if not score_dir.is_dir():
            print(f"skip {key}: no 4B scores", file=sys.stderr)
            continue
        if not dense_trials_path(dense_root).exists():
            print(f"skip {key}: no dense trials", file=sys.stderr)
            continue
        print(f"eval {key}", file=sys.stderr, flush=True)
        official = official_summary(stat_path)
        traj = build_trajectories(dataset, seed, official, dense_root)
        n = official["n"]
        if len(traj) != n:
            print(f"  warn n_traj={len(traj)} n_official={n}", file=sys.stderr)
        lines += [
            f"## {dataset} seed={seed} (n={n})",
            "",
            "| method | Acc | Tok | notes |",
            "|---|---:|---:|---|",
            fmt_row("Full-CoT", official["full_acc"], official["full_acc_n"], n, official["full_tok"], "original tokens"),
            fmt_row(
                "Official PUMA",
                official["puma_acc"],
                official["puma_acc_n"],
                n,
                official["puma_tok"],
                f"early {official['puma_early']:.1%}; regen+RD",
            ),
        ]
        for signal in ("geo_conf", "B_final", "B_3way_correct", "EAGLE_yn", "EAGLE_10bin"):
            row = evaluate(traj, signal)
            extra = (
                f"early {row['early']:.1%}; false-stop {row['false_stops']}; "
                f"G-deliv {row['g_delivery']:.1%}; delay {row['delay']:.1f}; scored {row['scored_q']}/{row['n']}"
            )
            lines.append(fmt_row(signal, row["acc"], row["acc_n"], row["n"], row["tok"], extra))
        lines.append("")
        print(json.dumps({"dataset": dataset, "seed": seed, "n": n}, indent=None), file=sys.stderr)
    text = "\n".join(lines) + "\n"
    out = AE / "tables/probe_confcal_4b_acctok.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
