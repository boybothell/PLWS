#!/usr/bin/env python3
"""Offline Acc/Tok replay: geo first-stop vs 4B forecast veto.

Policies after a veto (forecast_p_change >= t at the first geo candidate):
  next_geo   next geo confirmation, else last trial
  plus8      first geo confirmation at least 8 probes later, else last trial
  change     first later trial whose answer differs, else last trial
  full       last trial (Full-CoT prefix end)
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

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from pilot_active_stop import first_geo_candidate, finite  # noqa: E402
from report_confcal_4b_acctok import DATASETS, official_summary, same  # noqa: E402

GEO = dict(threshold=0.98, k=2, epsilon=0.03, mss=10)
THRESHOLDS = (0.10, 0.30, 0.50, 0.70, 0.90, 0.95)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset_paths(dataset: str) -> tuple[Path, Path]:
    for name, _seed, stat, dense in DATASETS:
        if name == dataset:
            return stat, dense
    raise KeyError(dataset)


def all_geo_indices(rows: list[dict[str, Any]]) -> list[int]:
    found: list[int] = []
    offset = 0
    while offset < len(rows):
        rel = first_geo_candidate(rows[offset:], **GEO)
        if rel is None:
            break
        abs_i = offset + rel
        if found and abs_i <= found[-1]:
            offset += 1
            continue
        found.append(abs_i)
        offset = abs_i + 1
    return found


def build_items(dataset: str) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    stat_path, dense_root = dataset_paths(dataset)
    official = official_summary(stat_path)
    meta = {
        int(row["question_idx"]): row
        for row in json.loads((dense_root / "per_sample.json").read_text())
    }
    trials = json.loads((dense_root / "dense_puma" / "trial_answers.json").read_text())
    by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        qi = int(trial["question_idx"])
        info = official["by_q"].get(qi) or {}
        sample = meta.get(qi) or {}
        ans = str(trial.get("final_answer") or "")
        a_final = info.get("a_final") or sample.get("A_final")
        gt = info.get("gt") or sample.get("ground_truth")
        orig_ok = bool(info.get("original_correct"))
        by_q[qi].append(
            {
                "step": int(trial["stopped_len"]),
                "answer": ans,
                "gt": bool(same(ans, gt) or (same(ans, a_final) and orig_ok)),
                "geo": finite(trial.get("confidence")),
                "reasoning_tokens": int(trial.get("count_reasoning_tokens") or 0),
                "answer_tokens": int(trial.get("count_answer_tokens") or 0),
                "raw": trial,
            }
        )
    for items in by_q.values():
        items.sort(key=lambda item: item["step"])
    return official, dict(by_q)


def tokens_to(items: list[dict[str, Any]], index: int) -> int:
    chosen = items[index]
    return chosen["reasoning_tokens"] + sum(
        item["answer_tokens"] for item in items if item["step"] <= chosen["step"]
    )


def load_forecasts(paths: list[Path]) -> dict[tuple[int, int, str], float]:
    out: dict[tuple[int, int, str], float] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            value = finite(row.get("forecast_p_change"))
            if not math.isfinite(value):
                continue
            out[(int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))] = value
    return out


def choose_after_veto(items: list[dict[str, Any]], first: int, geos: list[int], policy: str) -> int:
    if policy == "next_geo":
        later = [idx for idx in geos if idx > first]
        return later[0] if later else len(items) - 1
    if policy == "plus8":
        later = [idx for idx in geos if idx >= first + 8]
        return later[0] if later else len(items) - 1
    if policy == "change":
        current = items[first]["answer"]
        for idx in range(first + 1, len(items)):
            if items[idx]["answer"] != current:
                return idx
        return len(items) - 1
    if policy == "full":
        return len(items) - 1
    raise ValueError(policy)


def evaluate(
    by_q: dict[int, list[dict[str, Any]]],
    forecasts: dict[tuple[int, int, str], float],
    *,
    threshold: float | None,
    policy: str,
) -> dict[str, Any]:
    n = len(by_q)
    acc = tok = veto = veto_wrong = veto_right = missing = no_geo = 0
    for qi, items in by_q.items():
        geos = all_geo_indices([item["raw"] for item in items])
        if not geos:
            index = len(items) - 1
            no_geo += 1
        else:
            first = geos[0]
            key = (qi, items[first]["step"], items[first]["answer"])
            score = forecasts.get(key, float("nan"))
            if threshold is None:
                index = first
            elif not math.isfinite(score):
                index = first
                missing += 1
            elif score >= threshold:
                index = choose_after_veto(items, first, geos, policy)
                veto += 1
                if items[first]["gt"]:
                    veto_right += 1
                else:
                    veto_wrong += 1
            else:
                index = first
        acc += int(items[index]["gt"])
        tok += tokens_to(items, index)
    return {
        "n": n,
        "acc": acc / n,
        "acc_n": acc,
        "tok": tok / n,
        "veto": veto,
        "veto_wrong": veto_wrong,
        "veto_right": veto_right,
        "missing": missing,
        "no_geo": no_geo,
    }


def fmt(row: dict[str, Any], extra: str = "") -> str:
    return (
        f"| {extra} | {row['acc']:.1%} ({row['acc_n']}/{row['n']}) | "
        f"{row['tok']:.0f} | veto {row['veto']} "
        f"(wrong {row['veto_wrong']}, right {row['veto_right']}) "
        f"miss {row['missing']} no-geo {row['no_geo']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--forecast", action="append", type=Path, required=True)
    args = parser.parse_args()
    forecasts = load_forecasts(args.forecast)
    print(f"loaded forecasts={len(forecasts)}", flush=True)
    lines = [
        "# Forecast veto Acc/Tok replay",
        "",
        "Gate: first geo stop τ=0.98, k=2, ε=0.03, mss=10. Veto if 4B P(Revise) ≥ t.",
        "Tok = reasoning_at_stop + trial answers ≤ stop. Acc = official GT rule.",
        "",
    ]
    for dataset in args.dataset:
        official, by_q = build_items(dataset)
        geo = evaluate(by_q, forecasts, threshold=None, policy="full")
        oracle_scores: dict[tuple[int, int, str], float] = {}
        for qi, items in by_q.items():
            geos = all_geo_indices([item["raw"] for item in items])
            if not geos:
                continue
            first = geos[0]
            oracle_scores[(qi, items[first]["step"], items[first]["answer"])] = (
                0.0 if items[first]["gt"] else 1.0
            )
        oracle = evaluate(by_q, oracle_scores, threshold=0.5, policy="full")
        lines += [
            f"## {dataset} (n={official['n']})",
            "",
            "| method | Acc | Tok | notes |",
            "|---|---:|---:|---|",
            f"| Full-CoT | {official['full_acc']:.1%} ({official['full_acc_n']}/{official['n']}) | {official['full_tok']:.0f} | original tokens |",
            f"| Official PUMA | {official['puma_acc']:.1%} ({official['puma_acc_n']}/{official['n']}) | {official['puma_tok']:.0f} | regen+RD |",
            fmt(geo, "geo"),
            fmt(oracle, "oracle veto→full"),
        ]
        for policy in ("next_geo", "plus8", "change", "full"):
            for threshold in THRESHOLDS:
                row = evaluate(by_q, forecasts, threshold=threshold, policy=policy)
                d_acc = 100 * (row["acc"] - geo["acc"])
                d_tok = row["tok"] - geo["tok"]
                lines.append(
                    f"| p≥{threshold:.2f} →{policy} | {row['acc']:.1%} ({row['acc_n']}/{row['n']}) | "
                    f"{row['tok']:.0f} | Δacc {d_acc:+.2f}pp Δtok {d_tok:+.0f}; "
                    f"veto {row['veto']} (wrong {row['veto_wrong']}, right {row['veto_right']}) |"
                )
            lines.append("")
        print(f"done {dataset}", flush=True)
    text = "\n".join(lines) + "\n"
    out = AE / "tables/probe_forecast_veto_acctok.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
