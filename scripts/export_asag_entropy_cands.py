#!/usr/bin/env python3
"""Pick PUMA-step checkpoints to test ASAG attention-entropy drop."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg

OUT = AE / "results/asag_entropy"
FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)
JOBS = (
    ("r1_7b", "math-500", 42),
    ("r1_7b", "gpqa-diamond", 42),
)


def nearest(rows: list[dict[str, Any]], target: int) -> dict[str, Any]:
    return min(rows, key=lambda r: (abs(int(r["stopped_len"]) - target), int(r["stopped_len"])))


def first_hit(rows: list[dict[str, Any]], target: Any) -> dict[str, Any] | None:
    for row in rows:
        if rg.same(row.get("final_answer"), target):
            return row
    return None


def ladder(rows: list[dict[str, Any]], end: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    last = int(end["stopped_len"])
    out = []
    seen: set[int] = set()
    for frac in FRACS:
        pick = nearest(rows, max(1, int(round(frac * last))))
        step = int(pick["stopped_len"])
        if step in seen:
            continue
        seen.add(step)
        tag = f"{kind}_{frac:.1f}"
        if frac == 1.0:
            tag = f"{kind}_hf"
        rec = dict(pick)
        rec["tag"] = tag
        rec["kind"] = kind
        rec["frac"] = frac
        rec["anchor_step"] = last
        out.append(rec)
    return out


def export_one(model: str, dataset: str, seed: int) -> Path:
    official = {int(r["question_idx"]): r for r in dd.load_json(dd.puma_stat_path(model, dataset, seed))}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(dd.trial_path(model, dataset, seed)):
        by[int(row["question_idx"])].append(row)
    jobs = []
    n_gold = n_af = n_wrong = 0
    for qi, info in sorted(official.items()):
        trials = rg.usable_rows(by.get(qi) or [])
        if not trials:
            continue
        gt = info.get("ground_truth")
        original = info.get("original_answer")
        orig_ok = bool(info.get("original_correct"))
        gold = first_hit(trials, gt)
        af = first_hit(trials, original)
        picked: list[dict[str, Any]] = []
        if orig_ok and gold is not None:
            picked.extend(ladder(trials, gold, "gold"))
            n_gold += 1
        if orig_ok and af is not None:
            picked.extend(ladder(trials, af, "afinal"))
            n_af += 1
        if (not orig_ok) and trials:
            picked.extend(ladder(trials, trials[-1], "wrong"))
            n_wrong += 1
        for rec in picked:
            jobs.append(
                {
                    "question_idx": qi,
                    "decision_step": int(rec["stopped_len"]),
                    "question": info.get("question") or rec.get("question") or "",
                    "reasoning_prefix": rec.get("reasoning_prefix") or "",
                    "answer": rec.get("final_answer") or "",
                    "confidence": rec.get("confidence"),
                    "tag": rec["tag"],
                    "kind": rec["kind"],
                    "frac": rec["frac"],
                    "anchor_step": rec["anchor_step"],
                    "orig_ok": orig_ok,
                    "ground_truth": gt,
                    "original_answer": original,
                }
            )
    out = OUT / f"{model}_{dataset}_s{seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in jobs))
    print(f"{model} {dataset} jobs={len(jobs)} gold_q={n_gold} af_q={n_af} wrong_q={n_wrong} -> {out}", flush=True)
    return out


def main() -> None:
    for model, dataset, seed in JOBS:
        export_one(model, dataset, seed)


if __name__ == "__main__":
    main()
