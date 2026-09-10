#!/usr/bin/env python3
"""第一扇剩窗若早于密探k4 停点，导出重写用 final_candidates。"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_first_lock_room as room
import report_k4_hyps as hy
import report_k4_second_lock as sl

OUT = AE / "results/leftover_regen"
MODELS = ("r1_7b", "nemotron_8b")
DS = (
    "math-500",
    "olympiadbench",
    "gpqa-diamond",
    "aime24",
    "aime25",
)


def cell_out(model: str, dataset: str, seed: int) -> Path:
    return OUT / model / dataset / f"s{seed}"


def trial_tokens_upto(trials: list[dict[str, Any]], step: int) -> tuple[int, int]:
    used = [
        x
        for x in trials
        if dd.should_probe(int(x["stopped_len"]))
        and int(x["stopped_len"]) <= step
        and str(x.get("final_answer") or "")
        and dd.finite(x.get("confidence")) == dd.finite(x.get("confidence"))
    ]
    return len(used), sum(dd.trial_answer_tokens(x) for x in used)


def export_cell(model: str, dataset: str, seed: int) -> dict[str, Any]:
    trials_path = room.dense_trial_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    qpath = dd.puma_dir(model, dataset, seed) / "filtered_steps.json"
    regen_path = dd.regen_stat_path(model, dataset, seed)
    if not trials_path.is_file() or not puma_path.is_file() or not qpath.is_file() or not regen_path.is_file():
        return {"model": model, "dataset": dataset, "seed": seed, "missing": True}
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    regen = {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
    questions = dd.load_json(qpath)
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    cands = []
    reuse = []
    n_mismatch = 0
    for qi, info in sorted(official.items()):
        host = regen.get(qi)
        if not host:
            continue
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        wins = sl.same_windows(rows)
        left = next((w for w in wins if hy.leftover(w)), None)
        host_step = int(host.get("stopped_len") or 10**9)
        question = str(info.get("question") or trials[0].get("question") or "")
        if qi - 1 >= len(questions) or questions[qi - 1].get("question") != question:
            n_mismatch += 1
            continue
        if left is None or int(left["step"]) >= host_step:
            reuse.append(qi)
            continue
        n_gen, tok = trial_tokens_upto(trials, int(left["step"]))
        conf = float(left["c"])
        cands.append(
            {
                "question_idx": qi,
                "stopped_len": int(left["step"]),
                "original_len_reasoning_steps": int(
                    host.get("original_len_reasoning_steps") or max(int(x["stopped_len"]) for x in trials)
                ),
                "question": question,
                "skipped": False,
                "skip_reason": None,
                "similarity": None,
                "success": True,
                "generated_trial_answers": n_gen,
                "tokens_trial_answers": tok,
                "tokens_trial_answers_online": tok,
                "stop_reason": f"leftover_{left['kind']}",
                "stop_confidence": None if not math.isfinite(conf) else conf,
                "stop_threshold": None,
                "consecutive_confidences": None,
                "confidence_trajectory": None,
                "step_similarities": None,
                "final_answer": left["ans"],
                "host_stopped_len": host_step,
                "leftover_kind": left["kind"],
            }
        )
    if n_mismatch:
        raise RuntimeError(f"{model} {dataset} s{seed}: {n_mismatch} question mismatches")
    out_dir = cell_out(model, dataset, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_candidates.json").write_text(
        json.dumps(cands, indent=2, ensure_ascii=False) + "\n"
    )
    meta = {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "n_fire": len(cands),
        "n_reuse_k4": len(reuse),
        "n_mix": sum(1 for c in cands if c["leftover_kind"] == "mix"),
        "n_low": sum(1 for c in cands if c["leftover_kind"] == "low"),
        "reuse_k4": reuse,
        "k4_prefixed": str(dd.cell_out_dir(model, dataset, seed) / "prefixed_answers.json"),
        "k4_stats": str(regen_path),
        "questions_file": str(qpath),
        "answers_file": str(dd.puma_dir(model, dataset, seed) / "answers.json"),
    }
    (out_dir / "candidates_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"{model:12} {dataset:14} s{seed:<4} fire={len(cands):3} "
        f"mix={meta['n_mix']:3} low={meta['n_low']:3} reuse_k4={len(reuse)}",
        flush=True,
    )
    return meta


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    n = 0
    for model in MODELS:
        for dataset in DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                rec = export_cell(model, dataset, seed)
                if rec.get("missing"):
                    print(f"missing {model} {dataset} s{seed}", flush=True)
                    continue
                n += rec["n_fire"]
    print(f"total fire={n} → {OUT}", flush=True)


if __name__ == "__main__":
    main()
