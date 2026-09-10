#!/usr/bin/env python3
"""Export leftover low-conf windows for a different trial suffix."""
from __future__ import annotations

import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_cand_collapse as cc
import report_leftover_stability as st

OUT = AE / "results/commit_probe"
JOBS = (("r1_7b", "math-500", 42), ("r1_7b", "gpqa-diamond", 42))


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    OUT.mkdir(parents=True, exist_ok=True)
    for model, dataset, seed in JOBS:
        pack = st.load_cell(model, dataset, seed, {})
        if pack is None:
            raise SystemExit(f"missing {model} {dataset}")
        jobs = []
        seen: set[tuple[int, int]] = set()
        for q in pack["questions"]:
            for w in q["windows"]:
                key = (int(q["qi"]), int(w["step"]))
                if key in seen:
                    continue
                seen.add(key)
                row = q["rows"][int(w["end"])]
                jobs.append(
                    {
                        "question_idx": int(q["qi"]),
                        "decision_step": int(w["step"]),
                        "question": row.get("question") or "",
                        "reasoning_prefix": row.get("reasoning_prefix") or "",
                        "old_answer": w["ans"],
                        "old_conf": rg.finite(row.get("confidence")),
                        "pos": bool(w["pos"]),
                        "gt": q["gt"],
                        "original": q["original"],
                        "orig_ok": q["orig_ok"],
                        "orig_tok": q["orig_tok"],
                        "host_ok": q["host_ok"],
                        "host_tok": q["host_tok"],
                    }
                )
        path = OUT / f"{model}_{dataset}_s{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in jobs))
        n_pos = sum(1 for r in jobs if r["pos"])
        print(f"{model} {dataset} leftover={len(jobs)} pos={n_pos} -> {path}", flush=True)


if __name__ == "__main__":
    main()
