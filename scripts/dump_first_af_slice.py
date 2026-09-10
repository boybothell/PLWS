#!/usr/bin/env python3
"""MATH first A_final visit vs earlier non-Af flashes."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq, finite, load_refs  # noqa: E402

OUT = AE / "results/confcal_judge/v2/first_af_slice/math-500.jsonl"


def main() -> None:
    a_final, _ = load_refs("math-500")
    grouped: dict[int, list] = defaultdict(list)
    path = AE / "results/dense_G_r1_7b/math-500/dense_puma/trial_answers.json"
    for row in json.loads(path.read_text()):
        grouped[int(row["question_idx"])].append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = 0
    with OUT.open("w") as handle:
        for qi, seq in grouped.items():
            seq.sort(key=lambda row: int(row["stopped_len"]))
            af = a_final.get(qi)
            flags = [_fast_eq(str(row.get("final_answer") or ""), af) for row in seq]
            if not any(flags):
                continue
            first = flags.index(True)
            left = any(not flags[index] for index in range(first + 1, len(flags)))
            for index, row in enumerate(seq[: first + 1]):
                if index < first and flags[index]:
                    continue
                label = int(index == first)
                if label:
                    n_pos += 1
                else:
                    n_neg += 1
                handle.write(
                    json.dumps(
                        {
                            "dataset": "math-500",
                            "question_idx": qi,
                            "index": index,
                            "decision_step": int(row["stopped_len"]),
                            "answer": str(row.get("final_answer") or ""),
                            "a_final": str(af or ""),
                            "is_first_af": label,
                            "will_leave": int(left),
                            "geo_conf": finite(row.get("confidence")),
                            "question": str(row.get("question") or ""),
                            "reasoning_prefix": str(row.get("reasoning_prefix") or ""),
                        }
                    )
                    + "\n"
                )
    print(f"wrote {OUT} pos={n_pos} neg={n_neg}", flush=True)


if __name__ == "__main__":
    main()
