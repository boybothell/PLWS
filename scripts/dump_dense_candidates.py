#!/usr/bin/env python3
"""Write every dense trial as a candidate row for process-probe scripts."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

# Heavy analysis imports stay in main() so vLLM workers can import trial_path
# without sklearn.

DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
OUT_DIR = AE / "results/confcal_judge/v2/dense_candidates"


def trial_path(model_tag: str, dataset: str, seed: int | None) -> Path:
    root = AE / f"results/dense_G_{model_tag}/{dataset}"
    if seed is not None:
        return root / f"seed_{seed}" / "dense_puma" / "trial_answers.json"
    direct = root / "dense_puma" / "trial_answers.json"
    if direct.exists():
        return direct
    seeds = sorted(root.glob("seed_*/dense_puma/trial_answers.json"))
    if len(seeds) == 1:
        return seeds[0]
    raise FileNotFoundError(f"no trial_answers under {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--history", type=int, default=5)
    args = parser.parse_args()
    from analyze_confcal_v1 import _fast_eq, finite  # noqa: E402
    from analyze_dense_layer_curve import load_refs  # noqa: E402

    a_final, gold = load_refs(args.dataset, model_tag=args.model_tag, seed=args.seed)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    path = trial_path(args.model_tag, args.dataset, args.seed)
    for row in json.loads(path.read_text()):
        grouped[int(row["question_idx"])].append(row)
    if args.out:
        out = args.out
    elif args.model_tag == "r1_7b" and args.dataset in {"math-500", "olympiadbench", "gpqa-diamond"}:
        out = OUT_DIR / f"{args.dataset}.jsonl"
    else:
        name = f"{args.dataset}.jsonl" if args.seed is None else f"{args.dataset}_s{args.seed}.jsonl"
        out = OUT_DIR / args.model_tag / name
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as handle:
        for qi, seq in grouped.items():
            seq.sort(key=lambda row: int(row["stopped_len"]))
            answers = [str(row.get("final_answer") or "") for row in seq]
            flags = [_fast_eq(answer, a_final.get(qi)) or _fast_eq(answer, gold.get(qi)) for answer in answers]
            for index, row in enumerate(seq):
                neighbor = (index > 0 and flags[index - 1]) or (index + 1 < len(flags) and flags[index + 1])
                handle.write(
                    json.dumps(
                        {
                            "dataset": args.dataset,
                            "model_tag": args.model_tag,
                            "question_idx": qi,
                            "index": index,
                            "decision_step": int(row["stopped_len"]),
                            "answer": answers[index],
                            "geo_conf": finite(row.get("confidence")),
                            "is_g": int(flags[index]),
                            "is_g_window": int(flags[index] and neighbor),
                            "question": str(row.get("question") or ""),
                            "reasoning_prefix": str(row.get("reasoning_prefix") or ""),
                            "recent_answers": answers[max(0, index - args.history) : index + 1],
                        }
                    )
                    + "\n"
                )
                n += 1
    print(f"wrote {out} n={n}", flush=True)


if __name__ == "__main__":
    main()
