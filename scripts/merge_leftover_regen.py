#!/usr/bin/env python3
"""把剩窗重写和密探k4 prefixed 合成完整一盘。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AE = Path(__file__).resolve().parents[1]


def load(path: Path) -> list:
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    out = AE / "results/leftover_regen" / args.model / args.dataset / f"s{args.seed}"
    meta = json.loads((out / "candidates_meta.json").read_text())
    k4 = load(Path(meta["k4_prefixed"]))
    fire = load(out / "prefixed_answers_fire.json")
    by = {int(r["question_idx"]): r for r in k4}
    n_rep = 0
    for row in fire:
        by[int(row["question_idx"])] = row
        n_rep += 1
    merged = [by[k] for k in sorted(by)]
    dest = out / "prefixed_answers.json"
    dest.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"merge {args.model} {args.dataset} s{args.seed} k4={len(k4)} fire={n_rep} out={len(merged)} → {dest}")


if __name__ == "__main__":
    main()
