#!/usr/bin/env python3
"""第一扇已是 High/Mix 的题，切点与旧窗后压相同，直接抄分数。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

OLD = AE / "results/leftover_suppress_toend"


def load_old(tag: str, seed: int, kind: str) -> dict[str, dict]:
    suf = "" if kind == "low" else f"_{kind}"
    folder = OLD / f"{tag}_s{seed}_suppress{suf}"
    out: dict[str, dict] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") not in ("ok", "too_long"):
                continue
            uid = rec.get("uid")
            if uid:
                out[str(uid)] = rec
    return out


def seed_one(tag: str, seed: int) -> tuple[int, int, int]:
    jobs_path = AE / f"results/first_hm_gate/jobs/{tag}_s{seed}.jsonl"
    out_dir = AE / f"results/first_hm_gate/{tag}_s{seed}_suppress_hm"
    out = out_dir / "scores_shard99.jsonl"
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
    by_kind = {kind: load_old(tag, seed, kind) for kind in ("high", "mix")}
    reused = delayed = miss = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for job in jobs:
            uid = job["uid"]
            if job.get("delayed"):
                delayed += 1
                continue
            rec = by_kind.get(job["kind"], {}).get(uid)
            if rec is None:
                miss += 1
                continue
            if int(rec.get("left_step") or 0) != int(job["left_step"]):
                miss += 1
                continue
            rec = dict(rec)
            rec["gate"] = "first_hm"
            rec["reused"] = True
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            reused += 1
    print(f"s{seed} reuse={reused} delayed={delayed} miss={miss} -> {out}", flush=True)
    return reused, delayed, miss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default="42,0,1,123")
    args = parser.parse_args()
    seeds = [int(args.seed)] if args.seed is not None else [int(x) for x in args.seeds.split(",") if x.strip()]
    bad = 0
    for seed in seeds:
        jobs_path = AE / f"results/first_hm_gate/jobs/{args.model_tag}_s{seed}.jsonl"
        if not jobs_path.is_file():
            print(f"s{seed} skip no jobs {jobs_path}", flush=True)
            continue
        reused, delayed, miss = seed_one(args.model_tag, seed)
        bad += miss
    if bad:
        print(f"reuse miss={bad} will generate", flush=True)


if __name__ == "__main__":
    main()
