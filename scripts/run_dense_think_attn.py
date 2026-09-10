#!/usr/bin/env python3
"""Launch 4-way HF eager shards for dense </think> attention probes."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
PYTHON = AE / ".venv/bin/python"
SCRIPT = AE / "scripts/pilot_think_attn.py"
DEFAULT_GPUS = (2, 3, 4, 5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpus", default=",".join(map(str, DEFAULT_GPUS)))
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    candidates = AE / "results/confcal_judge/v2/dense_candidates" / f"{args.dataset}.jsonl"
    out_dir = AE / "results/confcal_judge/v2/dense_think_attn" / args.dataset
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard_id, gpu in enumerate(gpus):
        out = out_dir / f"scores_shard{shard_id}.jsonl"
        log = (log_dir / f"shard{shard_id}.log").open("a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        command = [
            str(PYTHON),
            str(SCRIPT),
            "score",
            "--candidates",
            str(candidates),
            "--out",
            str(out),
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(len(gpus)),
            "--device",
            "cuda:0",
        ]
        print(f"launch think-attn {args.dataset} shard={shard_id} gpu={gpu}", flush=True)
        processes.append(subprocess.Popen(command, cwd=AE, env=env, stdout=log, stderr=subprocess.STDOUT))
        log.close()
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"think-attn failed {args.dataset}: {codes}")
    print(f"think-attn complete {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
