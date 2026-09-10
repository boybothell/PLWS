#!/usr/bin/env python3
"""Serially extract frozen-Qwen3-4B variant scores on all r1_7b dense sets.

Uses four GPUs in parallel shards, one dataset/seed at a time.  Already-scored
steps are skipped.  If another variant extractor is running, this waits first
so MATH-500 is not launched twice.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
PYTHON = AE / ".venv/bin/python"
SCRIPT = AE / "scripts/score_confcal_4b_variants.py"
LOG_DIR = AE / "results/_logs"
GPUS = (0, 2, 3, 4)

# Finish MATH-500 first, then the other official 7B sets.  AIME keeps every
# sampled seed because official PUMA folders are single-run but dense traces exist.
QUEUE: list[tuple[str, int]] = [
    ("math-500", 42),
    ("olympiadbench", 42),
    ("gpqa-diamond", 42),
    ("aime24", 42),
    ("aime24", 0),
    ("aime24", 1),
    ("aime24", 123),
    ("aime25", 42),
    ("aime25", 0),
    ("aime25", 1),
    ("aime25", 123),
]


def running_extractors() -> list[int]:
    out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    pids: list[int] = []
    for line in out.splitlines():
        if "score_confcal_4b_variants.py" in line and "grep" not in line:
            pids.append(int(line.split(None, 1)[0]))
    return pids


def wait_for_extractors(label: str) -> None:
    while True:
        pids = running_extractors()
        if not pids:
            return
        print(f"[wait] {label}: still running {pids}", flush=True)
        time.sleep(30)


def run_dataset(dataset: str, seed: int, gpus: tuple[int, ...]) -> None:
    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    n = len(gpus)
    procs: list[subprocess.Popen] = []
    print(f"[start] {dataset} seed={seed} shards={n} gpus={gpus}", flush=True)
    for shard_id, gpu in enumerate(gpus):
        log = log_dir / f"confcal_variant_{dataset}_s{seed}_sh{shard_id}.log"
        cmd = [
            str(PYTHON),
            str(SCRIPT),
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(n),
        ]
        handle = log.open("a")
        handle.write(f"\n# launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu}\n")
        handle.flush()
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=str(AE),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        )
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"{dataset} seed={seed} shard exits {codes}")
    print(f"[done] {dataset} seed={seed} exits {codes}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,2,3,4")
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument(
        "--from",
        dest="from_dataset",
        default="",
        help="start queue at this dataset (inclusive), e.g. olympiadbench",
    )
    args = parser.parse_args()
    gpus = tuple(int(x) for x in args.gpus.split(",") if x.strip())
    if not gpus:
        raise ValueError("no GPUs")
    queue = list(QUEUE)
    if args.from_dataset:
        idx = next(
            (i for i, (ds, _) in enumerate(queue) if ds == args.from_dataset),
            None,
        )
        if idx is None:
            raise ValueError(f"unknown --from dataset {args.from_dataset}")
        queue = queue[idx:]
    if not args.skip_wait:
        wait_for_extractors("existing extractors")
    for dataset, seed in queue:
        run_dataset(dataset, seed, gpus)
    print("[all done]", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr, flush=True)
        raise
