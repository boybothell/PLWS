#!/usr/bin/env python3
"""Run v1 H3/H4 extraction with vLLM, serially by dataset, four shards at a time."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
PYTHON = VLLM_ROOT / ".venv/bin/python"
SCRIPT = AE / "scripts/score_confcal_v1.py"
LOG_DIR = AE / "results/_logs"
DEFAULT_GPUS = (0, 2, 3, 4)
QUEUE = (("math-500", 42), ("olympiadbench", 42), ("gpqa-diamond", 42))


def nvidia_lib_path() -> str:
    extra = ":".join(
        sorted(str(path) for path in (VLLM_ROOT / ".venv" / "lib").glob("**/nvidia/*/lib") if path.is_dir())
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


def run_dataset(dataset: str, seed: int, gpus: tuple[int, ...]) -> None:
    log_dir = LOG_DIR / "confcal_v1" / f"{dataset}_s{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    for shard_id, gpu in enumerate(gpus):
        command = [
            str(PYTHON), str(SCRIPT), "--dataset", dataset, "--seed", str(seed),
            "--device", "cuda:0", "--shard-id", str(shard_id),
            "--num-shards", str(len(gpus)),
        ]
        handle = (log_dir / f"shard{shard_id}.log").open("a")
        print(f"launch {dataset} shard={shard_id}/{len(gpus)} physical_gpu={gpu} backend=vllm", flush=True)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_LENS_DISABLE": "1",
            "LD_LIBRARY_PATH": nvidia_lib_path(),
        }
        processes.append(
            subprocess.Popen(command, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT)
        )
        handle.close()
    failures = [process.wait() for process in processes]
    if any(failures):
        raise RuntimeError(f"{dataset} failed: shard exit codes={failures}; logs={log_dir}")
    print(f"complete {dataset}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=",".join(map(str, DEFAULT_GPUS)))
    parser.add_argument("--from", dest="from_dataset", choices=[dataset for dataset, _ in QUEUE])
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("at least one GPU is required")
    queue = QUEUE
    if args.from_dataset:
        start = next(index for index, (dataset, _) in enumerate(queue) if dataset == args.from_dataset)
        queue = queue[start:]
    for dataset, seed in queue:
        run_dataset(dataset, seed, gpus)


if __name__ == "__main__":
    main()
