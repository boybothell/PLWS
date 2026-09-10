#!/usr/bin/env python3
"""Launch 4-way vLLM shards for dense 7B solver probes."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
PYTHON = VLLM_ROOT / ".venv/bin/python"
SCRIPT = AE / "scripts/score_dense_solver_probes.py"
DEFAULT_GPUS = (2, 3, 4, 5)


def nvidia_lib_path() -> str:
    extra = ":".join(
        sorted(str(path) for path in (VLLM_ROOT / ".venv" / "lib").glob("**/nvidia/*/lib") if path.is_dir())
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpus", default=",".join(map(str, DEFAULT_GPUS)))
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    log_dir = AE / "results/confcal_judge/v2/dense_solver_probes" / args.dataset / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard_id, gpu in enumerate(gpus):
        out = AE / "results/confcal_judge/v2/dense_solver_probes" / args.dataset / f"scores_shard{shard_id}.jsonl"
        log = (log_dir / f"shard{shard_id}.log").open("a")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_LENS_DISABLE": "1",
            "LD_LIBRARY_PATH": nvidia_lib_path(),
        }
        command = [
            str(PYTHON),
            str(SCRIPT),
            "--dataset",
            args.dataset,
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(len(gpus)),
            "--out",
            str(out),
        ]
        print(f"launch {args.dataset} shard={shard_id} gpu={gpu}", flush=True)
        processes.append(subprocess.Popen(command, cwd=AE, env=env, stdout=log, stderr=subprocess.STDOUT))
        log.close()
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"failed {args.dataset}: {codes}")
    print(f"complete {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
