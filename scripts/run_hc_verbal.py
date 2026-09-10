#!/usr/bin/env python3
"""Launch 4B verbalized-distribution scoring on free GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
PYTHON = VLLM_ROOT / ".venv/bin/python"
SCRIPT = AE / "scripts/score_hc_verbal.py"
LOG_DIR = AE / "results/_logs/hc_verbal"


def libraries() -> str:
    extra = ":".join(
        sorted(str(path) for path in (VLLM_ROOT / ".venv/lib").glob("**/nvidia/*/lib") if path.is_dir())
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for shard_id, gpu in enumerate(gpus):
        command = [
            str(PYTHON),
            str(SCRIPT),
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(len(gpus)),
        ]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu,
            "VLLM_LENS_DISABLE": "1",
            "LD_LIBRARY_PATH": libraries(),
        }
        handle = (LOG_DIR / f"shard{shard_id}.log").open("a")
        print(f"launch hc-verbal shard={shard_id}/{len(gpus)} gpu={gpu}", flush=True)
        procs.append(subprocess.Popen(command, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT))
        handle.close()
    codes = [proc.wait() for proc in procs]
    if any(codes):
        raise SystemExit(f"hc-verbal failed: {codes}; logs={LOG_DIR}")
    print("hc-verbal complete", flush=True)


if __name__ == "__main__":
    main()
