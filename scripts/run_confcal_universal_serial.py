#!/usr/bin/env python3
"""Run Universal-Qwen3 Base/Instruct extraction serially by dataset."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
PYTHON = VLLM_ROOT / ".venv/bin/python"
SCRIPT = AE / "scripts/score_confcal_universal.py"
LOG_DIR = AE / "results/_logs"
QUEUE = ("math-500", "olympiadbench", "gpqa-diamond")
# 0/2/3/4 is the historical default, but 2/3/5 are often occupied by
# other jobs. Prefer currently empty cards unless the caller overrides.
DEFAULT_GPUS = "0,1,4,6"


def libraries() -> str:
    extra = ":".join(sorted(str(path) for path in (VLLM_ROOT / ".venv/lib").glob("**/nvidia/*/lib") if path.is_dir()))
    return f"{extra}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--from", dest="from_dataset", choices=QUEUE)
    parser.add_argument("--only", choices=QUEUE, help="Run exactly one dataset.")
    parser.add_argument("--mode", choices=("base", "instruct"))
    args = parser.parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if args.only and args.from_dataset:
        raise ValueError("--only and --from are mutually exclusive")
    queue = (args.only,) if args.only else (QUEUE[QUEUE.index(args.from_dataset):] if args.from_dataset else QUEUE)
    modes = (args.mode,) if args.mode else ("base", "instruct")
    failed: list[str] = []
    for dataset in queue:
        for mode in modes:
            log_dir = LOG_DIR / "confcal_universal" / f"{dataset}_s42_{mode}"
            log_dir.mkdir(parents=True, exist_ok=True)
            procs = []
            for shard_id, gpu in enumerate(gpus):
                command = [str(PYTHON), str(SCRIPT), "--dataset", dataset, "--mode", mode,
                           "--shard-id", str(shard_id), "--num-shards", str(len(gpus))]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "VLLM_LENS_DISABLE": "1", "LD_LIBRARY_PATH": libraries()}
                handle = (log_dir / f"shard{shard_id}.log").open("a")
                print(f"launch {dataset} {mode} shard={shard_id}/{len(gpus)} gpu={gpu}", flush=True)
                procs.append(subprocess.Popen(command, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT))
                handle.close()
            codes = [proc.wait() for proc in procs]
            if any(codes):
                print(f"WARN {dataset} {mode} failed: {codes}; logs={log_dir}", flush=True)
                failed.append(f"{dataset}/{mode}:{codes}")
                continue
            print(f"complete {dataset} {mode}", flush=True)
    if failed:
        raise RuntimeError("universal serial had failures: " + "; ".join(failed))


if __name__ == "__main__":
    main()
