#!/usr/bin/env python3
"""Run Family-KeyToken H-A extraction: gate, solver, small, merge."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
PYTHON = VLLM_ROOT / ".venv/bin/python"
SCRIPT = AE / "scripts/score_confcal_keytoken.py"
LOG_DIR = AE / "results/_logs"
QUEUE = ("math-500", "olympiadbench", "gpqa-diamond")
DEFAULT_GPUS = "0,1,4,6"
SMALL = Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-1.5B")


def libraries() -> str:
    extra = ":".join(sorted(str(path) for path in (VLLM_ROOT / ".venv/lib").glob("**/nvidia/*/lib") if path.is_dir()))
    return f"{extra}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def launch(dataset: str, phase: str, gpus: tuple[str, ...], small: Path) -> None:
    log_dir = LOG_DIR / "confcal_keytoken" / f"{dataset}_s42_{phase}"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for shard_id, gpu in enumerate(gpus):
        command = [
            str(PYTHON), str(SCRIPT), "--phase", phase, "--dataset", dataset,
            "--small-model", str(small), "--shard-id", str(shard_id),
            "--num-shards", str(len(gpus)),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "VLLM_LENS_DISABLE": "1", "LD_LIBRARY_PATH": libraries()}
        handle = (log_dir / f"shard{shard_id}.log").open("a")
        print(f"launch keytoken {dataset} {phase} shard={shard_id}/{len(gpus)} gpu={gpu}", flush=True)
        procs.append(subprocess.Popen(command, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT))
        handle.close()
    failures = [proc.wait() for proc in procs]
    if any(failures):
        raise RuntimeError(f"{dataset} {phase} failed: {failures}; logs={log_dir}")
    print(f"complete keytoken {dataset} {phase}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--only", choices=QUEUE)
    parser.add_argument("--small-model", type=Path, default=SMALL)
    args = parser.parse_args()
    if not args.small_model.exists() or not (args.small_model / "config.json").exists():
        raise FileNotFoundError(f"family-small checkpoint missing: {args.small_model}")
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    queue = (args.only,) if args.only else QUEUE
    env = {**os.environ, "VLLM_LENS_DISABLE": "1", "LD_LIBRARY_PATH": libraries()}
    failed: list[str] = []
    for dataset in queue:
        try:
            subprocess.run(
                [str(PYTHON), str(SCRIPT), "--phase", "gate", "--dataset", dataset, "--small-model", str(args.small_model)],
                cwd=AE, env=env, check=True,
            )
            launch(dataset, "solver", gpus, args.small_model)
            launch(dataset, "small", gpus, args.small_model)
            launch(dataset, "merge", gpus, args.small_model)
        except Exception as exc:
            print(f"WARN keytoken {dataset} failed: {exc}", flush=True)
            failed.append(f"{dataset}:{exc}")
    if failed:
        raise RuntimeError("keytoken serial had failures: " + "; ".join(failed))


if __name__ == "__main__":
    main()
