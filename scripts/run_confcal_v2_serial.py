#!/usr/bin/env python3
"""Serial v2 run: H-A KeyToken, then H-B/H-C Universal, then unified analysis.

H-D is already complete and is not re-extracted.  Four GPUs are used inside
each dataset/phase; datasets and methods stay serial.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
VLLM_PYTHON = VLLM_ROOT / ".venv/bin/python"
PYTHON = sys.executable
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond")
SMALL = Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-1.5B")


def libraries() -> str:
    extra = ":".join(sorted(str(path) for path in (VLLM_ROOT / ".venv/lib").glob("**/nvidia/*/lib") if path.is_dir()))
    return f"{extra}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=AE, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,4,6,7")
    parser.add_argument("--small-model", type=Path, default=SMALL)
    parser.add_argument("--skip-keytoken", action="store_true")
    parser.add_argument("--skip-universal", action="store_true")
    args = parser.parse_args()
    env = {**os.environ, "VLLM_LENS_DISABLE": "1", "LD_LIBRARY_PATH": libraries()}
    if not args.skip_keytoken:
        if not (args.small_model / "config.json").exists() or not any(args.small_model.glob("*.safetensors")):
            raise FileNotFoundError(f"H-A requires a complete family-small checkpoint at {args.small_model}")
        run(
            [VLLM_PYTHON, str(AE / "scripts/run_confcal_keytoken_serial.py"), "--gpus", args.gpus, "--small-model", str(args.small_model)],
            env,
        )
    if not args.skip_universal:
        run([VLLM_PYTHON, str(AE / "scripts/run_confcal_universal_serial.py"), "--gpus", args.gpus], env)
    run([PYTHON, str(AE / "scripts/analyze_confcal_v2.py")])


if __name__ == "__main__":
    main()
