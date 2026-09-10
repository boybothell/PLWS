#!/usr/bin/env python3
"""CPU-only matrix prep: export missing window jobs and import AIME core scores."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plws.matrix import (  # noqa: E402
    ALL_MODELS,
    DATASETS,
    FIVE_MODELS,
    SEEDS,
    ensure_firstwin_jobs,
    import_aime4s_core,
    needs_cpu_export,
)
from plws.paths import PLWSPaths  # noqa: E402

PY = "/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python"
PATHS = PLWSPaths(ROOT)


def export_jobs(model: str, dataset: str, seed: int) -> None:
    env = os.environ.copy()
    env["PLWS_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
        os.pathsep
    )
    subprocess.run(
        [
            PY,
            str(ROOT / "scripts" / "export_leftover_suppress_jobs.py"),
            "--model-tag",
            model,
            "--seed",
            str(seed),
            "--datasets",
            dataset,
            "--kinds",
            "firstwin",
            "--k",
            "4",
            "--lexicon",
            "core",
        ],
        cwd=ROOT,
        check=True,
        env=env,
    )


def main() -> int:
    imported: dict[str, dict[str, int]] = {}
    for model in ("qwen3_4b", "qwen3_8b"):
        for seed in SEEDS:
            copied = import_aime4s_core(PATHS, model, seed)
            if any(copied.values()):
                imported[f"{model}:s{seed}"] = copied

    exported: list[str] = []
    merged: list[str] = []
    for model in ALL_MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                if ensure_firstwin_jobs(PATHS, model, dataset, seed):
                    merged.append(f"{model} {dataset} s{seed}")
                if needs_cpu_export(PATHS, model, dataset, seed):
                    export_jobs(model, dataset, seed)
                    exported.append(f"{model} {dataset} s{seed}")

    print(
        json.dumps(
            {
                "imported_aime4s": imported,
                "merged_firstwin": merged,
                "exported_jobs": exported,
                "five_models": list(FIVE_MODELS),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
