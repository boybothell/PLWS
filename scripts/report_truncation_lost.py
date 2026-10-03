#!/usr/bin/env python3
"""Paired loss: full CoT correct and an early-exit method incorrect.

Joins each method to the official Full-CoT ``original_correct`` flag by
question text. DEER answers use the same grader as the mean@3 table.
The other methods use the correctness stored with their per-question records.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUMA = ROOT.parent / "PUMA"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PUMA))
sys.path.insert(0, str(PUMA / "puma"))

from plws.extra_baselines import extra_baseline_output_dir  # noqa: E402
from plws.grading import require_grader  # noqa: E402
from plws.matrix import deer_output_dir  # noqa: E402
from plws.paths import PLWSPaths  # noqa: E402
from report_fullcot_puma_plws import grade_deer_item  # noqa: E402

MODELS = (("r1_7b", "R1-7B"), ("r1_14b", "R1-14B"), ("r1_32b", "R1-32B"))
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond")
SEEDS = (42, 0, 1)
METHODS = ("answer_convergence", "dynasor", "deer", "puma")
OUT = ROOT / "tables" / "truncation_lost.json"


def norm(text: object) -> str:
    return " ".join(str(text or "").split())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def delivery_path(paths: PLWSPaths, model: str, dataset: str, seed: int) -> Path:
    backfill = (
        paths.results
        / "baselines"
        / "puma"
        / "backfill"
        / model
        / dataset
        / f"seed_{seed}"
        / "statistics.json"
    )
    if backfill.is_file():
        return backfill
    return paths.puma_statistics_path(model, dataset, seed)


def full_by_question(paths: PLWSPaths, model: str, dataset: str, seed: int) -> dict[str, bool]:
    rows = json.loads(paths.puma_statistics_path(model, dataset, seed).read_text())
    out = {}
    for row in rows:
        key = norm(row.get("question"))
        if not key:
            raise RuntimeError(f"empty question {model} {dataset} seed={seed}")
        if key in out:
            raise RuntimeError(f"duplicate question {model} {dataset} seed={seed}")
        out[key] = bool(row["original_correct"])
    return out


def method_flags(
    paths: PLWSPaths, method: str, model: str, dataset: str, seed: int
) -> dict[str, bool]:
    if method in {"answer_convergence", "dynasor"}:
        path = (
            extra_baseline_output_dir(paths.root, method, model, dataset, seed)
            / "final_answers.jsonl"
        )
        rows = load_jsonl(path)
        return {norm(row.get("question")): bool(row["correct"]) for row in rows}
    if method == "puma":
        rows = json.loads(delivery_path(paths, model, dataset, seed).read_text())
        return {norm(row.get("question")): bool(row["compressed_correct"]) for row in rows}
    if method == "deer":
        path = deer_output_dir(paths, model, dataset, seed) / "deer.jsonl"
        if not path.is_file():
            return {}
        rows = load_jsonl(path)
        items = [
            (
                dataset,
                str(row.get("generated_text") or ""),
                str(row.get("gold_answer") or ""),
                0.0,
            )
            for row in rows
        ]
        with ProcessPoolExecutor(max_workers=8) as pool:
            graded = list(pool.map(grade_deer_item, items, chunksize=16))
        return {
            norm(row.get("question")): bool(ok)
            for row, (ok, _tok) in zip(rows, graded)
        }
    raise RuntimeError(method)


def accumulate(full: dict[str, bool], method: dict[str, bool]) -> tuple[int, int]:
    if not method:
        return 0, 0
    missing = [key for key in full if key not in method]
    if missing:
        raise RuntimeError(f"unjoined questions: {len(missing)}")
    full_ok = 0
    lost = 0
    for key, ok in full.items():
        if not ok:
            continue
        full_ok += 1
        if not method[key]:
            lost += 1
    return full_ok, lost


def main() -> None:
    require_grader()
    paths = PLWSPaths.discover()
    cells = []
    for model, label in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                full = full_by_question(paths, model, dataset, seed)
                for method in METHODS:
                    flags = method_flags(paths, method, model, dataset, seed)
                    if not flags:
                        cells.append(
                            {
                                "model": label,
                                "model_tag": model,
                                "dataset": dataset,
                                "seed": seed,
                                "method": method,
                                "status": "missing",
                            }
                        )
                        continue
                    full_ok, lost = accumulate(full, flags)
                    cells.append(
                        {
                            "model": label,
                            "model_tag": model,
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "status": "ok",
                            "full_correct": full_ok,
                            "lost": lost,
                            "rate": (100.0 * lost / full_ok) if full_ok else None,
                        }
                    )
                    print(
                        f"{label} {dataset} s{seed} {method} "
                        f"{lost}/{full_ok} = {cells[-1]['rate']:.1f}",
                        flush=True,
                    )
    pooled = []
    for model, label in MODELS:
        for method in METHODS:
            rows = [
                cell
                for cell in cells
                if cell["model"] == label
                and cell["method"] == method
                and cell["status"] == "ok"
            ]
            datasets = {cell["dataset"] for cell in rows}
            seeds = {cell["seed"] for cell in rows}
            complete = datasets == set(DATASETS) and seeds == set(SEEDS)
            full_ok = sum(cell["full_correct"] for cell in rows)
            lost = sum(cell["lost"] for cell in rows)
            pooled.append(
                {
                    "model": label,
                    "method": method,
                    "complete": complete,
                    "full_correct": full_ok,
                    "lost": lost,
                    "rate": (100.0 * lost / full_ok) if full_ok and complete else None,
                }
            )
    OUT.write_text(json.dumps({"cells": cells, "pooled": pooled}, indent=2) + "\n")
    print("wrote", OUT)
    for row in pooled:
        print(
            f"POOL {row['model']} {row['method']} complete={row['complete']} "
            f"{row['lost']}/{row['full_correct']} rate={row['rate']}"
        )


if __name__ == "__main__":
    main()
