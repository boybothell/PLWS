#!/usr/bin/env python3
"""GPU queue for PLWS-only completion and PUMA-aligned DEER."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "run_dynamic_experiment_queue.py"
spec = importlib.util.spec_from_file_location("dynamic_queue_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load scheduler: {SOURCE}")
queue = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = queue
spec.loader.exec_module(queue)

RUN_ROOT = ROOT / "results" / "runs" / "aligned_plws_deer_queue"
queue.RUN_ROOT = RUN_ROOT
queue.LOG_ROOT = RUN_ROOT / "logs"
queue.STATUS_PATH = RUN_ROOT / "status.json"
queue.EVENTS_PATH = RUN_ROOT / "events.jsonl"

PUMA_DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
SEEDS = (0, 1, 42, 123)
DEER_MODELS = (
    "r1_7b",
    "nemotron_8b",
    "r1_14b",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_30b_a3b",
)
PLWS_MODELS = ("qwen3_4b", "qwen3_8b", "qwen3_30b_a3b")
DEER_ROOT = ROOT / "results" / "baselines" / "deer" / "puma_fullcot_32k_v2"
QUEUE_MODE = os.environ.get("ALIGNED_QUEUE_MODE", "all").strip().lower()
if QUEUE_MODE not in {"all", "deer", "plws"}:
    raise ValueError("ALIGNED_QUEUE_MODE must be one of: all, deer, plws")


def nonempty_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def puma_dir(model: str, dataset: str, seed: int) -> Path:
    tag = f"puma_offline_{model}" if seed == 42 else f"puma_offline_{model}_s{seed}"
    return ROOT / "results" / "baselines" / "puma" / tag / dataset


def deer_task(model: str, dataset: str, seed: int) -> queue.Task:
    return queue.Task(
        task_id=f"aligned_deer__{model}__{dataset}__s{seed}",
        priority=10,
        gpu_count=2 if model == "qwen3_30b_a3b" else 1,
        command=("bash", str(ROOT / "scripts" / "run_deer_official.sh")),
        env={
            "MODEL": queue.MODELS[model],
            "MODEL_TAG": model,
            "DATASET": dataset,
            "SEED": str(seed),
            "OUT": str(DEER_ROOT / model / dataset / f"seed_{seed}"),
        },
        description=f"PUMA-aligned DEER {model} {dataset} seed={seed}",
    )


def plws_task(model: str, dataset: str, seed: int) -> queue.Task:
    return queue.Task(
        task_id=f"puma_plws__{model}__{dataset}__s{seed}",
        priority=10,
        gpu_count=2 if model == "qwen3_30b_a3b" else 1,
        command=("bash", str(ROOT / "scripts" / "run_qwen_plws_cell.sh")),
        env={"MODEL_TAG": model, "DATASET": dataset, "SEED": str(seed)},
        description=f"PUMA + PLWS {model} {dataset} seed={seed}",
    )


def deer_complete(task: queue.Task) -> tuple[bool, str]:
    output = Path(task.env["OUT"])
    result = output / "deer.jsonl"
    manifest = output / "manifest.json"
    data = ROOT.parent / "PUMA" / "data" / f"{task.env['DATASET']}_test.jsonl"
    if not result.is_file() or not manifest.is_file():
        return False, "missing deer.jsonl or manifest"
    try:
        identity = json.loads(manifest.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in result.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        actual = len(rows)
        expected = nonempty_count(data)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid DEER output: {exc}"
    is_qwen3 = task.env["MODEL_TAG"].startswith("qwen3_")
    expected_profile = "qwen3" if is_qwen3 else "standard"
    expected_ratio = 0.8 if is_qwen3 else 0.6
    expected_policy = "avg2" if is_qwen3 else "avg1"
    deer = identity.get("deer", {})
    valid = (
        identity.get("model_tag") == task.env["MODEL_TAG"]
        and identity.get("protocol_id") == "puma-fullcot-32k-v2"
        and identity.get("dataset") == task.env["DATASET"]
        and identity.get("seed") == int(task.env["SEED"])
        and identity.get("fullcot_generation_tokens") == 32768
        and identity.get("truncated_answer_fix_tokens") == 2048
        and identity.get("max_model_len") == 37888
        and deer.get("family_policy") == expected_profile
        and deer.get("think_ratio") == expected_ratio
        and deer.get("confidence_policy") == expected_policy
        and deer.get("require_probe_think_close") is is_qwen3
        and actual == expected
        and all(row.get("protocol_id") == "puma-fullcot-32k-v2" for row in rows)
        and all(row.get("deer_family_policy") == expected_profile for row in rows)
        and all(row.get("think_ratio") == expected_ratio for row in rows)
        and all(row.get("confidence_policy") == expected_policy for row in rows)
        and all("Error:" not in str(row.get("generated_text", "")) for row in rows)
    )
    return (True, f"{actual} records") if valid else (
        False,
        f"identity/count mismatch actual={actual} expected={expected}",
    )


def plws_complete(task: queue.Task) -> tuple[bool, str]:
    return queue.puma_plws_complete(task)


def task_complete(task: queue.Task) -> tuple[bool, str]:
    if task.task_id.startswith("aligned_deer__"):
        return deer_complete(task)
    if task.task_id.startswith("puma_plws__"):
        return plws_complete(task)
    return False, "unknown task type"


def has_plws_prerequisites(model: str, dataset: str, seed: int) -> bool:
    directory = puma_dir(model, dataset, seed)
    return all(
        (directory / name).is_file()
        for name in (
            "statistics.json",
            "prefixed_answers.json",
            "filtered_steps.json",
            "answers.json",
        )
    )


def build_tasks() -> list[queue.Task]:
    deer = (
        [
            deer_task(model, dataset, seed)
            for model in DEER_MODELS
            for dataset in PUMA_DATASETS
            for seed in SEEDS
        ]
        if QUEUE_MODE in {"all", "deer"}
        else []
    )
    plws = (
        [
            plws_task(model, dataset, seed)
            for model in PLWS_MODELS
            for dataset in PUMA_DATASETS
            for seed in SEEDS
        ]
        if QUEUE_MODE in {"all", "plws"}
        else []
    )
    deer = [
        task
        for task in deer
        if not deer_complete(task)[0]
        and not (
            task.env["MODEL_TAG"] == "nemotron_8b"
            and task.env["DATASET"] == "gpqa-diamond"
        )
    ]
    plws = [task for task in plws if not plws_complete(task)[0]]
    ordered: list[queue.Task] = []
    while deer or plws:
        ordered.extend(deer[:1])
        del deer[:1]
        ordered.extend(plws[:1])
        del plws[:1]
    return ordered


def validate(tasks: list[queue.Task], gpus: tuple[str, ...]) -> None:
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        raise ValueError("aligned PLWS/DEER queue requires at least two unique GPUs")
    scripts = []
    if QUEUE_MODE in {"all", "deer"}:
        scripts.append("run_deer_official.sh")
    if QUEUE_MODE in {"all", "plws"}:
        scripts.append("run_qwen_plws_cell.sh")
    for script in scripts:
        if not (ROOT / "scripts" / script).is_file():
            raise FileNotFoundError(script)
    for task in tasks:
        if task.task_id.startswith("aligned_deer__"):
            data = ROOT.parent / "PUMA" / "data" / f"{task.env['DATASET']}_test.jsonl"
            if not data.is_file():
                raise FileNotFoundError(data)


queue.build_tasks = build_tasks
queue.task_complete = task_complete
queue.validate = validate

if __name__ == "__main__":
    raise SystemExit(queue.main())
