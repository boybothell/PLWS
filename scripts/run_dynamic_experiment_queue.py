#!/usr/bin/env python3
"""Resource-aware queue for official DEER and Qwen PUMA/PLWS cells."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "runs" / "dynamic_experiment_queue"
LOG_ROOT = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.json"
EVENTS_PATH = RUN_ROOT / "events.jsonl"

DATASETS = ("math", "olympiadbench", "gpqa", "aime", "aime25")
PUMA_DATASETS = {
    "math": "math-500",
    "olympiadbench": "olympiadbench",
    "gpqa": "gpqa-diamond",
    "aime": "aime24",
    "aime25": "aime25",
}
SEEDS = (0, 1, 42, 123)

MODELS = {
    "r1_7b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B",
    "nemotron_8b": "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1",
    "r1_14b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B",
    "qwen3_4b": "/mnt/d/lsj/models/Qwen3-4B",
    "qwen3_8b": "/mnt/d/lsj/models/Qwen3-8B",
    "qwen3_30b_a3b": "/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    priority: int
    gpu_count: int
    command: tuple[str, ...]
    env: dict[str, str]
    description: str


@dataclass
class Running:
    task: Task
    process: subprocess.Popen[bytes]
    gpus: tuple[str, ...]
    attempt: int
    run_attempt: int
    log_handle: object
    log_path: Path
    started_at: str


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def event(kind: str, **fields: object) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": now(), "event": kind, **fields}
    with EVENTS_PATH.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def deer_task(model: str, dataset: str, *, priority: int) -> Task:
    qwen = model.startswith("qwen3_")
    result_root = (
        ROOT
        / "results"
        / "baselines"
        / "deer"
        / (
            "github_official_greedy_16k_qwen3"
            if qwen
            else "github_official_greedy_16k_priority_7b_8b_14b"
        )
    )
    gpu_count = 2 if model == "qwen3_30b_a3b" else 1
    return Task(
        task_id=f"deer__{model}__{dataset}",
        priority=priority,
        gpu_count=gpu_count,
        command=("bash", str(ROOT / "scripts" / "run_deer_github_official_cell.sh")),
        env={
            "MODEL": MODELS[model],
            "MODEL_TAG": model,
            "DATASET": dataset,
            "FAMILY": "qwen3" if qwen else "standard",
            "OUT": str(result_root / model / dataset),
        },
        description=f"official DEER {model} {dataset}",
    )


def qwen_plws_task(model: str, dataset: str, seed: int) -> Task:
    gpu_count = 2 if model == "qwen3_30b_a3b" else 1
    puma_dataset = PUMA_DATASETS[dataset]
    return Task(
        task_id=f"puma_plws__{model}__{puma_dataset}__s{seed}",
        priority=20,
        gpu_count=gpu_count,
        command=("bash", str(ROOT / "scripts" / "run_qwen_plws_cell.sh")),
        env={
            "MODEL_TAG": model,
            "DATASET": puma_dataset,
            "SEED": str(seed),
        },
        description=f"PUMA + PLWS {model} {puma_dataset} seed={seed}",
    )


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    for model in ("r1_7b", "nemotron_8b", "r1_14b"):
        for dataset in DATASETS:
            tasks.append(deer_task(model, dataset, priority=0))
    for model in ("qwen3_4b", "qwen3_8b", "qwen3_30b_a3b"):
        for dataset in DATASETS:
            tasks.append(deer_task(model, dataset, priority=10))
    for model in ("qwen3_4b", "qwen3_8b", "qwen3_30b_a3b"):
        for dataset in DATASETS:
            for seed in SEEDS:
                tasks.append(qwen_plws_task(model, dataset, seed))
    return sorted(tasks, key=lambda task: (task.priority, task.task_id))


def nonempty_line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def deer_complete(task: Task) -> tuple[bool, str]:
    output = Path(task.env["OUT"])
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False, "missing manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid manifest: {exc}"
    if (
        manifest.get("model_tag") != task.env["MODEL_TAG"]
        or manifest.get("dataset") != task.env["DATASET"]
    ):
        return False, "manifest identity mismatch"
    data_path = ROOT.parent / "DEER" / "data" / task.env["DATASET"] / "test.jsonl"
    expected = nonempty_line_count(data_path)
    result_files = sorted(output.rglob("*.jsonl"))
    complete = [
        path for path in result_files if nonempty_line_count(path) == expected
    ]
    if not complete:
        counts = [nonempty_line_count(path) for path in result_files]
        return False, f"expected {expected} records, found {counts or 'no JSONL'}"
    return True, f"{expected} records"


def puma_plws_complete(task: Task) -> tuple[bool, str]:
    model = task.env["MODEL_TAG"]
    dataset = task.env["DATASET"]
    seed = int(task.env["SEED"])
    if seed == 42:
        puma_dir = (
            ROOT / "results" / "baselines" / "puma"
            / f"puma_offline_{model}" / dataset
        )
    else:
        puma_dir = (
            ROOT / "results" / "baselines" / "puma"
            / f"puma_offline_{model}_s{seed}" / dataset
        )
    for name in ("statistics.json", "prefixed_answers.json"):
        if not (puma_dir / name).is_file():
            return False, f"missing PUMA {name}"

    dense_root = (
        ROOT / "results" / "upstream" / "dense_trials"
        / f"dense_G_{model}" / dataset / f"seed_{seed}"
    )
    if not (dense_root / "dense_puma" / "trial_answers.json").is_file():
        return False, "missing dense trials"
    if not (dense_root / "per_sample.json").is_file():
        return False, "missing dense per_sample"

    cell = (
        ROOT / "results" / "runs" / "plws" / "window_first"
        / "k_4" / "lexicon_core" / model / dataset / f"seed_{seed}"
    )
    shard_pattern = re.compile(r"^shard_\d+\.jsonl$")
    for kind in ("low", "mix", "high"):
        jobs_path = cell / "jobs" / f"{kind}.jsonl"
        if not jobs_path.is_file():
            return False, f"missing {kind} jobs"
        jobs = {
            str(record["uid"])
            for record in jsonl_records(jobs_path)
            if record.get("uid")
        }
        scores: set[str] = set()
        score_dir = cell / "scores" / kind
        if score_dir.is_dir():
            for path in sorted(score_dir.iterdir()):
                if not shard_pattern.match(path.name):
                    continue
                for record in jsonl_records(path):
                    if (
                        record.get("uid")
                        and record.get("status") in ("ok", "too_long")
                    ):
                        scores.add(str(record["uid"]))
        missing = jobs - scores
        if missing:
            return False, f"{kind} scores missing {len(missing)}/{len(jobs)}"
    return True, "PUMA, dense, jobs and PLWS scores complete"


def task_complete(task: Task) -> tuple[bool, str]:
    try:
        if task.task_id.startswith("deer__"):
            return deer_complete(task)
        if task.task_id.startswith("puma_plws__"):
            return puma_plws_complete(task)
        return False, "unknown task kind"
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        return False, f"completion check error: {exc}"


def existing_attempts(task_id: str) -> int:
    pattern = re.compile(rf"^{re.escape(task_id)}\.attempt_(\d+)\.log$")
    attempts = [
        int(match.group(1))
        for path in LOG_ROOT.glob(f"{task_id}.attempt_*.log")
        if (match := pattern.match(path.name))
    ]
    return max(attempts, default=0)


def validate(tasks: list[Task], gpus: tuple[str, ...]) -> None:
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU pool contains duplicates")
    if len(gpus) < max(task.gpu_count for task in tasks):
        raise ValueError("GPU pool is too small for the largest task")
    for script in (
        ROOT / "scripts" / "run_deer_github_official_cell.sh",
        ROOT / "scripts" / "run_qwen_plws_cell.sh",
    ):
        if not script.is_file():
            raise FileNotFoundError(script)
    for path in MODELS.values():
        if not (Path(path) / "config.json").is_file():
            raise FileNotFoundError(f"model missing: {path}")
    deer_data = ROOT.parent / "DEER" / "data"
    for dataset in DATASETS:
        if not (deer_data / dataset / "test.jsonl").is_file():
            raise FileNotFoundError(deer_data / dataset / "test.jsonl")


def write_status(
    pending: list[Task],
    running: dict[int, Running],
    succeeded: list[str],
    failed: list[str],
    reconciled: list[str],
    cancelled: bool = False,
) -> None:
    atomic_json(
        STATUS_PATH,
        {
            "updated_at": now(),
            "cancelled": cancelled,
            "counts": {
                "pending": len(pending),
                "running": len(running),
                "succeeded": len(succeeded),
                "failed": len(failed),
                "reconciled": len(reconciled),
            },
            "running": [
                {
                    "task_id": item.task.task_id,
                    "description": item.task.description,
                    "gpus": list(item.gpus),
                    "pid": item.process.pid,
                    "attempt": item.attempt,
                    "run_attempt": item.run_attempt,
                    "started_at": item.started_at,
                    "log": str(item.log_path.relative_to(ROOT)),
                }
                for item in running.values()
            ],
            "pending": [task.task_id for task in pending],
            "succeeded": succeeded,
            "failed": failed,
            "reconciled": reconciled,
        },
    )


def terminate_all(running: dict[int, Running]) -> None:
    for item in running.values():
        try:
            os.killpg(item.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 20
    while time.time() < deadline and any(item.process.poll() is None for item in running.values()):
        time.sleep(1)
    for item in running.values():
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=os.environ.get("DYNAMIC_GPUS", "0,1,2,3,4,7"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    tasks = build_tasks()
    validate(tasks, gpus)
    if args.dry_run:
        by_kind: dict[tuple[int, int], int] = {}
        for task in tasks:
            key = (task.priority, task.gpu_count)
            by_kind[key] = by_kind.get(key, 0) + 1
        print(f"validated {len(tasks)} tasks on GPUs {','.join(gpus)}")
        for (priority, gpu_count), count in sorted(by_kind.items()):
            print(f"priority={priority} gpu_count={gpu_count}: {count}")
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pending: list[Task] = []
    running: dict[int, Running] = {}
    attempts: dict[str, int] = {
        task.task_id: existing_attempts(task.task_id) for task in tasks
    }
    run_attempts: dict[str, int] = {}
    succeeded: list[str] = []
    failed: list[str] = []
    reconciled: list[str] = []
    for task in tasks:
        complete, reason = task_complete(task)
        if complete:
            succeeded.append(task.task_id)
            reconciled.append(task.task_id)
            event("task_reconciled", task_id=task.task_id, reason=reason)
        else:
            pending.append(task)
    stopping = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        event("stop_requested", signal=signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    event(
        "queue_started",
        gpus=list(gpus),
        tasks=len(tasks),
        pending=len(pending),
        reconciled=len(reconciled),
    )

    try:
        while pending or running:
            finished: list[int] = []
            for pid, item in running.items():
                code = item.process.poll()
                if code is None:
                    continue
                item.log_handle.close()
                finished.append(pid)
                complete, reason = task_complete(item.task)
                if complete:
                    succeeded.append(item.task.task_id)
                    if code == 0:
                        event(
                            "task_succeeded",
                            task_id=item.task.task_id,
                            gpus=list(item.gpus),
                            reason=reason,
                        )
                    else:
                        reconciled.append(item.task.task_id)
                        event(
                            "task_reconciled",
                            task_id=item.task.task_id,
                            gpus=list(item.gpus),
                            returncode=code,
                            reason=reason,
                        )
                elif item.run_attempt < args.max_attempts and not stopping:
                    pending.append(item.task)
                    pending.sort(key=lambda task: (task.priority, task.task_id))
                    event(
                        "task_retry",
                        task_id=item.task.task_id,
                        returncode=code,
                        attempt=item.attempt,
                        run_attempt=item.run_attempt,
                        incomplete_reason=reason,
                    )
                else:
                    failed.append(item.task.task_id)
                    event(
                        "task_failed",
                        task_id=item.task.task_id,
                        returncode=code,
                        attempt=item.attempt,
                        run_attempt=item.run_attempt,
                        incomplete_reason=reason,
                    )
            for pid in finished:
                del running[pid]

            if stopping:
                write_status(
                    pending,
                    running,
                    succeeded,
                    failed,
                    reconciled,
                    cancelled=True,
                )
                terminate_all(running)
                return 130

            used = {gpu for item in running.values() for gpu in item.gpus}
            free = [gpu for gpu in gpus if gpu not in used]
            launched = False
            while free:
                fit_index = next(
                    (i for i, task in enumerate(pending) if task.gpu_count <= len(free)),
                    None,
                )
                if fit_index is None:
                    break
                task = pending.pop(fit_index)
                assigned = tuple(free[: task.gpu_count])
                del free[: task.gpu_count]
                attempt = attempts.get(task.task_id, 0) + 1
                attempts[task.task_id] = attempt
                run_attempt = run_attempts.get(task.task_id, 0) + 1
                run_attempts[task.task_id] = run_attempt
                log_path = LOG_ROOT / f"{task.task_id}.attempt_{attempt}.log"
                log_handle = log_path.open("wb")
                env = os.environ.copy()
                env.update(task.env)
                env.update(
                    {
                        "PLWS_ROOT": str(ROOT),
                        "GPU": ",".join(assigned),
                    }
                )
                process = subprocess.Popen(
                    task.command,
                    cwd=ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[process.pid] = Running(
                    task=task,
                    process=process,
                    gpus=assigned,
                    attempt=attempt,
                    run_attempt=run_attempt,
                    log_handle=log_handle,
                    log_path=log_path,
                    started_at=now(),
                )
                event(
                    "task_started",
                    task_id=task.task_id,
                    description=task.description,
                    gpus=list(assigned),
                    pid=process.pid,
                    attempt=attempt,
                    run_attempt=run_attempt,
                )
                launched = True

            write_status(pending, running, succeeded, failed, reconciled)
            if running:
                time.sleep(5)
            elif pending and not launched:
                raise RuntimeError("pending tasks remain but none fit the GPU pool")
    finally:
        for item in running.values():
            try:
                item.log_handle.close()
            except Exception:
                pass

    event("queue_finished", succeeded=len(succeeded), failed=len(failed))
    write_status(pending, running, succeeded, failed, reconciled)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

