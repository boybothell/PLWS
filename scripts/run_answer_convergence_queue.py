#!/usr/bin/env python3
"""Resource-aware queue for canonical Answer Convergence cells."""

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
sys.path.insert(0, str(ROOT / "src"))

from plws.protocol import (  # noqa: E402
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROMPT_RESERVE_TOKENS,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)
from plws.host_protocol import validate_fullcot_sample_meta  # noqa: E402

RUN_ROOT = ROOT / "results" / "runs" / "answer_convergence_queue"
LOG_ROOT = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.json"
EVENTS_PATH = RUN_ROOT / "events.jsonl"
OUTPUT_ROOT = (
    ROOT
    / "results"
    / "baselines"
    / "answer_convergence"
    / "puma_fullcot_32k_v2_k10"
)

MODELS = (
    ("r1_7b", "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"),
    ("nemotron_8b", "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"),
    ("r1_14b", "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"),
)
DATASETS = ("aime24", "math-500", "olympiadbench", "gpqa-diamond", "aime25")
SEEDS = (0, 1, 42, 123)


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    model_tag: str
    model: str
    dataset: str
    seed: int

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.model_tag / self.dataset / f"seed_{self.seed}"


@dataclass(slots=True)
class Running:
    task: Task
    process: subprocess.Popen[bytes]
    gpu: str
    log_handle: object
    log_path: Path
    attempt: int


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def event(kind: str, **fields: object) -> None:
    payload = {"time": now(), "event": kind, **fields}
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def expected_count(task: Task) -> int:
    sample = (
        ROOT
        / "samples"
        / task.model_tag
        / task.dataset
        / f"seed_{task.seed}"
        / "answers.json"
    )
    rows = json.loads(sample.read_text(encoding="utf-8"))
    return len(rows)


def complete(task: Task) -> tuple[bool, str]:
    output_dir = task.output_dir
    manifest_path = output_dir / "manifest.json"
    output_path = output_dir / "final_answers.jsonl"
    if not manifest_path.is_file() or not output_path.is_file():
        return False, "missing manifest or final_answers.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        count = len(rows)
        expected = expected_count(task)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid output: {exc}"
    valid = (
        manifest.get("method") == "answer_convergence"
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("model_tag") == task.model_tag
        and manifest.get("dataset") == task.dataset
        and manifest.get("seed") == task.seed
        and manifest.get("threshold") == 10
        and manifest.get("fullcot_generation_tokens") == FULLCOT_GENERATION_TOKENS
        and manifest.get("prompt_reserve_tokens") == PROMPT_RESERVE_TOKENS
        and manifest.get("truncated_answer_fix_tokens") == TRUNCATED_ANSWER_FIX_TOKENS
        and manifest.get("max_model_len") == MAX_MODEL_LEN
        and manifest.get("expected_records") == expected
        and count == expected
        and all(row.get("protocol_id") == PROTOCOL_ID for row in rows)
    )
    return (True, f"{count} records") if valid else (
        False,
        f"identity/count mismatch count={count} expected={expected}",
    )


def existing_attempts(task_id: str) -> int:
    pattern = re.compile(rf"^{re.escape(task_id)}\.attempt_(\d+)\.log$")
    values = [
        int(match.group(1))
        for path in LOG_ROOT.glob(f"{task_id}.attempt_*.log")
        if (match := pattern.match(path.name))
    ]
    return max(values, default=0)


def write_status(
    pending: list[Task],
    running: dict[int, Running],
    succeeded: list[str],
    failed: list[str],
    reconciled: list[str],
    *,
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
                    "gpu": item.gpu,
                    "pid": item.process.pid,
                    "attempt": item.attempt,
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


def validate(tasks: list[Task]) -> None:
    wrapper = ROOT / "scripts" / "run_answer_convergence_cell.sh"
    if not wrapper.is_file():
        raise FileNotFoundError(wrapper)
    for task in tasks:
        if not (Path(task.model) / "config.json").is_file():
            raise FileNotFoundError(task.model)
        sample = (
            ROOT
            / "samples"
            / task.model_tag
            / task.dataset
            / f"seed_{task.seed}"
            / "answers.json"
        )
        rows = json.loads(sample.read_text(encoding="utf-8"))
        if not rows:
            raise ValueError(f"empty sample: {sample}")
        if any(row.get("dataset") != task.dataset for row in rows):
            raise ValueError(f"dataset mismatch: {sample}")
        validate_fullcot_sample_meta(
            sample.parent / "sample_meta.json",
            model_tag=task.model_tag,
            dataset=task.dataset,
            seed=task.seed,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,7")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must be non-empty and unique")

    tasks = [
        Task(
            task_id=f"ansconv__{model_tag}__{dataset}__s{seed}",
            model_tag=model_tag,
            model=model,
            dataset=dataset,
            seed=seed,
        )
        for model_tag, model in MODELS
        for dataset in DATASETS
        for seed in SEEDS
    ]
    validate(tasks)
    if args.dry_run:
        print(f"validated {len(tasks)} Answer Convergence cells")
        print(f"GPU pool: {','.join(gpus)}")
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pending: list[Task] = []
    succeeded: list[str] = []
    failed: list[str] = []
    reconciled: list[str] = []
    for task in tasks:
        is_complete, reason = complete(task)
        if is_complete:
            succeeded.append(task.task_id)
            reconciled.append(task.task_id)
            event("task_reconciled", task_id=task.task_id, reason=reason)
        else:
            pending.append(task)

    running: dict[int, Running] = {}
    attempts = {task.task_id: existing_attempts(task.task_id) for task in tasks}
    run_attempts: dict[str, int] = {}
    stopping = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        event("stop_requested", signal=signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    event("queue_started", tasks=len(tasks), pending=len(pending), gpus=list(gpus))

    try:
        while pending or running:
            for pid, item in list(running.items()):
                code = item.process.poll()
                if code is None:
                    continue
                item.log_handle.close()
                del running[pid]
                is_complete, reason = complete(item.task)
                run_attempts[item.task.task_id] = (
                    run_attempts.get(item.task.task_id, 0) + 1
                )
                if is_complete:
                    succeeded.append(item.task.task_id)
                    event(
                        "task_succeeded",
                        task_id=item.task.task_id,
                        gpu=item.gpu,
                        returncode=code,
                        reason=reason,
                    )
                elif (
                    run_attempts[item.task.task_id] < args.max_attempts
                    and not stopping
                ):
                    pending.append(item.task)
                    event(
                        "task_retry",
                        task_id=item.task.task_id,
                        returncode=code,
                        reason=reason,
                    )
                else:
                    failed.append(item.task.task_id)
                    event(
                        "task_failed",
                        task_id=item.task.task_id,
                        returncode=code,
                        reason=reason,
                    )

            if stopping:
                for item in running.values():
                    try:
                        os.killpg(item.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                write_status(
                    pending,
                    running,
                    succeeded,
                    failed,
                    reconciled,
                    cancelled=True,
                )
                return 130

            used = {item.gpu for item in running.values()}
            free = [gpu for gpu in gpus if gpu not in used]
            while free and pending:
                task = pending.pop(0)
                gpu = free.pop(0)
                attempt = attempts[task.task_id] + 1
                attempts[task.task_id] = attempt
                log_path = LOG_ROOT / f"{task.task_id}.attempt_{attempt}.log"
                log_handle = log_path.open("wb")
                env = os.environ.copy()
                env.update(
                    {
                        "PLWS_ROOT": str(ROOT),
                        "MODEL": task.model,
                        "MODEL_TAG": task.model_tag,
                        "DATASET": task.dataset,
                        "SEED": str(task.seed),
                        "GPU": gpu,
                        "OUT": str(task.output_dir),
                    }
                )
                process = subprocess.Popen(
                    ("bash", str(ROOT / "scripts" / "run_answer_convergence_cell.sh")),
                    cwd=ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[process.pid] = Running(
                    task=task,
                    process=process,
                    gpu=gpu,
                    log_handle=log_handle,
                    log_path=log_path,
                    attempt=attempt,
                )
                event(
                    "task_started",
                    task_id=task.task_id,
                    gpu=gpu,
                    pid=process.pid,
                    attempt=attempt,
                )

            write_status(pending, running, succeeded, failed, reconciled)
            if running:
                time.sleep(5)
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
    raise SystemExit(main())
