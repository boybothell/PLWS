#!/usr/bin/env python3
"""Fill idle cards with official Answer Convergence / Dynasor cells."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plws.contest import (  # noqa: E402
    MODELS,
    _proc_environ,
    claimed_gpu_ids,
    engine_loaded_from_logs,
    gpu_count,
    idle_contest_gpus,
    leftover_8b_gpus,
    take_gpu_lane,
    vllm_workers_loading,
)
from plws.extra_baselines import (  # noqa: E402
    EXTRA_BASELINE_DATASETS,
    EXTRA_BASELINE_METHODS,
    EXTRA_BASELINE_MODELS,
    build_extra_baseline_plan,
    extra_baseline_complete,
    extra_baseline_output_dir,
    extra_baseline_sample_dir,
    extra_baseline_task_id,
    extra_cell_method_from_cmd,
    select_extra_start,
)
from plws.grading import require_grader  # noqa: E402
from plws.host_protocol import validate_fullcot_sample_meta  # noqa: E402
from plws.machine_runs import resolve_run_root  # noqa: E402
from plws.runtime import load_dotenv  # noqa: E402

load_dotenv(ROOT)

PY = os.environ.get("PLWS_PY", sys.executable)
RUN_ROOT = resolve_run_root(
    ROOT,
    env_name="EXTRA_BASELINE_RUN_ROOT",
    default_name="extra_baseline_fill",
)
LOG_ROOT = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.json"
EVENTS_PATH = RUN_ROOT / "events.jsonl"


@dataclass(frozen=True, slots=True)
class Task:
    method: str
    model_tag: str
    dataset: str
    seed: int

    @property
    def task_id(self) -> str:
        return extra_baseline_task_id(
            self.method, self.model_tag, self.dataset, self.seed
        )

    @property
    def output_dir(self) -> Path:
        return extra_baseline_output_dir(
            ROOT, self.method, self.model_tag, self.dataset, self.seed
        )


@dataclass(slots=True)
class Running:
    task: Task
    pid: int
    gpu: str
    log_path: Path
    attempt: int
    started_at: str


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


def process_state(pid: int) -> str | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return line.split()[1]
    except OSError:
        return None
    return None


def process_alive(pid: int) -> bool:
    state = process_state(pid)
    return state is not None and not state.startswith("Z")


def existing_attempts(task_id: str) -> int:
    values = []
    for path in LOG_ROOT.glob(f"{task_id}.attempt_*.log"):
        suffix = path.name[len(task_id) + len(".attempt_") : -len(".log")]
        if suffix.isdigit():
            values.append(int(suffix))
    return max(values, default=0)


def command_for(task: Task) -> list[str]:
    if task.method == "answer_convergence":
        return ["bash", str(ROOT / "scripts" / "run_answer_convergence_cell.sh")]
    if task.method == "dynasor":
        return ["bash", str(ROOT / "baselines" / "dynasor" / "run_cell.sh")]
    raise ValueError(task.method)


def env_for(task: Task, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PLWS_ROOT": str(ROOT),
            "PLWS_PY": PY,
            "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
                os.pathsep
            ),
            "MODEL": MODELS[task.model_tag],
            "MODEL_TAG": task.model_tag,
            "DATASET": task.dataset,
            "SEED": str(task.seed),
            "GPU": gpu,
            "OUT": str(task.output_dir),
        }
    )
    return env


def launch(task: Task, gpu: str, attempt: int) -> Running:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{task.task_id}.attempt_{attempt}.log"
    with log_path.open("ab") as handle:
        handle.write(f"\n# {now()} start gpus={gpu}\n".encode())
    handle = log_path.open("ab")
    process = subprocess.Popen(
        command_for(task),
        cwd=ROOT,
        env=env_for(task, gpu),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return Running(
        task=task,
        pid=process.pid,
        gpu=gpu,
        log_path=log_path,
        attempt=attempt,
        started_at=now(),
    )


def task_is_done(task: Task) -> tuple[bool, str]:
    return extra_baseline_complete(
        ROOT, task.method, task.model_tag, task.dataset, task.seed
    )


def write_status(
    pending: list[Task],
    running: dict[int, Running],
    succeeded: list[str],
    failed: list[str],
    reconciled: list[str],
    *,
    waiting: str | None = None,
    cancelled: bool = False,
) -> None:
    atomic_json(
        STATUS_PATH,
        {
            "updated_at": now(),
            "protocol_id": "puma-fullcot-32k-v2",
            "lane": "extra_baseline",
            "cancelled": cancelled,
            "waiting": waiting,
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
                    "pid": item.pid,
                    "attempt": item.attempt,
                    "started_at": item.started_at,
                    "log": str(item.log_path.relative_to(ROOT)),
                    "loading": not engine_loaded_from_logs([item.log_path]),
                }
                for item in running.values()
            ],
            "pending": [task.task_id for task in pending],
            "succeeded": succeeded,
            "failed": failed,
            "reconciled": reconciled,
        },
    )


def validate_tasks(tasks: list[Task]) -> None:
    for script in (
        ROOT / "scripts" / "run_answer_convergence_cell.sh",
        ROOT / "baselines" / "dynasor" / "run_cell.sh",
    ):
        if not script.is_file():
            raise FileNotFoundError(script)
    for task in tasks:
        model = Path(MODELS[task.model_tag])
        if not (model / "config.json").is_file():
            raise FileNotFoundError(model)
        sample_dir = extra_baseline_sample_dir(
            ROOT, task.model_tag, task.dataset, task.seed
        )
        sample = sample_dir / "answers.json"
        rows = json.loads(sample.read_text(encoding="utf-8"))
        if not rows:
            raise ValueError(f"empty sample: {sample}")
        if any(row.get("dataset") != task.dataset for row in rows):
            raise ValueError(f"dataset mismatch: {sample}")
        validate_fullcot_sample_meta(
            sample_dir / "sample_meta.json",
            model_tag=task.model_tag,
            dataset=task.dataset,
            seed=task.seed,
        )


def discover_live_extra_cells() -> dict[str, Running]:
    found: dict[str, Running] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        method = extra_cell_method_from_cmd(cmd)
        if method is None:
            continue
        env = _proc_environ(proc.name)
        try:
            task = Task(method, env["MODEL_TAG"], env["DATASET"], int(env["SEED"]))
        except (KeyError, ValueError):
            continue
        gpu = env.get("GPU") or env.get("CUDA_VISIBLE_DEVICES") or ""
        if not gpu or task.task_id in found:
            continue
        found[task.task_id] = Running(
            task=task,
            pid=int(proc.name),
            gpu=gpu,
            log_path=LOG_ROOT / f"{task.task_id}.attempt_{max(1, existing_attempts(task.task_id))}.log",
            attempt=max(1, existing_attempts(task.task_id)),
            started_at=now(),
        )
    return found


def extra_queue_alive(*, exclude_pid: int | None = None) -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        if exclude_pid is not None and int(proc.name) == exclude_pid:
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if any(part.endswith("run_extra_baseline_queue.py") for part in cmd):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,7")
    parser.add_argument("--poll", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", default=",".join(EXTRA_BASELINE_MODELS))
    parser.add_argument("--datasets", default=",".join(EXTRA_BASELINE_DATASETS))
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in (42, 0, 1))
    )
    parser.add_argument("--methods", default=",".join(EXTRA_BASELINE_METHODS))
    parser.add_argument(
        "--allow-parallel",
        action="store_true",
        help="Allow a second extra-baseline lane.",
    )
    args = parser.parse_args()
    require_grader()
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must be non-empty and unique")
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    planned = [
        Task(method, model_tag, dataset, seed)
        for method, model_tag, dataset, seed in build_extra_baseline_plan(
            models=models, datasets=datasets, seeds=seeds, methods=methods
        )
    ]
    validate_tasks(planned)
    if args.dry_run:
        print(f"validated {len(planned)} extra-baseline cells")
        print(f"GPU pool: {','.join(gpus)}")
        return 0
    if not args.allow_parallel and extra_queue_alive(exclude_pid=os.getpid()):
        raise RuntimeError("another extra-baseline queue is already running")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pending: list[Task] = []
    succeeded: list[str] = []
    failed: list[str] = []
    reconciled: list[str] = []
    live = discover_live_extra_cells()
    running: dict[int, Running] = {}
    for task in planned:
        done, reason = task_is_done(task)
        if done:
            succeeded.append(task.task_id)
            reconciled.append(task.task_id)
            event("task_reconciled", task_id=task.task_id, reason=reason)
            continue
        adopted = live.get(task.task_id)
        if adopted is not None:
            running[adopted.pid] = adopted
            event(
                "task_adopted",
                task_id=task.task_id,
                gpu=adopted.gpu,
                pid=adopted.pid,
            )
            continue
        pending.append(task)

    attempts = {task.task_id: existing_attempts(task.task_id) for task in planned}
    run_attempts: dict[str, int] = {}
    stopping = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        event("stop_requested", signal=signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    event(
        "queue_started",
        tasks=len(planned),
        pending=len(pending),
        gpus=list(gpus),
    )

    try:
        while pending or running:
            for pid, item in list(running.items()):
                if process_alive(pid):
                    continue
                del running[pid]
                done, reason = task_is_done(item.task)
                run_attempts[item.task.task_id] = (
                    run_attempts.get(item.task.task_id, 0) + 1
                )
                if done:
                    succeeded.append(item.task.task_id)
                    event(
                        "task_succeeded",
                        task_id=item.task.task_id,
                        gpu=item.gpu,
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
                        reason=reason,
                    )
                else:
                    failed.append(item.task.task_id)
                    event(
                        "task_failed",
                        task_id=item.task.task_id,
                        reason=reason,
                    )

            if stopping:
                write_status(
                    pending, running, succeeded, failed, reconciled, cancelled=True
                )
                return 130

            claimed = claimed_gpu_ids(item.gpu for item in running.values())
            leftover = leftover_8b_gpus()
            idle = idle_contest_gpus(gpus, claimed, leftover=leftover)
            still_loading = [
                item
                for item in running.values()
                if not engine_loaded_from_logs([item.log_path])
            ]
            foreign_loading = vllm_workers_loading()
            any_loading = bool(still_loading) or foreign_loading
            waiting = None
            if leftover & set(gpus):
                waiting = "8B leftover holding " + ",".join(
                    sorted(leftover & set(gpus), key=int)
                )
            elif idle and pending and any_loading:
                names = [item.task.task_id for item in still_loading]
                if foreign_loading:
                    names.append("foreign-vllm")
                waiting = "serial load: waiting for " + ",".join(names)
            starts = select_extra_start(
                pending, idle_count=len(idle), any_loading=any_loading
            )
            launched = 0
            while idle and starts:
                task = starts.pop(0)
                if task not in pending:
                    continue
                lane = take_gpu_lane(idle, gpu_count(task.model_tag))
                if lane is None:
                    continue
                pending.remove(task)
                attempt = attempts[task.task_id] + 1
                attempts[task.task_id] = attempt
                item = launch(task, lane, attempt)
                running[item.pid] = item
                launched += 1
                event(
                    "task_started",
                    task_id=task.task_id,
                    gpu=lane,
                    pid=item.pid,
                    attempt=attempt,
                )
                # Whole machine: only one new engine at a time.
                break

            write_status(
                pending,
                running,
                succeeded,
                failed,
                reconciled,
                waiting=waiting,
            )
            if running or launched:
                time.sleep(max(1, args.poll))
            elif pending:
                time.sleep(max(1, args.poll))
    finally:
        write_status(pending, running, succeeded, failed, reconciled)

    event("queue_finished", succeeded=len(succeeded), failed=len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
