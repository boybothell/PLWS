#!/usr/bin/env python3
"""Contest-set PUMA + PLWS queue. No DEER. Isolated from the main matrix."""

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
    DATASETS,
    MODELS,
    QUEUE_MODELS,
    ContestTask,
    batch_size,
    build_tasks,
    cells_needing_cpu_export,
    contest_blocked_phases,
    engine_loaded_from_logs,
    fill_leftover_dispatch,
    is_task_log_name,
    first_open_phase,
    idle_contest_gpus,
    leftover_8b_gpus,
    select_cold_starts,
    summarize,
    task_complete,
)
from plws.matrix import (  # noqa: E402
    ensure_firstwin_jobs,
    needs_cpu_export,
)
from plws.paths import PLWSPaths  # noqa: E402

PY = "/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python"
RUN_ROOT = ROOT / "results" / "runs" / "contest_queue"
LOG_ROOT = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.json"
EVENTS_PATH = RUN_ROOT / "events.jsonl"
DEFAULT_GPUS = ("0", "1", "2", "3", "4")
PATHS = PLWSPaths(ROOT)


@dataclass
class Running:
    task: ContestTask
    pid: int
    gpus: tuple[str, ...]
    adopted: bool
    started_at: str
    log_path: Path
    loaded: bool = False


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def event(kind: str, **fields: object) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a") as handle:
        handle.write(json.dumps({"time": now(), "event": kind, **fields}) + "\n")


def pid_state(pid: int) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return line.split()[1]
    except OSError:
        return ""
    return "?"


def pid_alive(pid: int) -> bool:
    state = pid_state(pid)
    if not state or state.startswith("Z"):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def proc_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def child_pids(pid: int) -> list[int]:
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text()
    except OSError:
        return []
    return [int(item) for item in raw.split() if item]


def descendant_pids(pid: int) -> list[int]:
    found: list[int] = []
    stack = [pid]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        children = child_pids(current)
        found.extend(children)
        stack.extend(children)
    return found


def proc_comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def worker_pids(root_pid: int) -> list[int]:
    workers: list[int] = []
    for pid in descendant_pids(root_pid):
        comm = proc_comm(pid)
        if comm.startswith("VLLM::Worker"):
            workers.append(pid)
            continue
        joined = " ".join(proc_cmdline(pid))
        if "VLLM::Worker" in joined:
            workers.append(pid)
    return workers


def task_log_paths(item: Running) -> list[Path]:
    paths = [item.log_path]
    paths.extend(
        sorted(
            path
            for path in LOG_ROOT.glob("*.log")
            if is_task_log_name(item.task.task_id, path.name)
        )
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def running_engine_loaded(item: Running) -> bool:
    workers = worker_pids(item.pid)
    if any(pid_state(pid).startswith("D") for pid in workers):
        item.loaded = False
        return False
    if workers:
        item.loaded = True
        return True
    if engine_loaded_from_logs(task_log_paths(item)):
        item.loaded = True
        return True
    item.loaded = False
    return False


def loading_items(running: dict[int, Running]) -> list[Running]:
    return [item for item in running.values() if not running_engine_loaded(item)]


def export_cell_from_cmd(cmd: list[str]) -> tuple[str, str, int] | None:
    if not any(part.endswith("export_leftover_suppress_jobs.py") for part in cmd):
        return None
    try:
        model = cmd[cmd.index("--model-tag") + 1]
        dataset = cmd[cmd.index("--datasets") + 1]
        seed = int(cmd[cmd.index("--seed") + 1])
    except (ValueError, IndexError):
        return None
    return model, dataset, seed


def live_cpu_exports() -> set[tuple[str, str, int]]:
    found: set[tuple[str, str, int]] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if not pid_alive(pid):
            continue
        cell = export_cell_from_cmd(proc_cmdline(pid))
        if cell is not None:
            found.add(cell)
    return found


def start_cpu_export(model: str, dataset: str, seed: int) -> subprocess.Popen:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"export__{model}__{dataset}__s{seed}.log"
    with log_path.open("ab") as handle:
        handle.write(f"\n# {now()} start cpu export\n".encode())
    handle = log_path.open("ab")
    process = subprocess.Popen(
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
        env={
            **os.environ,
            "PLWS_ROOT": str(ROOT),
            "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(
                os.pathsep
            ),
        },
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return process


def kick_cpu_exports(
    models: tuple[str, ...],
    started: dict[tuple[str, str, int], subprocess.Popen],
) -> None:
    for key, proc in list(started.items()):
        if proc.poll() is None:
            continue
        started.pop(key)
        fields = {"model": key[0], "dataset": key[1], "seed": key[2], "pid": proc.pid}
        if proc.returncode:
            event("cpu_export_failed", code=proc.returncode, **fields)
        else:
            event("cpu_export_done", **fields)
    live = live_cpu_exports() | set(started)
    for model, dataset, seed in cells_needing_cpu_export(PATHS, models):
        key = (model, dataset, seed)
        if key in live:
            continue
        started[key] = start_cpu_export(model, dataset, seed)
        event(
            "cpu_export_started",
            model=model,
            dataset=dataset,
            seed=seed,
            pid=started[key].pid,
        )


def command_for(task: ContestTask) -> list[str]:
    if task.kind == "prereq":
        return ["bash", str(ROOT / "scripts" / "run_contest_prereq_cell.sh")]
    if task.kind == "plws":
        return ["bash", str(ROOT / "scripts" / "run_contest_plws_cell.sh")]
    raise ValueError(task.kind)


def env_for(task: ContestTask, gpus: tuple[str, ...]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PLWS_ROOT": str(ROOT),
            "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
                os.pathsep
            ),
            "GPU": ",".join(gpus),
            "CUDA_VISIBLE_DEVICES": ",".join(gpus),
            "MODEL": MODELS[task.model],
            "MODEL_TAG": task.model,
            "DATASET": task.dataset,
            "SEED": str(task.seed),
            "KIND": task.run_kind,
            "SHARD_ID": str(task.shard_id),
            "NUM_SHARDS": str(task.num_shards),
            "BATCH_SIZE": str(batch_size(task.model)),
            "VLLM_LENS_DISABLE": "1",
        }
    )
    return env


def adopt_running(tasks: list[ContestTask]) -> dict[int, Running]:
    wanted = {task.task_id: task for task in tasks}
    adopted: dict[int, Running] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if not pid_alive(pid):
            continue
        cmd = proc_cmdline(pid)
        env = proc_environ(pid)
        task: ContestTask | None = None
        if any(part.endswith("run_contest_prereq_cell.sh") for part in cmd):
            task = wanted.get(
                f"prereq__{env.get('MODEL_TAG', '')}__"
                f"{env.get('DATASET', '')}__s{env.get('SEED', '')}"
            )
        elif any(
            part.endswith("run_contest_plws_cell.sh")
            or part.endswith("run_matrix_plws_cell.sh")
            for part in cmd
        ) or (
            any(part.endswith("score_leftover_suppress.py") for part in cmd)
            and "/window_first/k_4/lexicon_core/" in " ".join(cmd)
        ):
            try:
                model = env.get("MODEL_TAG") or cmd[cmd.index("--model-tag") + 1]
                dataset = env.get("DATASET") or cmd[cmd.index("--dataset") + 1]
                seed = env.get("SEED") or cmd[cmd.index("--seed") + 1]
                shard_id = env.get("SHARD_ID") or (
                    cmd[cmd.index("--shard-id") + 1] if "--shard-id" in cmd else "0"
                )
                num_shards = env.get("NUM_SHARDS") or (
                    cmd[cmd.index("--num-shards") + 1] if "--num-shards" in cmd else "1"
                )
            except (ValueError, IndexError):
                continue
            shard = f"__s{shard_id}of{num_shards}" if int(num_shards) > 1 else ""
            task = wanted.get(f"plws__{model}__{dataset}__s{seed}{shard}")
        if task is None:
            continue
        gpus = tuple(
            item.strip()
            for item in env.get("CUDA_VISIBLE_DEVICES", env.get("GPU", "")).split(",")
            if item.strip()
        )
        if not gpus:
            continue
        existing = next(
            (item for item in adopted.values() if item.task.task_id == task.task_id),
            None,
        )
        if existing is not None:
            continue
        adopted[pid] = Running(
            task=task,
            pid=pid,
            gpus=gpus,
            adopted=True,
            started_at=now(),
            log_path=LOG_ROOT / f"{task.task_id}.adopted.log",
        )
    return adopted


def launch(task: ContestTask, assigned: tuple[str, ...], attempt: int) -> Running:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{task.task_id}.attempt_{attempt}.log"
    with log_path.open("ab") as handle:
        handle.write(f"\n# {now()} start gpus={','.join(assigned)}\n".encode())
    handle = log_path.open("ab")
    process = subprocess.Popen(
        command_for(task),
        cwd=ROOT,
        env=env_for(task, assigned),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return Running(
        task=task,
        pid=process.pid,
        gpus=assigned,
        adopted=False,
        started_at=now(),
        log_path=log_path,
    )


def write_status(
    pending: list[ContestTask],
    running: dict[int, Running],
    succeeded: list[str],
    failed: list[str],
    phase: str | None,
    *,
    waiting: str | None = None,
) -> None:
    rows = []
    for item in running.values():
        complete, reason = task_complete(PATHS, item.task)
        rows.append(
            {
                "task_id": item.task.task_id,
                "phase": item.task.phase,
                "gpus": list(item.gpus),
                "pid": item.pid,
                "adopted": item.adopted,
                "started_at": item.started_at,
                "loading": not running_engine_loaded(item),
                "progress": reason,
                "complete": complete,
            }
        )
    atomic_json(
        STATUS_PATH,
        {
            "updated_at": now(),
            "protocol_id": "puma-fullcot-32k-v2",
            "lexicon": "core",
            "lane": "contest",
            "phase": phase,
            "waiting": waiting,
            "counts": {
                "pending": len(pending),
                "running": len(running),
                "succeeded": len(succeeded),
                "failed": len(failed),
            },
            "running": rows,
            "pending": [task.task_id for task in pending],
            "succeeded": succeeded,
            "failed": failed,
        },
    )


def validate(gpus: tuple[str, ...]) -> None:
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU pool contains duplicates")
    unknown = [gpu for gpu in gpus if gpu not in set("01234567")]
    if unknown:
        raise ValueError(f"unknown GPU ids: {unknown}")
    if len(gpus) < 2:
        raise ValueError("contest lane needs at least one TP=2 pair")
    for model, path in MODELS.items():
        if not (Path(path) / "config.json").is_file():
            raise FileNotFoundError(f"model missing: {model} {path}")
    for script in (
        ROOT / "scripts" / "run_contest_prereq_cell.sh",
        ROOT / "scripts" / "run_contest_plws_cell.sh",
    ):
        if not script.is_file():
            raise FileNotFoundError(script)
    for dataset in DATASETS:
        data = ROOT.parent / "PUMA" / "data" / f"{dataset}_test.jsonl"
        if not data.is_file():
            raise FileNotFoundError(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=",".join(DEFAULT_GPUS))
    parser.add_argument("--poll", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated contest model tags.",
    )
    args = parser.parse_args()
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    validate(gpus)
    if args.models.strip():
        chosen_models = tuple(
            item.strip() for item in args.models.split(",") if item.strip()
        )
        unknown = [model for model in chosen_models if model not in MODELS]
        if unknown:
            raise ValueError(f"unknown models: {unknown}")
    else:
        chosen_models = QUEUE_MODELS
    tasks = build_tasks(PATHS, models=chosen_models)

    if not args.dry_run:
        for model in chosen_models:
            for dataset in DATASETS:
                for seed in (42, 0, 1, 123, 7):
                    if ensure_firstwin_jobs(PATHS, model, dataset, seed):
                        event(
                            "ensured_firstwin",
                            model=model,
                            dataset=dataset,
                            seed=seed,
                        )
                    if needs_cpu_export(PATHS, model, dataset, seed):
                        start_cpu_export(model, dataset, seed)
                        event(
                            "exported_jobs",
                            model=model,
                            dataset=dataset,
                            seed=seed,
                        )

    counts = summarize(tasks)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "gpus": list(gpus),
                    "models": list(chosen_models),
                    "seeds": [42, 0, 1, 123, 7],
                    "datasets": list(DATASETS),
                    "counts": counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        for task in tasks:
            complete, reason = task_complete(PATHS, task)
            if not complete:
                print(f"{task.phase}\t{task.task_id}\t{reason}")
        return 0

    running = adopt_running(tasks)
    cpu_exports: dict[tuple[str, str, int], subprocess.Popen] = {}
    claimed = {item.task.task_id for item in running.values()}
    pending: list[ContestTask] = []
    succeeded: list[str] = []
    failed: list[str] = []
    attempts: dict[str, int] = {}
    for task in tasks:
        if task.task_id in claimed:
            event("task_adopted", task_id=task.task_id)
            continue
        complete, reason = task_complete(PATHS, task)
        if complete:
            succeeded.append(task.task_id)
            event("task_reconciled", task_id=task.task_id, reason=reason)
        else:
            pending.append(task)

    event(
        "queue_started",
        gpus=list(gpus),
        counts=counts,
        pending=len(pending),
        adopted=len(running),
        succeeded=len(succeeded),
    )
    stopping = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        event("stop_requested", signal=signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while pending or running:
        try:
            finished = [pid for pid in running if not pid_alive(pid)]
            for pid in finished:
                item = running.pop(pid)
                complete, reason = task_complete(PATHS, item.task)
                if complete:
                    succeeded.append(item.task.task_id)
                    event(
                        "task_succeeded",
                        task_id=item.task.task_id,
                        gpus=list(item.gpus),
                        reason=reason,
                    )
                elif stopping:
                    failed.append(item.task.task_id)
                    event("task_failed", task_id=item.task.task_id, reason=reason)
                elif attempts.get(item.task.task_id, 1) < args.max_attempts:
                    pending.append(item.task)
                    event(
                        "task_retry",
                        task_id=item.task.task_id,
                        reason=reason,
                        attempt=attempts.get(item.task.task_id, 1),
                    )
                else:
                    failed.append(item.task.task_id)
                    event("task_failed", task_id=item.task.task_id, reason=reason)

            if stopping:
                phase = first_open_phase(
                    [*pending, *(item.task for item in running.values())],
                    {task_id: "done" for task_id in succeeded + failed},
                )
                write_status(pending, running, succeeded, failed, phase)
                return 0

            finished_ids = {task_id: "done" for task_id in succeeded + failed}
            phase = first_open_phase(
                [*pending, *(item.task for item in running.values())],
                finished_ids,
            )
            used = {gpu for item in running.values() for gpu in item.gpus}
            leftover = leftover_8b_gpus()
            free = idle_contest_gpus(gpus, used, leftover=leftover)
            still_loading = loading_items(running)
            kick_cpu_exports(chosen_models, cpu_exports)
            blocked = contest_blocked_phases(
                pending, {item.task.phase for item in running.values()}
            )
            candidates = fill_leftover_dispatch(
                pending, len(free), PATHS, blocked_phases=blocked
            )
            waiting = None
            need = max((task.gpu_count for task in pending), default=1)
            if leftover:
                waiting = "8B leftover holding " + ",".join(sorted(leftover, key=int))
            elif still_loading:
                waiting = "serial load: waiting for " + ",".join(
                    item.task.task_id for item in still_loading
                )
            elif pending and not candidates and len(free) < need:
                waiting = f"only {len(free)} idle GPU in pool {','.join(gpus)}"
            for fit in select_cold_starts(candidates, any_loading=bool(still_loading)):
                if len(free) < fit.gpu_count:
                    break
                pending.remove(fit)
                assigned = tuple(free[: fit.gpu_count])
                del free[: fit.gpu_count]
                attempts[fit.task_id] = attempts.get(fit.task_id, 0) + 1
                item = launch(fit, assigned, attempts[fit.task_id])
                running[item.pid] = item
                event(
                    "task_started",
                    task_id=fit.task_id,
                    phase=fit.phase,
                    gpus=list(assigned),
                    pid=item.pid,
                    attempt=attempts[fit.task_id],
                    serial_load=True,
                )

            write_status(
                pending, running, succeeded, failed, phase, waiting=waiting
            )
            if not (running or pending):
                break
            deadline = time.time() + max(1, args.poll)
            while time.time() < deadline:
                if any(not pid_alive(pid) for pid in running):
                    break
                time.sleep(1)

        except Exception as exc:
            event("queue_tick_error", error=repr(exc))
            time.sleep(max(1, args.poll))
    event("queue_finished", succeeded=len(succeeded), failed=len(failed))
    write_status(pending, running, succeeded, failed, None)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
