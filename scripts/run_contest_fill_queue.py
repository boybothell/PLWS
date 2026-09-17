#!/usr/bin/env python3
"""Resource-aware PUMA/PLWS fill queue for selected models and datasets."""

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

from plws.runtime import dataset_path, load_dotenv  # noqa: E402

load_dotenv(ROOT)
from plws.contest import (  # noqa: E402
    FILL_DATASETS,
    FILL_MODELS,
    MODELS,
    SEEDS,
    ContestTask,
    batch_size,
    build_fill_tasks,
    cells_needing_cpu_export,
    contest_fill_queue_alive,
    engine_loaded_from_logs,
    engine_ready_for_next_cold_start,
    fill_dispatch,
    fill_fullcot_complete,
    fill_task_complete,
    filter_fill_tasks,
    fullcot_only_tasks,
    first_open_phase,
    idle_contest_gpus,
    is_task_log_name,
    leftover_8b_gpus,
    log_loaded_after_latest_start,
    select_cold_starts,
    summarize,
    vllm_workers_loading,
)
from plws.matrix import (  # noqa: E402
    ensure_firstwin_jobs,
    needs_cpu_export,
)
from plws.paths import PLWSPaths  # noqa: E402

PY = os.environ.get("PLWS_PY", sys.executable)
RUN_ROOT = Path(
    os.environ.get("CONTEST_FILL_RUN_ROOT", ROOT / "results" / "runs" / "contest_fill")
)
LOG_ROOT = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.json"
EVENTS_PATH = RUN_ROOT / "events.jsonl"
DEFAULT_GPUS = ("0", "1", "2", "3")
PATHS = PLWSPaths(ROOT)


def _prereq_or_plws_done(paths: PLWSPaths, task: ContestTask) -> tuple[bool, str]:
    return fill_task_complete(paths, task)


def _fullcot_done(paths: PLWSPaths, task: ContestTask) -> tuple[bool, str]:
    return fill_fullcot_complete(paths, task.model, task.dataset, task.seed)


_TASK_IS_DONE = _prereq_or_plws_done


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
        if comm.startswith("VLLM::Worker") or comm.startswith("VLLM::EngineCor"):
            workers.append(pid)
            continue
        joined = " ".join(proc_cmdline(pid))
        if "VLLM::Worker" in joined or "VLLM::EngineCore" in joined:
            workers.append(pid)
    return workers


def our_worker_pids(running: dict[int, Running]) -> set[int]:
    held: set[int] = set()
    for item in running.values():
        held.add(item.pid)
        held.update(descendant_pids(item.pid))
        held.update(worker_pids(item.pid))
    return held


def latest_task_log(task_id: str) -> Path:
    attempts = sorted(
        LOG_ROOT.glob(f"{task_id}.attempt_*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    if attempts:
        return attempts[-1]
    return LOG_ROOT / f"{task_id}.adopted.log"


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


def current_start_log_loaded(item: Running) -> bool:
    path = item.log_path
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return log_loaded_after_latest_start(text)


def running_engine_loaded(item: Running) -> bool:
    workers = worker_pids(item.pid)
    ready = engine_ready_for_next_cold_start(
        workers_in_d=any(pid_state(pid).startswith("D") for pid in workers),
        log_loaded=current_start_log_loaded(item),
    )
    item.loaded = ready
    return ready


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
            "CONTEST_RUN_ROOT": str(RUN_ROOT),
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
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
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
    for model, dataset, seed in cells_needing_cpu_export(
        PATHS, models, datasets=datasets, seeds=seeds
    ):
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
            "CONTEST_RUN_ROOT": str(RUN_ROOT),
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
            if env.get("CONTEST_RUN_ROOT") != str(RUN_ROOT):
                continue
            task = wanted.get(
                f"prereq__{env.get('MODEL_TAG', '')}__"
                f"{env.get('DATASET', '')}__s{env.get('SEED', '')}"
            )
        elif any(part.endswith("run_contest_plws_cell.sh") for part in cmd):
            if env.get("CONTEST_RUN_ROOT") != str(RUN_ROOT):
                continue
            task = wanted.get(
                f"plws__{env.get('MODEL_TAG', '')}__"
                f"{env.get('DATASET', '')}__s{env.get('SEED', '')}"
            )
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
            log_path=latest_task_log(task.task_id),
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


def task_is_done(task: ContestTask) -> tuple[bool, str]:
    return _TASK_IS_DONE(PATHS, task)


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
        complete, reason = task_is_done(item.task)
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
            "lane": "contest_fill",
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


def validate(
    gpus: tuple[str, ...],
    models: tuple[str, ...],
    datasets: tuple[str, ...],
) -> None:
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU pool contains duplicates")
    if not gpus:
        raise ValueError("fill queue needs a GPU pool")
    if any(not gpu.isdigit() for gpu in gpus):
        raise ValueError(f"GPU IDs must be non-negative integers: {gpus}")
    for model in models:
        path = Path(MODELS[model])
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"model missing: {model} {path}")
    for script in (
        ROOT / "scripts" / "run_contest_prereq_cell.sh",
        ROOT / "scripts" / "run_contest_plws_cell.sh",
    ):
        if not script.is_file():
            raise FileNotFoundError(script)
    puma_root = Path(os.environ.get("PUMA_ROOT", ROOT.parent / "PUMA"))
    for dataset in datasets:
        data = dataset_path(dataset, puma_root)
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
        help="Comma-separated fill model tags.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(FILL_DATASETS),
        help="Comma-separated PUMA dataset slugs.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="Comma-separated seeds. Official new-run first wave is 42,0,1.",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated task ids to keep after planning.",
    )
    parser.add_argument(
        "--allow-parallel",
        action="store_true",
        help="Allow a second fill lane while another contest-fill queue is live.",
    )
    parser.add_argument(
        "--fullcot-only",
        action="store_true",
        help="Sample Full-CoT only, skip PUMA/dense/PLWS, then exit.",
    )
    args = parser.parse_args()
    global _TASK_IS_DONE
    if args.fullcot_only:
        os.environ["FULLCOT_ONLY"] = "1"
        _TASK_IS_DONE = _fullcot_done
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if args.models.strip():
        chosen_models = tuple(
            item.strip() for item in args.models.split(",") if item.strip()
        )
        unknown = [model for model in chosen_models if model not in FILL_MODELS]
        if unknown:
            raise ValueError(f"unknown fill models: {unknown}")
    else:
        chosen_models = FILL_MODELS
    chosen_datasets = tuple(
        item.strip() for item in args.datasets.split(",") if item.strip()
    )
    if not chosen_datasets:
        raise ValueError("fill queue needs at least one dataset")
    try:
        chosen_seeds = tuple(
            int(item.strip()) for item in args.seeds.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError(f"seeds must be integers: {args.seeds}") from exc
    if not chosen_seeds:
        raise ValueError("fill queue needs at least one seed")
    validate(gpus, chosen_models, chosen_datasets)
    tasks = build_fill_tasks(
        PATHS,
        models=chosen_models,
        datasets=chosen_datasets,
        seeds=chosen_seeds,
    )
    if args.task_ids.strip():
        tasks = filter_fill_tasks(
            tasks,
            (item.strip() for item in args.task_ids.split(",") if item.strip()),
        )
    if args.fullcot_only:
        tasks = fullcot_only_tasks(tasks)

    if not args.dry_run:
        if not args.allow_parallel and contest_fill_queue_alive(
            exclude_pid=os.getpid()
        ):
            raise RuntimeError("another contest-fill queue is already running")
        if not args.fullcot_only:
            for model in chosen_models:
                for dataset in chosen_datasets:
                    for seed in chosen_seeds:
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
        pending_rows = []
        for task in tasks:
            complete, reason = task_is_done(task)
            if not complete:
                pending_rows.append(f"{task.phase}\t{task.task_id}\t{reason}")
        print(
            json.dumps(
                {
                    "gpus": list(gpus),
                    "models": list(chosen_models),
                    "seeds": list(chosen_seeds),
                    "datasets": list(chosen_datasets),
                    "counts": counts,
                    "pending": len(pending_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        for row in pending_rows:
            print(row)
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
        complete, reason = task_is_done(task)
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
                complete, reason = task_is_done(item.task)
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
            foreign_loading = vllm_workers_loading(our_worker_pids(running))
            kick_cpu_exports(
                chosen_models, chosen_datasets, chosen_seeds, cpu_exports
            )
            running_gpu_counts = [item.task.gpu_count for item in running.values()]
            candidates = fill_dispatch(
                pending,
                len(free),
                PATHS,
                running_gpu_counts=running_gpu_counts,
                pool_size=len(gpus),
            )
            waiting = None
            need = max((task.gpu_count for task in pending), default=1)
            dual_pending = any(task.gpu_count >= 2 for task in pending)
            dual_running = any(count >= 2 for count in running_gpu_counts)
            if leftover:
                waiting = "8B leftover holding " + ",".join(sorted(leftover, key=int))
            elif still_loading or foreign_loading:
                names = [item.task.task_id for item in still_loading]
                if foreign_loading:
                    names.append("foreign-vllm")
                waiting = "serial load: waiting for " + ",".join(names)
            elif (
                dual_pending
                and not dual_running
                and pending
                and not candidates
                and 0 < len(free) < 2
            ):
                waiting = (
                    f"holding {len(free)} GPU in pool {','.join(gpus)} "
                    "to pair TP=2 next to the 1-GPU lane"
                )
            elif pending and not candidates and len(free) < need:
                waiting = f"only {len(free)} idle GPU in pool {','.join(gpus)}"
            any_loading = bool(still_loading) or foreign_loading
            for fit in select_cold_starts(candidates, any_loading=any_loading):
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
