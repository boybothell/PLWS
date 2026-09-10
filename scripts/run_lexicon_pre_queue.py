#!/usr/bin/env python3
"""Dynamically assign lexicon pre-experiment cells to free GPUs."""

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

from plws.artifacts import done_uids, load_jsonl  # noqa: E402

STAGE = "pre"
LANE = ROOT / "scripts" / "run_lexicon_confirm_lane.sh"
JOBS_DIR = ROOT / "results" / "experiments" / "lexicon_ablation" / "jobs"
OUT_ROOT = ROOT / "results" / "experiments" / "lexicon_ablation" / STAGE
RUN_ROOT = OUT_ROOT / "queue"
LOG_ROOT = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.json"
EVENTS_PATH = RUN_ROOT / "events.jsonl"
DEFAULT_GPUS = ("0", "1", "2", "3", "4")

# Already-running unsharded workers stay one cell. Everything else is a
# 1-GPU shard so five cards drain the same pool and finish together.
KEEP_WHOLE = {
    ("r1_7b", "wait"),
    ("r1_7b", "core_plus_let_me"),
    ("qwen3_4b", "core"),
}
CELL_SPECS = (
    ("r1_7b", "wait"),
    ("r1_7b", "core_plus_let_me"),
    ("qwen3_4b", "core"),
    ("qwen3_4b", "wait"),
    ("qwen3_4b", "core_plus_let_me"),
    ("nemotron_8b", "core"),
    ("nemotron_8b", "wait"),
    ("nemotron_8b", "core_plus_let_me"),
    ("qwen3_8b", "core"),
    ("qwen3_8b", "wait"),
    ("qwen3_8b", "core_plus_let_me"),
    ("r1_14b", "wait"),
    ("r1_14b", "core_plus_let_me"),
)


@dataclass(frozen=True)
class Task:
    model: str
    lexicon: str
    shard_id: int = 0
    num_shards: int = 1

    @property
    def gpu_count(self) -> int:
        return 1

    @property
    def task_id(self) -> str:
        if self.num_shards <= 1:
            return f"{self.model}__{self.lexicon}"
        return f"{self.model}__{self.lexicon}__s{self.shard_id}of{self.num_shards}"

    @property
    def jobs_path(self) -> Path:
        return JOBS_DIR / f"{STAGE}_{self.model}.jsonl"

    @property
    def scores_path(self) -> Path:
        folder = OUT_ROOT / self.model / self.lexicon
        if self.num_shards <= 1:
            return folder / "scores.jsonl"
        return folder / f"scores_shard{self.shard_id}.jsonl"

    @property
    def score_dir(self) -> Path:
        return OUT_ROOT / self.model / self.lexicon


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    for model, lexicon in CELL_SPECS:
        shards = 1 if (model, lexicon) in KEEP_WHOLE else 2
        for shard_id in range(shards):
            tasks.append(Task(model, lexicon, shard_id, shards))
    return tasks


@dataclass
class Running:
    task: Task
    pid: int
    gpus: tuple[str, ...]
    adopted: bool
    started_at: str
    log_path: Path


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


def task_jobs(task: Task) -> list[str]:
    rows = [str(row["uid"]) for row in load_jsonl(task.jobs_path) if row.get("uid")]
    if task.num_shards <= 1:
        return rows
    return [
        uid
        for index, uid in enumerate(rows)
        if index % task.num_shards == task.shard_id
    ]


def task_complete(task: Task) -> tuple[bool, str, int, int]:
    jobs = set(task_jobs(task))
    if not jobs:
        return False, "missing jobs", 0, 0
    done = done_uids({task.scores_path, task.score_dir})
    missing = len(jobs - done)
    return missing == 0, f"done {len(jobs & done)}/{len(jobs)}", len(jobs), missing


def pid_state(pid: int) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return line.split()[1]
    except OSError:
        return ""
    return "?"


def pid_alive(pid: int) -> bool:
    # SIGSTOP'd parents cannot reap children. Finished workers then sit as
    # zombies; os.kill(pid, 0) still succeeds and the GPU looks occupied.
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


def adopt_running(tasks: list[Task]) -> dict[int, Running]:
    wanted = {task.task_id: task for task in tasks}
    adopted: dict[int, Running] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if not pid_alive(pid):
            continue
        cmd = proc_cmdline(pid)
        if not any(part.endswith("score_leftover_suppress.py") for part in cmd):
            continue
        try:
            model = cmd[cmd.index("--model-tag") + 1]
            lexicon = cmd[cmd.index("--lexicon") + 1]
            out = cmd[cmd.index("--out") + 1]
        except (ValueError, IndexError):
            continue
        if f"/lexicon_ablation/{STAGE}/" not in out.replace("\\", "/"):
            continue
        if "--shard-id" in cmd:
            shard_id = int(cmd[cmd.index("--shard-id") + 1])
            num_shards = int(cmd[cmd.index("--num-shards") + 1])
            task_id = f"{model}__{lexicon}__s{shard_id}of{num_shards}"
        else:
            task_id = f"{model}__{lexicon}"
        task = wanted.get(task_id)
        if task is None:
            continue
        env = proc_environ(int(proc.name))
        gpus = tuple(
            item.strip()
            for item in env.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item.strip()
        )
        if not gpus:
            continue
        adopted[int(proc.name)] = Running(
            task=task,
            pid=int(proc.name),
            gpus=gpus,
            adopted=True,
            started_at=now(),
            log_path=OUT_ROOT / task.model / task.lexicon / "logs" / f"{task.lexicon}.log",
        )
    return adopted


def write_status(
    pending: list[Task],
    running: dict[int, Running],
    succeeded: list[str],
    failed: list[str],
) -> None:
    rows = []
    for item in running.values():
        _, reason, total, missing = task_complete(item.task)
        rows.append(
            {
                "task_id": item.task.task_id,
                "gpus": list(item.gpus),
                "pid": item.pid,
                "adopted": item.adopted,
                "started_at": item.started_at,
                "progress": reason,
                "pending_uids": missing,
                "jobs": total,
            }
        )
    atomic_json(
        STATUS_PATH,
        {
            "updated_at": now(),
            "principle": "five_cards_drain_together",
            "stage": STAGE,
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


def launch(task: Task, assigned: tuple[str, ...], attempt: int) -> Running:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{task.task_id}.attempt_{attempt}.log"
    env = os.environ.copy()
    env.update(
        {
            "PLWS_ROOT": str(ROOT),
            "PLWS_LEX_STAGE": STAGE,
            "CUDA_VISIBLE_DEVICES": ",".join(assigned),
            "SHARD_ID": str(task.shard_id),
            "NUM_SHARDS": str(task.num_shards),
        }
    )
    with log_path.open("ab") as handle:
        handle.write(f"\n# {now()} start gpus={','.join(assigned)}\n".encode())
    handle = log_path.open("ab")
    process = subprocess.Popen(
        ["bash", str(LANE), ",".join(assigned), f"{task.model}:{task.lexicon}"],
        cwd=ROOT,
        env=env,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=",".join(DEFAULT_GPUS))
    parser.add_argument("--poll", type=int, default=10)
    args = parser.parse_args()
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    tasks = build_tasks()
    if max(task.gpu_count for task in tasks) > len(gpus):
        raise SystemExit("GPU pool is smaller than the largest cell")

    running = adopt_running(tasks)
    claimed = {item.task.task_id for item in running.values()}
    pending: list[Task] = []
    succeeded: list[str] = []
    failed: list[str] = []
    attempts: dict[str, int] = {}
    for task in tasks:
        if task.task_id in claimed:
            event(
                "task_adopted",
                task_id=task.task_id,
                pid=next(item.pid for item in running.values() if item.task.task_id == task.task_id),
                gpus=list(
                    next(item.gpus for item in running.values() if item.task.task_id == task.task_id)
                ),
            )
            continue
        complete, reason, _, _ = task_complete(task)
        if complete:
            succeeded.append(task.task_id)
            event("task_reconciled", task_id=task.task_id, reason=reason)
        else:
            pending.append(task)

    event(
        "queue_started",
        gpus=list(gpus),
        pending=[task.task_id for task in pending],
        adopted=[item.task.task_id for item in running.values()],
        succeeded=list(succeeded),
    )
    stopping = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        event("stop_requested", signal=signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while pending or running:
        finished = [pid for pid in running if not pid_alive(pid)]
        for pid in finished:
            item = running.pop(pid)
            complete, reason, _, missing = task_complete(item.task)
            if complete:
                succeeded.append(item.task.task_id)
                event("task_succeeded", task_id=item.task.task_id, gpus=list(item.gpus), reason=reason)
            elif stopping:
                failed.append(item.task.task_id)
                event("task_failed", task_id=item.task.task_id, reason=reason)
            else:
                pending.append(item.task)
                event(
                    "task_retry",
                    task_id=item.task.task_id,
                    reason=reason,
                    pending_uids=missing,
                )

        if stopping:
            write_status(pending, running, succeeded, failed)
            # Workers keep running; a later dispatcher start will adopt them.
            return 0

        used = {gpu for item in running.values() for gpu in item.gpus}
        free = [gpu for gpu in gpus if gpu not in used]
        while free:
            fit = next((task for task in pending if task.gpu_count <= len(free)), None)
            if fit is None:
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
                gpus=list(assigned),
                pid=item.pid,
                attempt=attempts[fit.task_id],
            )

        write_status(pending, running, succeeded, failed)
        if running or pending:
            time.sleep(max(2, args.poll))

    event("queue_finished", succeeded=len(succeeded), failed=len(failed))
    write_status(pending, running, succeeded, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
