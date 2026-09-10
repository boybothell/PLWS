#!/usr/bin/env python3
"""Two-GPU dynamic queue for Qwen3-8B AIME24/25 four-seed cells."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from plws.artifacts import done_uids, load_jsonl  # noqa: E402

PY = "/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python"
JOBS_DIR = ROOT / "results" / "experiments" / "lexicon_ablation" / "jobs"
OUT_ROOT = ROOT / "results" / "experiments" / "lexicon_ablation" / "aime4s" / "qwen3_8b"
RUN_ROOT = OUT_ROOT / "queue"
LOG_ROOT = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.json"
EVENTS_PATH = RUN_ROOT / "events.jsonl"
SEEDS = (0, 1, 123)
CONFIGS = ("core", "wait", "core_plus_let_me")
MODEL = "qwen3_8b"


@dataclass(frozen=True)
class Task:
    seed: int
    lexicon: str

    @property
    def task_id(self) -> str:
        return f"s{self.seed}__{self.lexicon}"

    @property
    def jobs_path(self) -> Path:
        return JOBS_DIR / f"aime4s_{MODEL}_s{self.seed}.jsonl"

    @property
    def scores_path(self) -> Path:
        return OUT_ROOT / f"s{self.seed}" / self.lexicon / "scores.jsonl"


@dataclass
class Running:
    task: Task
    pid: int
    gpu: str
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


def task_complete(task: Task) -> tuple[bool, str, int, int]:
    jobs = [str(row["uid"]) for row in load_jsonl(task.jobs_path) if row.get("uid")]
    if not jobs:
        return False, "missing jobs", 0, 0
    done = done_uids({task.scores_path, task.scores_path.parent})
    have = set(jobs) & done
    missing = len(set(jobs) - done)
    return missing == 0, f"done {len(have)}/{len(jobs)}", len(jobs), missing


def gpu_mem_mib(gpu: str) -> int:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return int(raw.splitlines()[0])
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 10**9


def ld_library_path() -> str:
    script = """
from pathlib import Path
root = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
print(":".join(sorted({
    str(path)
    for path in (root / ".venv" / "lib").glob("**/nvidia/*/lib")
    if path.is_dir()
})))
"""
    extra = subprocess.check_output([PY, "-c", script], text=True).strip()
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if current else extra


def launch(task: Task, gpu: str, attempt: int) -> Running:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = task.scores_path.parent
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.lexicon}.log"
    queue_log = LOG_ROOT / f"{task.task_id}.attempt_{attempt}.log"
    env = os.environ.copy()
    env.update(
        {
            "PLWS_ROOT": str(ROOT),
            "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep),
            "CUDA_VISIBLE_DEVICES": gpu,
            "VLLM_LENS_DISABLE": "1",
            "LD_LIBRARY_PATH": ld_library_path(),
        }
    )
    cmd = [
        PY,
        str(ROOT / "scripts" / "score_leftover_suppress.py"),
        "--mode",
        "suppress",
        "--model-tag",
        MODEL,
        "--lexicon",
        task.lexicon,
        "--seed",
        str(task.seed),
        "--run-kind",
        "pilot",
        "--max-context",
        "37888",
        "--think-tokens",
        "0",
        "--answer-tokens",
        "2048",
        "--batch-size",
        "64",
        "--sampling-seed",
        "20260904",
        "--isolated-output",
        "--jobs",
        str(task.jobs_path),
        "--out",
        str(task.scores_path),
    ]
    with queue_log.open("ab") as handle:
        handle.write(f"\n# {now()} start gpu={gpu}\n".encode())
    worker_log = log_path.open("ab")
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    worker_log.close()
    return Running(task=task, pid=process.pid, gpu=gpu, started_at=now(), log_path=log_path)


def write_status(pending: list[Task], running: dict[int, Running], succeeded: list[str], failed: list[str]) -> None:
    rows = []
    for item in running.values():
        _, reason, total, missing = task_complete(item.task)
        rows.append(
            {
                "task_id": item.task.task_id,
                "gpu": item.gpu,
                "pid": item.pid,
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
            "model": MODEL,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2")
    parser.add_argument("--claim-idle", default="0,1,3,4")
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--poll", type=int, default=10)
    args = parser.parse_args()
    pinned = [item.strip() for item in args.gpus.split(",") if item.strip()]
    claimable = [item.strip() for item in args.claim_idle.split(",") if item.strip()]
    pool = list(dict.fromkeys(pinned + claimable))
    tasks = [Task(seed, lexicon) for seed in SEEDS for lexicon in CONFIGS]
    pending: list[Task] = []
    succeeded: list[str] = []
    failed: list[str] = []
    running: dict[int, Running] = {}
    attempts: dict[str, int] = {}
    for task in tasks:
        complete, reason, _, _ = task_complete(task)
        if complete:
            succeeded.append(task.task_id)
            event("task_reconciled", task_id=task.task_id, reason=reason)
        else:
            pending.append(task)
    event("queue_started", gpus=pinned, claim_idle=claimable, pending=[task.task_id for task in pending])
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
                event("task_succeeded", task_id=item.task.task_id, gpu=item.gpu, reason=reason)
            elif stopping:
                failed.append(item.task.task_id)
                event("task_failed", task_id=item.task.task_id, reason=reason)
            else:
                pending.append(item.task)
                event("task_retry", task_id=item.task.task_id, reason=reason, pending_uids=missing)

        if stopping:
            write_status(pending, running, succeeded, failed)
            return 0

        used = {item.gpu for item in running.values()}
        idle = []
        for gpu in pool:
            if gpu in used:
                continue
            if gpu in pinned or gpu_mem_mib(gpu) < 200:
                idle.append(gpu)
        while pending and idle and len(running) < args.max_gpus:
            gpu = idle.pop(0)
            task = pending.pop(0)
            attempts[task.task_id] = attempts.get(task.task_id, 0) + 1
            item = launch(task, gpu, attempts[task.task_id])
            running[item.pid] = item
            event(
                "task_started",
                task_id=task.task_id,
                gpu=gpu,
                pid=item.pid,
                attempt=attempts[task.task_id],
            )

        write_status(pending, running, succeeded, failed)
        if running or pending:
            time.sleep(max(2, args.poll))

    event("queue_finished", succeeded=len(succeeded), failed=len(failed))
    write_status(pending, running, succeeded, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
