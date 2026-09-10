#!/usr/bin/env python3
"""0–6 空卡领取：7B/8B/14B 开局压核三词。已完成的 shard 跳过。不抢占用中的卡。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
from score_leftover_jump import done_uids  # noqa: E402

PY = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python")
ROOT = AE / "results/fromstart_core"
LOG = ROOT / "logs"
GPUS = tuple(int(x) for x in os.environ.get("FROMSTART_GPUS", "0,1,2,3").split(",") if x.strip())
MODELS = ("r1_7b", "nemotron_8b", "r1_14b")
SEEDS = (42, 0, 1, 123)
BUSY_MIB = 2048
VENV = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")


def nvidia_lib_path() -> str:
    extra = ":".join(
        sorted(str(path) for path in (VENV / ".venv" / "lib").glob("**/nvidia/*/lib") if path.is_dir())
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


def n_jobs(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def build_tasks() -> list[dict]:
    tasks: list[dict] = []
    for tag in MODELS:
        for seed in SEEDS:
            jobs = ROOT / f"{tag}_s{seed}" / "jobs.jsonl"
            n = n_jobs(jobs)
            if not n:
                continue
            shards = 6 if n >= 200 else 2
            tasks.append({"tag": tag, "seed": seed, "jobs": jobs, "nshard": shards, "n": n})
    out: list[dict] = []
    for spec in tasks:
        already: set[str] = set()
        score_dir = ROOT / f"{spec['tag']}_s{spec['seed']}_suppress"
        if score_dir.is_dir():
            for path in score_dir.glob("scores_shard*.jsonl"):
                already |= done_uids(path)
        jobs = []
        if spec["jobs"].is_file():
            jobs = [json.loads(line) for line in spec["jobs"].read_text().splitlines() if line.strip()]
        for shard in range(spec["nshard"]):
            shard_jobs = [job for i, job in enumerate(jobs) if i % spec["nshard"] == shard]
            pending = sum(1 for job in shard_jobs if job["uid"] not in already)
            if pending <= 0:
                print(
                    f"skip done {spec['tag']} s{spec['seed']} shard {shard}/{spec['nshard']}",
                    flush=True,
                )
                continue
            out.append({**spec, "shard": shard, "pending": pending})
    return out


def run_one(gpu: int, task: dict) -> int:
    tag = task["tag"]
    seed = task["seed"]
    shard = task["shard"]
    nshard = task["nshard"]
    log = LOG / f"{tag}_s{seed}_sh{shard}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_LENS_DISABLE"] = "1"
    env["LD_LIBRARY_PATH"] = nvidia_lib_path()
    cmd = [
        str(PY),
        str(AE / "scripts/score_leftover_suppress.py"),
        "--mode",
        "suppress",
        "--model-tag",
        tag,
        "--lexicon",
        "core",
        "--seed",
        str(seed),
        "--shard-id",
        str(shard),
        "--num-shards",
        str(nshard),
        "--max-context",
        "32768",
        "--think-tokens",
        "0",
        "--batch-size",
        "16",
        "--jobs",
        str(task["jobs"]),
        "--out-root",
        str(ROOT),
    ]
    print(
        f"gpu{gpu} start {tag} s{seed} shard {shard}/{nshard} n={task['n']} -> {log}",
        flush=True,
    )
    started = time.time()
    with log.open("a") as handle:
        handle.write(f"\n--- resume {time.strftime('%F %T')} gpu={gpu} pending={task.get('pending')} ---\n")
        proc = subprocess.run(cmd, cwd=str(AE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print(
        f"gpu{gpu} done {tag} s{seed} shard {shard}/{nshard} "
        f"exit={proc.returncode} {time.time() - started:.0f}s",
        flush=True,
    )
    return proc.returncode


def gpu_mem() -> dict[int, int]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return {}
    used: dict[int, int] = {}
    for line in raw.splitlines():
        if "," not in line:
            continue
        idx, mem = [x.strip() for x in line.split(",", 1)]
        used[int(idx)] = int(float(mem))
    return used


def main() -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    LOG.mkdir(parents=True, exist_ok=True)
    print(f"ld_library_path set cu13={('cu13' in os.environ['LD_LIBRARY_PATH'])}", flush=True)
    print(f"fromstart gpus {list(GPUS)}", flush=True)
    print("export jobs", flush=True)
    subprocess.run([str(PY), str(AE / "scripts/export_fromstart_core_jobs.py")], cwd=str(AE), check=True)
    tasks = build_tasks()
    if not tasks:
        print("fromstart nothing pending", flush=True)
        return
    pending = list(tasks)
    claimed: set[int] = set()
    inflight = 0
    stats = {"done": 0, "fail": 0, "total": len(tasks)}
    lock = threading.Lock()
    print(f"tasks {len(tasks)} pending shards", flush=True)

    def spawn(gpu: int, task: dict) -> None:
        nonlocal inflight

        def _run() -> None:
            nonlocal inflight
            try:
                code = run_one(gpu, task)
                with lock:
                    stats["done"] += 1
                    if code != 0:
                        stats["fail"] += 1
                    print(f"queue {stats['done']}/{stats['total']} fail={stats['fail']}", flush=True)
            finally:
                with lock:
                    inflight -= 1
                    claimed.discard(gpu)

        with lock:
            inflight += 1
        threading.Thread(target=_run, daemon=True).start()

    while True:
        with lock:
            n_pend = len(pending)
            n_fly = inflight
        if n_pend == 0 and n_fly == 0:
            break
        used = gpu_mem()
        if n_pend:
            for gpu in GPUS:
                with lock:
                    if gpu in claimed:
                        continue
                if used.get(gpu, 0) >= BUSY_MIB:
                    continue
                with lock:
                    if not pending:
                        break
                    task = pending.pop(0)
                    claimed.add(gpu)
                print(f"claim gpu{gpu} mem={used.get(gpu, 0)}MiB", flush=True)
                spawn(gpu, task)
        time.sleep(4)
    print(f"fromstart dyn6 done fail={stats['fail']}/{stats['total']}", flush=True)
    if stats["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
