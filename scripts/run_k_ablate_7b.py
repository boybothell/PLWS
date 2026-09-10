#!/usr/bin/env python3
"""7B 第一扇窗 k∈{2,3,5,6} 消融：复用密探，旁路 jobs，压 Wait。

新结果写统一 run 路径；旧 partial scores 只读并计入已完成 UID。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT_HINT = Path(os.environ.get("PLWS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT_HINT / "src"))

from plws.artifacts import atomic_write_jsonl, done_uids, load_jsonl  # noqa: E402
from plws.paths import PLWSPaths  # noqa: E402

PATHS = PLWSPaths.discover(__file__)
AE = PATHS.root

PY = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python")
VENV = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
KS = (2, 3, 5, 6)
SEEDS = (42, 0, 1, 123)
KINDS = ("high", "mix", "low")
FIVE_DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
FIVE = ",".join(FIVE_DATASETS)
GPUS = tuple(int(x) for x in os.environ.get("KABL_GPUS", "0,1,2,3,4,5").split(",") if x.strip())


def nvidia_lib_path() -> str:
    extra = ":".join(
        sorted(str(p) for p in (VENV / ".venv" / "lib").glob("**/nvidia/*/lib") if p.is_dir())
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


def jobs_path(dataset: str, seed: int, kind: str, k: int) -> Path:
    canonical = PATHS.jobs_path(
        "r1_7b", dataset, seed, kind, k=k, lexicon="core"
    )
    if canonical.is_file():
        return canonical
    legacy = PATHS.legacy_jobs_path(
        "r1_7b", seed, kind, stem=f"jobs_k{k}"
    )
    work = (
        PATHS.window_run_root(k=k, lexicon="core")
        / "work"
        / "legacy_jobs"
        / "r1_7b"
        / dataset
        / f"seed_{seed}"
        / f"{kind}.jsonl"
    )
    if legacy.is_file():
        atomic_write_jsonl(
            work,
            [row for row in load_jsonl(legacy) if row.get("dataset") == dataset],
        )
    return work


def n_jobs(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.open() if line.strip())


def export_all() -> None:
    missing = [
        PATHS.jobs_path("r1_7b", dataset, seed, kind, k=k, lexicon="core")
        for k in KS
        for seed in SEEDS
        for dataset in FIVE_DATASETS
        for kind in KINDS
        if not PATHS.jobs_path(
            "r1_7b", dataset, seed, kind, k=k, lexicon="core"
        ).is_file()
    ]
    if not missing:
        print("reuse 240 canonical per-cell job files; export skipped", flush=True)
        return
    print(f"canonical jobs missing={len(missing)}; rebuilding all k jobs", flush=True)
    for k in KS:
        print(f"export k={k}", flush=True)
        subprocess.run(
            [
                str(PY),
                str(AE / "scripts/export_leftover_suppress_jobs.py"),
                "--model-tag",
                "r1_7b",
                "--kinds",
                "all",
                "--seeds",
                "42,0,1,123",
                "--datasets",
                FIVE,
                "--k",
                str(k),
            ],
            cwd=str(AE),
            check=True,
        )


def build_tasks() -> list[dict]:
    tasks: list[dict] = []
    for k in KS:
        for seed in SEEDS:
            for dataset in FIVE_DATASETS:
                for kind in KINDS:
                    jobs = jobs_path(dataset, seed, kind, k)
                    n = n_jobs(jobs)
                    if not n:
                        print(f"skip empty k{k} {dataset} s{seed} {kind}", flush=True)
                        continue
                    already = done_uids(
                        PATHS.score_read_dirs(
                            "r1_7b",
                            dataset,
                            seed,
                            "suppress",
                            kind,
                            k=k,
                            lexicon="core",
                        )
                    )
                    uids = [json.loads(line)["uid"] for line in jobs.open() if line.strip()]
                    pending = sum(1 for uid in uids if uid not in already)
                    if pending <= 0:
                        print(f"skip done k{k} {dataset} s{seed} {kind} n={n}", flush=True)
                        continue
                    shards = 6 if n >= 200 else (2 if n >= 20 else 1)
                    for shard in range(shards):
                        tasks.append(
                            {
                                "k": k,
                                "dataset": dataset,
                                "seed": seed,
                                "kind": kind,
                                "jobs": jobs,
                                "out": PATHS.score_path(
                                    "r1_7b",
                                    dataset,
                                    seed,
                                    "suppress",
                                    kind,
                                    shard,
                                    k=k,
                                    lexicon="core",
                                ),
                                "n": n,
                                "pending": pending,
                                "shard": shard,
                                "nshard": shards,
                            }
                        )
                    print(
                        f"queue k{k} {dataset} s{seed} {kind} "
                        f"n={n} pending={pending} shards={shards}",
                        flush=True,
                    )
    return tasks


def run_one(gpu: int, task: dict) -> int:
    k, dataset, seed, kind = (
        task["k"],
        task["dataset"],
        task["seed"],
        task["kind"],
    )
    log = (
        PATHS.cell_dir("r1_7b", dataset, seed, k=k, lexicon="core")
        / "work"
        / "logs"
        / f"{kind}_sh{task['shard']}.log"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_LENS_DISABLE"] = "1"
    env["LD_LIBRARY_PATH"] = nvidia_lib_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"gpu{gpu} suppress k{k} {dataset} s{seed} {kind} "
        f"{task['shard']}/{task['nshard']} -> {log}",
        flush=True,
    )
    started = time.time()
    with log.open("a") as handle:
        proc = subprocess.run(
            [
                str(PY),
                str(AE / "scripts/score_leftover_suppress.py"),
                "--mode",
                "suppress",
                "--model-tag",
                "r1_7b",
                "--run-kind",
                kind,
                "--seed",
                str(seed),
                "--dataset",
                dataset,
                "--lexicon",
                "core",
                "--k",
                str(k),
                "--shard-id",
                str(task["shard"]),
                "--num-shards",
                str(task["nshard"]),
                "--max-context",
                "32768",
                "--think-tokens",
                "0",
                "--batch-size",
                "16",
                "--jobs",
                str(task["jobs"]),
                "--out",
                str(task["out"]),
            ],
            cwd=str(AE),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    print(
        f"gpu{gpu} k{k} {dataset} s{seed} {kind} "
        f"sh{task['shard']} exit={proc.returncode} "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    return proc.returncode


def main() -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    os.environ["VLLM_LENS_DISABLE"] = "1"
    print(f"k-ablate 7B gpus {list(GPUS)} ks={list(KS)}", flush=True)
    export_all()
    tasks = build_tasks()
    if not tasks:
        print("nothing pending", flush=True)
        return
    free = list(GPUS)
    lock = threading.Lock()
    fail = 0
    done = 0
    inflight = 0

    def spawn(gpu: int, task: dict) -> None:
        nonlocal fail, done, inflight

        def _run() -> None:
            nonlocal fail, done, inflight
            try:
                code = run_one(gpu, task)
                if code != 0:
                    with lock:
                        fail += 1
            finally:
                with lock:
                    free.append(gpu)
                    inflight -= 1
                    done += 1
                    print(
                        f"queue done={done} inflight={inflight} free={sorted(free)} fail={fail}",
                        flush=True,
                    )

        threading.Thread(target=_run, daemon=True).start()

    i = 0
    while i < len(tasks) or inflight:
        with lock:
            while i < len(tasks) and free:
                gpu = free.pop(0)
                inflight += 1
                spawn(gpu, tasks[i])
                i += 1
        time.sleep(2)
    print(f"k-ablate 7B done fail={fail}/{len(tasks)}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
