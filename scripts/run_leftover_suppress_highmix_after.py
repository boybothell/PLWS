#!/usr/bin/env python3
"""低把握压 Wait 跑完后，同一条卡队列接着跑高把握、混合。不打断正在跑的 low。"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
PY = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python")
LOG = AE / "results/leftover_suppress_toend/logs_five"
OUT = AE / "results/leftover_suppress_toend"
JOBS = AE / "results/leftover_jump"

LANES = (
    {
        "name": "gpu0",
        "gpus": "0",
        "wait": ("r1_7b", "nemotron_8b"),
        "run": ("r1_7b", "nemotron_8b"),
    },
    {
        "name": "gpu1",
        "gpus": "1",
        "wait": ("r1_14b",),
        "run": ("r1_14b",),
    },
    {
        "name": "gpu23",
        "gpus": "2,3",
        "wait": ("r1_32b", "qwen3_30b_a3b"),
        "run": ("r1_32b", "qwen3_30b_a3b"),
    },
)
KINDS = ("high", "mix")


def low_log(tag: str) -> Path:
    return LOG / f"{tag}_suppress.log"


def low_done(tag: str) -> bool:
    path = low_log(tag)
    if not path.is_file():
        return False
    text = path.read_text(errors="replace")
    return "done mode=suppress" in text


def run_one(tag: str, kind: str, gpus: str) -> None:
    jobs = JOBS / f"{tag}_s42" / f"jobs_{kind}.jsonl"
    log = LOG / f"{tag}_{kind}_suppress.log"
    if not jobs.is_file():
        print(f"skip {tag} {kind}: no jobs file", flush=True)
        return
    n = sum(1 for line in jobs.read_text().splitlines() if line.strip())
    if n == 0:
        print(f"skip {tag} {kind}: 0 jobs", flush=True)
        return
    print(f"launch {tag} {kind} n={n} gpu={gpus} -> {log}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["VLLM_LENS_DISABLE"] = "1"
    with log.open("w") as handle:
        proc = subprocess.run(
            [
                str(PY),
                str(AE / "scripts/score_leftover_suppress.py"),
                "--mode",
                "suppress",
                "--model-tag",
                tag,
                "--run-kind",
                kind,
                "--shard-id",
                "0",
                "--num-shards",
                "1",
                "--max-context",
                "32768",
                "--think-tokens",
                "0",
                "--batch-size",
                "0",
                "--jobs",
                str(jobs),
                "--out-root",
                str(OUT),
            ],
            cwd=str(AE),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"done {tag} {kind} exit={proc.returncode}", flush=True)


def lane_worker(lane: dict) -> None:
    name = lane["name"]
    print(f"{name} waiting low {lane['wait']}", flush=True)
    while True:
        if all(low_done(tag) for tag in lane["wait"]):
            break
        time.sleep(60)
    print(f"{name} low done, start high/mix", flush=True)
    for tag in lane["run"]:
        for kind in KINDS:
            run_one(tag, kind, lane["gpus"])
    print(f"{name} high/mix queue done", flush=True)


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    # run lanes in-process via children so one crash does not kill others
    import multiprocessing as mp

    workers = []
    for lane in LANES:
        p = mp.Process(target=lane_worker, args=(lane,), name=lane["name"])
        p.start()
        workers.append(p)
        print(f"started lane {lane['name']} pid={p.pid}", flush=True)
    for p in workers:
        p.join()
    print("all high/mix lanes done", flush=True)


if __name__ == "__main__":
    main()
