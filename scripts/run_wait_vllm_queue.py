#!/usr/bin/env python3
"""Dynamic Wait-only queue (cards 0-5). Never touches 6/7.

This pass re-scores Wait with the official PUMA header into dense_puma_wait.
1-GPU jobs: 7B / Nemotron-8B / 14B / Qwen3 leftovers. TP=2: 32B / 30B AIME.
No new Qwen3 sample/PUMA cells here — those stay on the live cell if any.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from threading import Condition
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from dump_dense_candidates import trial_path  # noqa: E402
from score_wait_vllm import wait_done_keys, wait_out_path  # noqa: E402

HF_PY = AE / ".venv/bin/python"
VLLM_PY = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python")
LOG_DIR = AE / "results/_logs"
STATUS = LOG_DIR / os.environ.get("WAIT_STATUS", "wait_vllm_status.json")
MAIN_LOG = LOG_DIR / os.environ.get("WAIT_MAIN_LOG", "wait_vllm_queue.log")

MODELS = {
    "r1_7b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"),
    "nemotron_8b": Path("/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"),
    "r1_14b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"),
    "r1_32b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-32B"),
    "qwen3_30b_a3b": Path("/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507"),
    "qwen3_4b": Path("/mnt/d/lsj/models/Qwen3-4B"),
    "qwen3_8b": Path("/mnt/d/lsj/models/Qwen3-8B"),
}
BIG = ("math-500", "olympiadbench", "gpqa-diamond")
AIME = ("aime24", "aime25")
SEEDS = (42, 0, 1, 123)
N_SHARDS_1 = 4
N_SHARDS_2 = 1
MAX_TP2 = 1

SKIP_WAIT: set[tuple[str, str, int | None]] = set()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_LOG.open("a") as handle:
        handle.write(line + "\n")


def write_status(payload: dict[str, Any]) -> None:
    STATUS.write_text(json.dumps({"ts": time.time(), **payload}, indent=2, ensure_ascii=False))


def cand_path(tag: str, dataset: str, seed: int | None) -> Path:
    root = AE / "results/confcal_judge/v2/dense_candidates"
    if tag == "r1_7b" and dataset in BIG:
        return root / f"{dataset}.jsonl"
    name = f"{dataset}.jsonl" if seed is None else f"{dataset}_s{seed}.jsonl"
    return root / tag / name


def dump_one(tag: str, dataset: str, seed: int | None) -> Path:
    out = cand_path(tag, dataset, seed)
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        trial_path(tag, dataset, seed if dataset.startswith("aime") else None)
    except FileNotFoundError:
        log(f"dump skip missing trial {tag} {dataset} seed={seed}")
        return out
    cmd = [str(HF_PY), "scripts/dump_dense_candidates.py", "--dataset", dataset, "--model-tag", tag]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    log(f"dump {' '.join(cmd[1:])}")
    try:
        subprocess.run(cmd, cwd=AE, check=True)
    except subprocess.CalledProcessError as exc:
        log(f"dump fail {tag} {dataset} seed={seed} code={exc.returncode}")
    return out


def wait_pending(candidates: Path, out: Path, shard_id: int, num_shards: int) -> int:
    if not candidates.exists():
        return -1
    rows = [json.loads(line) for line in candidates.read_text().splitlines() if line.strip()]
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for index, qi in enumerate(qis) if index % num_shards == shard_id}
    done = wait_done_keys(out)
    return sum(
        1
        for row in rows
        if int(row["question_idx"]) in keep
        and (int(row["question_idx"]), int(row["decision_step"]), str(row.get("answer") or "")) not in done
    )


def cell_done(tag: str, dataset: str, seed: int) -> bool:
    if dataset.startswith("aime"):
        wait = wait_out_path(tag, dataset, seed, 0)
        trial = AE / f"results/dense_G_{tag}/{dataset}/seed_{seed}/dense_puma/trial_answers.json"
    else:
        wait = wait_out_path(tag, dataset, None, 0)
        trial = AE / f"results/dense_G_{tag}/{dataset}/dense_puma/trial_answers.json"
    if seed != 42:
        puma = AE / f"results/puma_offline_{tag}_s{seed}/{dataset}/statistics.json"
    else:
        puma = AE / f"results/puma_offline_{tag}/{dataset}/statistics.json"
    if not (trial.exists() and puma.exists()):
        return False
    cands = cand_path(tag, dataset, seed if dataset.startswith("aime") else None)
    if not cands.exists():
        return False
    return wait_pending(cands, wait, 0, 1) == 0


def make_wait_job(tag: str, dataset: str, seed: int | None, shard: int, n_shards: int, tp: int) -> dict[str, Any]:
    return {
        "kind": "wait",
        "name": f"wait:{tag}:{dataset}:s{seed if seed is not None else '-'}:shard{shard}",
        "tag": tag,
        "dataset": dataset,
        "seed": seed,
        "shard": shard,
        "n_shards": n_shards,
        "tp": tp,
        "candidates": cand_path(tag, dataset, seed),
        "out": wait_out_path(tag, dataset, seed, shard),
    }


def make_cell_job(tag: str, dataset: str, seed: int) -> dict[str, Any]:
    return {
        "kind": "cell",
        "name": f"cell:{tag}:{dataset}:s{seed}",
        "tag": tag,
        "dataset": dataset,
        "seed": seed,
        "tp": 1,
    }


def job_pending(job: dict[str, Any]) -> int:
    if job["kind"] == "cell":
        return 0 if cell_done(job["tag"], job["dataset"], job["seed"]) else 1
    return wait_pending(job["candidates"], job["out"], job["shard"], job["n_shards"])


def command_for(job: dict[str, Any]) -> list[str]:
    if job["kind"] == "cell":
        return [
            "bash",
            str(AE / "scripts/run_qwen3_solver_cell.sh"),
        ]
    model = MODELS[job["tag"]]
    max_context = 8192 if job["tp"] == 2 else 16384
    batch = 4 if job["tp"] == 2 else 8
    mem = 0.85 if job["tp"] == 2 else 0.90
    return [
        str(VLLM_PY),
        "-u",
        "scripts/score_wait_vllm.py",
        "--candidates",
        str(job["candidates"]),
        "--out",
        str(job["out"]),
        "--model",
        str(model),
        "--dataset",
        job["dataset"],
        "--prompt-mode",
        "puma",
        "--shard-id",
        str(job["shard"]),
        "--num-shards",
        str(job["n_shards"]),
        "--max-context",
        str(max_context),
        "--batch-size",
        str(batch),
        "--gpu-mem-util",
        str(mem),
        "--tp",
        str(job["tp"]),
    ]


def nvidia_ld() -> str:
    root = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
    extra = ":".join(sorted(str(path) for path in (root / ".venv" / "lib").glob("**/nvidia/*/lib") if path.is_dir()))
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


class GpuPool:
    def __init__(self, gpus: list[int]) -> None:
        self.cv = Condition()
        self.free = list(gpus)

    def acquire(self, n: int) -> list[int]:
        with self.cv:
            while len(self.free) < n:
                self.cv.wait()
            taken = [self.free.pop(0) for _ in range(n)]
            return taken

    def try_acquire(self, n: int) -> list[int] | None:
        with self.cv:
            if len(self.free) < n:
                return None
            return [self.free.pop(0) for _ in range(n)]

    def release(self, gpus: list[int]) -> None:
        with self.cv:
            self.free.extend(gpus)
            self.free.sort()
            self.cv.notify_all()

    def snapshot(self) -> list[int]:
        with self.cv:
            return list(self.free)


def import_existing_puma_wait() -> None:
    """Qwen3 Wait was already scored with --prompt-mode puma; reuse those files."""
    # Leftover Qwen3 PUMA copies may still sit in the retired dense_wait name.
    src_root = AE / "results/confcal_judge/v2/dense_wait"
    dst_root = Path(os.environ.get("WAIT_ROOT") or AE / "results/confcal_judge/v2/dense_puma_wait")
    for tag in ("qwen3_4b", "qwen3_8b"):
        src = src_root / tag
        if not src.exists():
            continue
        for shard in src.rglob("scores_shard*.jsonl"):
            dest = dst_root / tag / shard.relative_to(src)
            if dest.exists() and dest.stat().st_size >= shard.stat().st_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shard, dest)
            log(f"import {shard.relative_to(src_root)} -> puma_wait")


def start_job(job: dict[str, Any], gpus: list[int]) -> subprocess.Popen[str]:
    log_path = LOG_DIR / os.environ.get("WAIT_JOB_DIR", "wait_jobs") / f"{job['name'].replace(':', '_')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
        "VLLM_LENS_DISABLE": "1",
        "LD_LIBRARY_PATH": nvidia_ld(),
        "PY": str(VLLM_PY),
        "AE_PY": str(HF_PY),
        "WAIT_ROOT": os.environ.get("WAIT_ROOT") or str(AE / "results/confcal_judge/v2/dense_puma_wait"),
    }
    if job["kind"] == "cell":
        env["MODEL_TAG"] = job["tag"]
        env["DATASET"] = job["dataset"]
        env["SEED"] = str(job["seed"])
        env["GPU"] = env["CUDA_VISIBLE_DEVICES"]
    cmd = command_for(job)
    log(f"start {job['name']} gpus={gpus} pending={job.get('_pending')}")
    handle = log_path.open("a")
    handle.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {job['name']} gpus={gpus} ====\n")
    handle.flush()
    proc = subprocess.Popen(cmd, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    proc._wait_log = handle  # type: ignore[attr-defined]
    return proc


def add_wait_jobs(bucket: list[dict[str, Any]], tag: str, dataset: str, seed: int | None, n_shards: int, tp: int) -> None:
    dump_one(tag, dataset, seed)
    for shard in range(n_shards):
        bucket.append(make_wait_job(tag, dataset, seed, shard, n_shards, tp))


def plan_jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    wait_1: list[dict[str, Any]] = []
    wait_2: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []

    for tag in ("r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b", "qwen3_8b"):
        for dataset in BIG:
            add_wait_jobs(wait_1, tag, dataset, None, N_SHARDS_1, 1)
        for dataset in AIME:
            for seed in SEEDS:
                add_wait_jobs(wait_1, tag, dataset, seed, N_SHARDS_1, 1)

    add_wait_jobs(wait_2, "r1_32b", "math-500", None, N_SHARDS_2, 2)
    for dataset in ("olympiadbench", "gpqa-diamond"):
        add_wait_jobs(wait_2, "r1_32b", dataset, None, N_SHARDS_2, 2)
    for dataset in AIME:
        for seed in SEEDS:
            add_wait_jobs(wait_2, "r1_32b", dataset, seed, N_SHARDS_2, 2)
            add_wait_jobs(wait_2, "qwen3_30b_a3b", dataset, seed, N_SHARDS_2, 2)
    return wait_1, wait_2, cells


def keep_pending(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for job in jobs:
        if job["kind"] != "cell" and (job["tag"], job["dataset"], job["seed"]) in SKIP_WAIT:
            continue
        pending = job_pending(job)
        job["_pending"] = pending
        if pending == 0:
            log(f"skip {job['name']} pending=0")
            continue
        if pending < 0:
            log(f"skip {job['name']} missing input")
            continue
        kept.append(job)
        log(f"queue {job['name']} pending={pending}")
    return kept


def main() -> None:
    raw = os.environ.get("GPUS", "0,1,2,3,4,5")
    gpus = [int(part) for part in raw.split(",") if part.strip()]
    forbidden = {6, 7}
    if forbidden & set(gpus):
        raise SystemExit("refusing to use GPUs 6/7")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"wait-vllm-queue start gpus={gpus}")
    import_existing_puma_wait()
    wait_1, wait_2, cells = plan_jobs()
    q_a = keep_pending(wait_1)
    q_tp2 = keep_pending(wait_2)
    q_b = keep_pending(cells)
    log(f"queued A1={len(q_a)} TP2={len(q_tp2)} B={len(q_b)}")

    pool = GpuPool(gpus)
    running: list[dict[str, Any]] = []
    prefer_b = False
    failed = 0

    def pop_1gpu() -> dict[str, Any] | None:
        nonlocal prefer_b
        if prefer_b and q_b:
            prefer_b = False
            return q_b.pop(0)
        if q_a:
            prefer_b = True
            return q_a.pop(0)
        if q_b:
            return q_b.pop(0)
        return None

    while q_a or q_tp2 or q_b or running:
        still: list[dict[str, Any]] = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                still.append(item)
                continue
            handle = getattr(item["proc"], "_wait_log", None)
            if handle:
                handle.close()
            pool.release(item["gpus"])
            leftover = job_pending(item["job"])
            log(f"end {item['job']['name']} code={code} leftover={leftover} gpus={item['gpus']}")
            if code or leftover > 0:
                failed += 1
                if leftover > 0:
                    item["job"]["_pending"] = leftover
                    if item["job"].get("tp") == 2:
                        q_tp2.append(item["job"])
                    elif item["job"]["kind"] == "cell":
                        q_b.append(item["job"])
                    else:
                        q_a.append(item["job"])
                    log(f"requeue {item['job']['name']} leftover={leftover}")
        running = still

        started = False
        tp2_running = sum(1 for item in running if item["job"].get("tp") == 2)
        if q_tp2 and tp2_running < MAX_TP2:
            taken = pool.try_acquire(2)
            if taken:
                job = q_tp2.pop(0)
                proc = start_job(job, taken)
                running.append({"job": job, "gpus": taken, "proc": proc})
                started = True
        if not started:
            job = pop_1gpu()
            if job is not None:
                taken = pool.try_acquire(1)
                if taken is None:
                    if job["kind"] == "cell":
                        q_b.insert(0, job)
                    else:
                        q_a.insert(0, job)
                else:
                    proc = start_job(job, taken)
                    running.append({"job": job, "gpus": taken, "proc": proc})
                    started = True

        write_status(
            {
                "busy": {",".join(str(g) for g in item["gpus"]): item["job"]["name"] for item in running},
                "free": pool.snapshot(),
                "left": {"A1": len(q_a), "TP2": len(q_tp2), "B": len(q_b), "running": len(running)},
                "failed": failed,
            }
        )
        if not started:
            time.sleep(5)

    write_status({"busy": {}, "free": gpus, "left": {"A1": 0, "TP2": 0, "B": 0, "running": 0}, "failed": failed})
    log(f"wait-vllm-queue done failed={failed}")


if __name__ == "__main__":
    main()
