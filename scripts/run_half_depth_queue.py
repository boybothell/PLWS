#!/usr/bin/env python3
"""Half-depth extract after Wait. Cards 0-5, never 6/7.

HF score_dense_once writes last-layer and half-depth lens. Already-written
lens rows are skipped. Wait-only files are not treated as half-depth done.
7B/8B complete sets stay skipped except 7B GPQA's four missing questions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Condition
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from dump_dense_candidates import trial_path  # noqa: E402
from score_dense_once import _scored_for_lag  # noqa: E402
from run_wait_vllm_queue import GpuPool, cand_path, dump_one, nvidia_ld  # noqa: E402

HF_PY = AE / ".venv/bin/python"
LOG_DIR = AE / "results/_logs"
STATUS = LOG_DIR / "half_depth_status.json"
MAIN_LOG = LOG_DIR / "half_depth_queue.log"

MODELS = {
    "r1_7b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"),
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
MAX_TP2 = 1


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_LOG.open("a") as handle:
        handle.write(line + "\n")


def write_status(payload: dict[str, Any]) -> None:
    STATUS.write_text(json.dumps({"ts": time.time(), **payload}, indent=2, ensure_ascii=False))


def lens_out_path(tag: str, dataset: str, seed: int | None, shard_id: int) -> Path:
    root = AE / "results/confcal_judge/v2/dense_lens"
    if tag == "r1_7b" and seed is None and dataset in BIG:
        return root / dataset / f"scores_shard{shard_id}.jsonl"
    stem = f"{dataset}_s{seed}" if seed is not None else dataset
    return root / tag / stem / f"scores_shard{shard_id}.jsonl"


def load_half_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    out: set[tuple[int, int, str]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if _scored_for_lag(row):
                out.add((int(row["question_idx"]), int(row["decision_step"]), str(row.get("answer") or "")))
    return out


def half_pending(candidates: Path, out: Path, shard_id: int, num_shards: int) -> int:
    if not candidates.exists():
        return -1
    rows = [json.loads(line) for line in candidates.read_text().splitlines() if line.strip()]
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for index, qi in enumerate(qis) if index % num_shards == shard_id}
    done = load_half_keys(out)
    return sum(
        1
        for row in rows
        if int(row["question_idx"]) in keep
        and (int(row["question_idx"]), int(row["decision_step"]), str(row.get("answer") or "")) not in done
    )


def make_job(tag: str, dataset: str, seed: int | None, shard: int, n_shards: int, tp: int) -> dict[str, Any]:
    return {
        "kind": "half",
        "name": f"half:{tag}:{dataset}:s{seed if seed is not None else '-'}:shard{shard}",
        "tag": tag,
        "dataset": dataset,
        "seed": seed,
        "shard": shard,
        "n_shards": n_shards,
        "tp": tp,
        "candidates": cand_path(tag, dataset, seed),
        "out": lens_out_path(tag, dataset, seed, shard),
    }


def plan_jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    one: list[dict[str, Any]] = []
    two: list[dict[str, Any]] = []

    only = {part.strip() for part in os.environ.get("ONLY_TAGS", "").split(",") if part.strip()}

    if not only or "r1_7b" in only:
        dump_one("r1_7b", "gpqa-diamond", None)
        for shard in range(N_SHARDS_1):
            one.append(make_job("r1_7b", "gpqa-diamond", None, shard, N_SHARDS_1, 1))

    if not only or "r1_14b" in only:
        for dataset in ("olympiadbench", "gpqa-diamond"):
            dump_one("r1_14b", dataset, None)
            for shard in range(N_SHARDS_1):
                one.append(make_job("r1_14b", dataset, None, shard, N_SHARDS_1, 1))
        for dataset in AIME:
            for seed in SEEDS:
                dump_one("r1_14b", dataset, seed)
                for shard in range(N_SHARDS_1):
                    one.append(make_job("r1_14b", dataset, seed, shard, N_SHARDS_1, 1))

    for tag in ("qwen3_4b", "qwen3_8b"):
        if only and tag not in only:
            continue
        for dataset in BIG:
            dump_one(tag, dataset, None)
            for shard in range(N_SHARDS_1):
                one.append(make_job(tag, dataset, None, shard, N_SHARDS_1, 1))
        for dataset in AIME:
            for seed in SEEDS:
                dump_one(tag, dataset, seed)
                for shard in range(N_SHARDS_1):
                    one.append(make_job(tag, dataset, seed, shard, N_SHARDS_1, 1))

    if not only or "r1_32b" in only:
        dump_one("r1_32b", "math-500", None)
        two.append(make_job("r1_32b", "math-500", None, 0, 1, 2))
        for dataset in ("olympiadbench", "gpqa-diamond"):
            dump_one("r1_32b", dataset, None)
            two.append(make_job("r1_32b", dataset, None, 0, 1, 2))
        for dataset in AIME:
            for seed in SEEDS:
                dump_one("r1_32b", dataset, seed)
                two.append(make_job("r1_32b", dataset, seed, 0, 1, 2))
    if not only or "qwen3_30b_a3b" in only:
        for dataset in AIME:
            for seed in SEEDS:
                dump_one("qwen3_30b_a3b", dataset, seed)
                two.append(make_job("qwen3_30b_a3b", dataset, seed, 0, 1, 2))
    return one, two


def keep_pending(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for job in jobs:
        pending = half_pending(job["candidates"], job["out"], job["shard"], job["n_shards"])
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


def command_for(job: dict[str, Any]) -> list[str]:
    tag = job["tag"]
    if tag == "r1_32b":
        max_context = "4096"
    elif tag in {"qwen3_4b", "qwen3_8b"}:
        max_context = "16384"
    else:
        max_context = "8192"
    cmd = [
        str(HF_PY),
        "scripts/score_dense_once.py",
        "--candidates",
        str(job["candidates"]),
        "--out",
        str(job["out"]),
        "--shard-id",
        str(job["shard"]),
        "--num-shards",
        str(job["n_shards"]),
        "--model",
        str(MODELS[tag]),
        "--max-context",
        max_context,
        "--prompt-mode",
        "puma",
        "--dataset",
        job["dataset"],
    ]
    if job["tp"] == 2:
        cmd.extend(["--device-map", "auto"])
    return cmd


def start_job(job: dict[str, Any], gpus: list[int]) -> subprocess.Popen[str]:
    log_path = LOG_DIR / "half_jobs" / f"{job['name'].replace(':', '_')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
        "VLLM_LENS_DISABLE": "1",
        "LD_LIBRARY_PATH": nvidia_ld(),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    cmd = command_for(job)
    log(f"start {job['name']} gpus={gpus} pending={job.get('_pending')}")
    handle = log_path.open("a")
    handle.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {job['name']} gpus={gpus} ====\n")
    handle.flush()
    proc = subprocess.Popen(cmd, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    proc._wait_log = handle  # type: ignore[attr-defined]
    return proc


def main() -> None:
    raw = os.environ.get("GPUS", "0,1,2,3,4,5")
    gpus = [int(part) for part in raw.split(",") if part.strip()]
    if {6, 7} & set(gpus):
        raise SystemExit("refusing to use GPUs 6/7")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"half-depth-queue start gpus={gpus}")
    one, two = plan_jobs()
    q_1 = keep_pending(one)
    q_2 = keep_pending(two)
    log(f"queued 1gpu={len(q_1)} tp2={len(q_2)}")

    pool = GpuPool(gpus)
    running: list[dict[str, Any]] = []
    failed = 0

    while q_1 or q_2 or running:
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
            leftover = half_pending(item["job"]["candidates"], item["job"]["out"], item["job"]["shard"], item["job"]["n_shards"])
            log(f"end {item['job']['name']} code={code} leftover={leftover} gpus={item['gpus']}")
            if code or leftover > 0:
                failed += 1
                retries = int(item["job"].get("_retries") or 0)
                if leftover > 0 and retries < 2:
                    item["job"]["_pending"] = leftover
                    item["job"]["_retries"] = retries + 1
                    (q_2 if item["job"]["tp"] == 2 else q_1).append(item["job"])
                    log(f"requeue {item['job']['name']} leftover={leftover} retry={retries + 1}")
        running = still

        started = False
        tp2_running = sum(1 for item in running if item["job"].get("tp") == 2)
        if q_2 and tp2_running < MAX_TP2:
            taken = pool.try_acquire(2)
            if taken:
                job = q_2.pop(0)
                running.append({"job": job, "gpus": taken, "proc": start_job(job, taken)})
                started = True
        if not started and q_1:
            taken = pool.try_acquire(1)
            if taken:
                job = q_1.pop(0)
                running.append({"job": job, "gpus": taken, "proc": start_job(job, taken)})
                started = True

        write_status(
            {
                "busy": {",".join(str(g) for g in item["gpus"]): item["job"]["name"] for item in running},
                "free": pool.snapshot(),
                "left": {"1gpu": len(q_1), "tp2": len(q_2), "running": len(running)},
                "failed": failed,
            }
        )
        if not started:
            time.sleep(5)

    write_status({"busy": {}, "free": gpus, "left": {"1gpu": 0, "tp2": 0, "running": 0}, "failed": failed})
    log(f"half-depth-queue done failed={failed}")


if __name__ == "__main__":
    main()
