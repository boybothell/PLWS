#!/usr/bin/env python3
"""Dynamic 4-GPU queue for lag-door screen extracts. One shard = one job.

Cards 0/2/3/4 grab the next unfinished shard as soon as they free.
Already-written keys are skipped. Half-depth + Wait (score_dense_once)
finish for every 7B/8B/14B set before any process-probe job is queued.
32B / 30B stay on the 2-GPU queue.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from dump_dense_candidates import trial_path  # noqa: E402


def row_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))


def scored_for_lag(row: dict[str, Any]) -> bool:
    status = row.get("status")
    if status in {"too_long", "oom", "no_answer"}:
        return True
    return bool(status == "ok" and ("stop_margin" in row or row.get("once")))


def load_keys(path: Path, pred) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    out: set[tuple[int, int, str]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if pred(row):
                out.add(row_key(row))
    return out


def twin_internal(lens_out: Path) -> Path:
    text = str(lens_out)
    if "/dense_lens/" in text:
        return Path(text.replace("/dense_lens/", "/dense_internal/"))
    return lens_out


def done_keys(path: Path) -> set[tuple[int, int, str]]:
    """Half-depth + Wait already written, or skipped. Old split extracts count
    if lens is ok and the twin internal file has Wait."""
    done = load_keys(path, scored_for_lag)
    if "/dense_lens/" not in str(path):
        return done
    lens_ok = load_keys(path, lambda row: row.get("status") == "ok")
    wait_ok = load_keys(twin_internal(path), scored_for_lag)
    return done | (lens_ok & wait_ok)

HF_PY = AE / ".venv/bin/python"
VLLM_PY = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python")
LOG_DIR = AE / "results/_logs"
STATUS = LOG_DIR / "lag_screen_status.json"
MAIN_LOG = LOG_DIR / "lag_screen_queue.log"

MODELS = {
    "r1_7b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B"),
    "nemotron_8b": Path("/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1"),
    "r1_14b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B"),
}
BIG = ("math-500", "olympiadbench", "gpqa-diamond")
AIME = ("aime24", "aime25")
SEEDS = (42, 0, 1, 123)
N_SHARDS = 4
GPUS = (0, 2, 3, 4)

LOCK = Lock()
BUSY: dict[int, str] = {}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_LOG.open("a") as handle:
        handle.write(line + "\n")


def write_status(extra: dict[str, Any] | None = None) -> None:
    payload = {"busy": dict(BUSY), "ts": time.time()}
    if extra:
        payload.update(extra)
    STATUS.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def cand_path(tag: str, dataset: str, seed: int | None) -> Path:
    root = AE / "results/confcal_judge/v2/dense_candidates"
    if tag == "r1_7b" and dataset in BIG:
        return root / f"{dataset}.jsonl"
    name = f"{dataset}.jsonl" if seed is None else f"{dataset}_s{seed}.jsonl"
    return root / tag / name


def score_dir(kind: str, tag: str, dataset: str, seed: int | None) -> Path:
    root = AE / f"results/confcal_judge/v2/dense_{kind}"
    if tag == "r1_7b" and dataset in BIG:
        return root / dataset
    stem = f"{dataset}_s{seed}" if seed is not None else dataset
    return root / tag / stem


def dump_one(tag: str, dataset: str, seed: int | None) -> Path:
    out = cand_path(tag, dataset, seed)
    if out.exists() and out.stat().st_size > 0:
        log(f"dump skip {tag} {dataset} seed={seed} ({out.name} exists)")
        return out
    cmd = [str(HF_PY), "scripts/dump_dense_candidates.py", "--dataset", dataset, "--model-tag", tag]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    log(f"dump {' '.join(cmd[1:])}")
    subprocess.run(cmd, cwd=AE, check=True)
    return out


def cand_pending(candidates: Path, out: Path, shard_id: int, num_shards: int) -> int:
    if not candidates.exists():
        return -1
    rows = [json.loads(line) for line in candidates.read_text().splitlines() if line.strip()]
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for i, qi in enumerate(qis) if i % num_shards == shard_id}
    done = done_keys(out)
    return sum(
        1
        for row in rows
        if int(row["question_idx"]) in keep
        and (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])) not in done
    )


def trial_pending(trial: Path, out: Path, shard_id: int, num_shards: int) -> int:
    if not trial.exists():
        return -1
    rows = json.loads(trial.read_text())
    qis = sorted({int(row["question_idx"]) for row in rows})
    keep = {qi for i, qi in enumerate(qis) if i % num_shards == shard_id}
    done = done_keys(out)
    return sum(
        1
        for row in rows
        if int(row["question_idx"]) in keep
        and (int(row["question_idx"]), int(row["stopped_len"]), str(row.get("final_answer") or "")) not in done
    )


def make_extract(kind: str, tag: str, dataset: str, seed: int | None) -> list[dict[str, Any]]:
    cands = cand_path(tag, dataset, seed)
    folder = "internal" if kind == "internal" else "lens" if kind == "lens" else "solver_probes"
    outd = score_dir(folder, tag, dataset, seed)
    jobs: list[dict[str, Any]] = []
    for shard in range(N_SHARDS):
        jobs.append(
            {
                "name": f"{kind}:{tag}:{dataset}:s{seed if seed is not None else '-'}:shard{shard}",
                "kind": kind,
                "tag": tag,
                "dataset": dataset,
                "seed": seed,
                "shard": shard,
                "candidates": cands,
                "trial_seed": seed if dataset.startswith("aime") else None,
                "out": outd / f"scores_shard{shard}.jsonl",
            }
        )
    return jobs


def enqueue(q: Queue, jobs: list[dict[str, Any]], stats: dict[str, int]) -> None:
    for job in jobs:
        pending = job_pending(job)
        if pending == 0:
            continue
        if pending < 0:
            log(f"queue skip missing {job['name']}")
            continue
        q.put(job)
        stats["queued"] += 1
        log(f"queue {job['name']} pending={pending}")


def job_pending(job: dict[str, Any]) -> int:
    if job["kind"] == "probes":
        trial = trial_path(job["tag"], job["dataset"], job["trial_seed"])
        return trial_pending(trial, job["out"], job["shard"], N_SHARDS)
    return cand_pending(job["candidates"], job["out"], job["shard"], N_SHARDS)


def command_for(job: dict[str, Any]) -> list[str]:
    model = str(MODELS[job["tag"]])
    if job["kind"] in ("internal", "lens", "once"):
        out = job["out"]
        if job["kind"] == "internal":
            # one-forward writer lives in dense_lens; skip leftover internal-only jobs
            out = Path(str(out).replace("/dense_internal/", "/dense_lens/"))
        return [
            str(HF_PY),
            "scripts/score_dense_once.py",
            "--candidates",
            str(job["candidates"]),
            "--out",
            str(out),
            "--shard-id",
            str(job["shard"]),
            "--num-shards",
            str(N_SHARDS),
            "--model",
            model,
        ]
    cmd = [
        str(VLLM_PY),
        "scripts/score_dense_solver_probes.py",
        "--dataset",
        job["dataset"],
        "--model-tag",
        job["tag"],
        "--model",
        model,
        "--shard-id",
        str(job["shard"]),
        "--num-shards",
        str(N_SHARDS),
        "--out",
        str(job["out"]),
        "--tp",
        "1",
    ]
    if job["tag"] == "r1_14b":
        cmd.extend(["--max-context", "8192", "--gpu-mem-util", "0.90"])
    if job["trial_seed"] is not None:
        cmd.extend(["--seed", str(job["trial_seed"])])
    return cmd


def run_one(gpu: int, job: dict[str, Any]) -> int:
    pending = job_pending(job)
    if pending == 0:
        log(f"gpu{gpu} skip {job['name']} pending=0")
        return 0
    if pending < 0:
        log(f"gpu{gpu} skip {job['name']} missing input")
        return 0
    log_path = job["out"].parent / "logs" / f"shard{job['shard']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "VLLM_LENS_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    cmd = command_for(job)
    log(f"gpu{gpu} start {job['name']} pending={pending}")
    with LOCK:
        BUSY[gpu] = job["name"]
        write_status()
    started = time.perf_counter()
    with log_path.open("a") as handle:
        handle.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {job['name']} ====\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    log(f"gpu{gpu} end {job['name']} code={proc.returncode} {elapsed:.0f}s")
    with LOCK:
        BUSY.pop(gpu, None)
        write_status()
    return proc.returncode


def worker(gpu: int, jobs: Queue) -> None:
    while True:
        try:
            job = jobs.get(timeout=1)
        except Empty:
            continue
        if job is None:
            jobs.task_done()
            break
        try:
            run_one(gpu, job)
        except Exception as exc:  # noqa: BLE001
            log(f"gpu{gpu} crash {job['name']}: {exc}")
        jobs.task_done()


def main() -> None:
    gpus = tuple(int(x) for x in os.environ.get("GPUS", "0,2,3,4").split(",") if x.strip())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"lag-screen-queue start gpus={gpus}")
    q: Queue = Queue()
    stats = {"queued": 0}
    threads = [Thread(target=worker, args=(gpu, q), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()

    # Failed 14B first, then leftover 7B/8B. Probes stay off until every
    # half-depth + Wait shard on these cards reports pending=0.
    for ds in BIG:
        dump_one("r1_14b", ds, None)
        enqueue(q, make_extract("lens", "r1_14b", ds, None), stats)
    write_status({"queued": stats["queued"]})
    for ds in AIME:
        for seed in SEEDS:
            dump_one("r1_14b", ds, seed)
            enqueue(q, make_extract("lens", "r1_14b", ds, seed), stats)

    enqueue(q, make_extract("lens", "r1_7b", "gpqa-diamond", None), stats)
    for tag in ("nemotron_8b", "r1_7b"):
        for ds in BIG:
            if tag == "r1_7b" and ds == "gpqa-diamond":
                continue
            dump_one(tag, ds, None)
            enqueue(q, make_extract("lens", tag, ds, None), stats)
        for ds in AIME:
            for seed in SEEDS:
                dump_one(tag, ds, seed)
                enqueue(q, make_extract("lens", tag, ds, seed), stats)

    log(f"queued {stats['queued']} lens shard jobs; probes not queued")
    write_status({"queued": stats["queued"]})
    for _ in gpus:
        q.put(None)
    q.join()
    for thread in threads:
        thread.join()
    log("lag-screen-queue done")


if __name__ == "__main__":
    main()
