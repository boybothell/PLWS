#!/usr/bin/env python3
"""2-GPU sequential queue for 32B / 30B lag-door extracts.

These models do not fit on one card. Default pair: 1,5. Does not touch 0/2/3/4.
30B only has AIME dense trials; 32B has math / olympiad / GPQA / AIME.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from dump_dense_candidates import trial_path  # noqa: E402

HF_PY = AE / ".venv/bin/python"
VLLM_PY = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python")
LOG_DIR = AE / "results/_logs"
STATUS = LOG_DIR / "lag_screen_tp2_status.json"
MAIN_LOG = LOG_DIR / "lag_screen_tp2.log"

MODELS = {
    "r1_32b": Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-32B"),
    "qwen3_30b_a3b": Path("/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507"),
}
BIG = ("math-500", "olympiadbench", "gpqa-diamond")
AIME = ("aime24", "aime25")
SEEDS = (42, 0, 1, 123)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_LOG.open("a") as handle:
        handle.write(line + "\n")


def write_status(current: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"current": current, "ts": time.time()}
    if extra:
        payload.update(extra)
    STATUS.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


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
    done = load_keys(path, scored_for_lag)
    if "/dense_lens/" not in str(path):
        return done
    lens_ok = load_keys(path, lambda row: row.get("status") == "ok")
    wait_ok = load_keys(twin_internal(path), scored_for_lag)
    return done | (lens_ok & wait_ok)


def cand_path(tag: str, dataset: str, seed: int | None) -> Path:
    name = f"{dataset}.jsonl" if seed is None else f"{dataset}_s{seed}.jsonl"
    return AE / "results/confcal_judge/v2/dense_candidates" / tag / name


def score_out(kind: str, tag: str, dataset: str, seed: int | None) -> Path:
    folder = {"internal": "internal", "lens": "lens", "probes": "solver_probes"}[kind]
    stem = f"{dataset}_s{seed}" if seed is not None else dataset
    return AE / f"results/confcal_judge/v2/dense_{folder}" / tag / stem / "scores_shard0.jsonl"


def dump_one(tag: str, dataset: str, seed: int | None) -> None:
    out = cand_path(tag, dataset, seed)
    if out.exists() and out.stat().st_size > 0:
        log(f"dump skip {tag} {dataset} seed={seed}")
        return
    cmd = [str(HF_PY), "scripts/dump_dense_candidates.py", "--dataset", dataset, "--model-tag", tag]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    log(f"dump {' '.join(cmd[1:])}")
    subprocess.run(cmd, cwd=AE, check=True)


def pending_count(candidates: Path, out: Path) -> int:
    if not candidates.exists():
        return -1
    rows = [json.loads(line) for line in candidates.read_text().splitlines() if line.strip()]
    done = done_keys(out)
    return sum(
        1
        for row in rows
        if (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])) not in done
    )


def jobs() -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []

    def add(kind: str, tag: str, dataset: str, seed: int | None) -> None:
        planned.append({"kind": kind, "tag": tag, "dataset": dataset, "seed": seed})

    add("lens", "r1_32b", "math-500", None)
    add("lens", "r1_32b", "olympiadbench", None)
    add("lens", "r1_32b", "gpqa-diamond", None)
    for ds in AIME:
        for seed in SEEDS:
            add("lens", "r1_32b", ds, seed)
            add("lens", "qwen3_30b_a3b", ds, seed)
    return planned


def run_one(job: dict[str, Any], gpus: str) -> int:
    tag, dataset, seed, kind = job["tag"], job["dataset"], job["seed"], job["kind"]
    name = f"{kind}:{tag}:{dataset}:s{seed if seed is not None else '-'}"
    cands = cand_path(tag, dataset, seed)
    out = score_out(kind, tag, dataset, seed)
    pending = pending_count(cands, out)
    if pending == 0:
        log(f"skip {name} pending=0")
        return 0
    if pending < 0:
        log(f"skip {name} missing candidates")
        return 0
    if kind == "probes":
        cmd = [
            str(VLLM_PY),
            "scripts/score_dense_solver_probes.py",
            "--dataset",
            dataset,
            "--model-tag",
            tag,
            "--model",
            str(MODELS[tag]),
            "--out",
            str(out),
            "--shard-id",
            "0",
            "--num-shards",
            "1",
            "--tp",
            "2",
            "--max-context",
            "8192",
            "--gpu-mem-util",
            "0.85",
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
    else:
        script = "scripts/score_dense_once.py"
        cmd = [
            str(HF_PY),
            script,
            "--candidates",
            str(cands),
            "--out",
            str(out),
            "--shard-id",
            "0",
            "--num-shards",
            "1",
            "--model",
            str(MODELS[tag]),
            "--device-map",
            "auto",
            "--max-context",
            "4096" if tag == "r1_32b" else "8192",
        ]
    log_path = out.parent / "logs" / "shard0.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpus,
        "VLLM_LENS_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    log(f"start {name} pending={pending} gpus={gpus}")
    write_status(name, {"pending": pending, "gpus": gpus})
    started = time.perf_counter()
    with log_path.open("a") as handle:
        handle.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} ====\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT)
    log(f"end {name} code={proc.returncode} {time.perf_counter() - started:.0f}s")
    write_status("", {"last": name, "code": proc.returncode})
    return proc.returncode


def main() -> None:
    gpus = os.environ.get("GPUS", "1,5")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"lag-screen-tp2 start gpus={gpus}")
    dump_one("r1_32b", "math-500", None)
    dump_one("r1_32b", "gpqa-diamond", None)
    dump_one("r1_32b", "olympiadbench", None)
    for ds in AIME:
        for seed in SEEDS:
            dump_one("r1_32b", ds, seed)
            dump_one("qwen3_30b_a3b", ds, seed)
    failed = 0
    planned = jobs()
    for job in planned:
        if job["seed"] is not None:
            try:
                trial_path(job["tag"], job["dataset"], job["seed"])
            except FileNotFoundError:
                log(f"skip missing trial {job}")
                continue
        code = run_one(job, gpus)
        if code:
            failed += 1
            log(f"continue after failure {job['tag']} {job['dataset']}")
    write_status("done", {"failed": failed, "n": len(planned)})
    log(f"lag-screen-tp2 done failed={failed}/{len(planned)}")


if __name__ == "__main__":
    main()
