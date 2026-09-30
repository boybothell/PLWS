#!/usr/bin/env bash
# Dynamic count-bias queue. One cold load at a time. Does not write 窗后压 scores.
#
#   GPUS=2,5 MODEL_TAG=r1_1p5b SEEDS=42,0,1 \
#     bash scripts/run_count_bias_queue.sh
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_TAG="${MODEL_TAG:-r1_1p5b}"
SEEDS="${SEEDS:-42,0,1}"
GPUS="${GPUS:?set GPUS, for example 2,5}"
DATASETS="${DATASETS:-amc23,aime25,gpqa-diamond,math-500,olympiadbench}"
export PLWS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/.venv/bin/python" - "$ROOT" "$MODEL_TAG" "$SEEDS" "$GPUS" "$DATASETS" <<'PY'
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

argv = __import__("sys").argv
ROOT = Path(argv[1])
MODEL = argv[2]
SEEDS = [int(item) for item in argv[3].split(",") if item.strip()]
GPUS = [item.strip() for item in argv[4].split(",") if item.strip()]
DATASETS = [item.strip() for item in argv[5].split(",") if item.strip()]
CELLS = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
LOG_ROOT = ROOT / "results" / "runs" / "plws" / "count_bias" / "queue_logs"
STATUS = ROOT / "results" / "runs" / "plws" / "count_bias" / "queue_status.json"
LOADED = (
    "Model loaded.",
    "init engine (profile, create kv cache, warmup model) took",
    "Running generation...",
    "official-batch step=",
)


def key(dataset: str, seed: int) -> str:
    return f"{dataset}__s{seed}"


def jobs_path(dataset: str, seed: int) -> Path:
    return (
        ROOT
        / "results/runs/plws/window_first/k_4/lexicon_core"
        / MODEL
        / dataset
        / f"seed_{seed}"
        / "jobs/firstwin.jsonl"
    )


def out_path(dataset: str, seed: int) -> Path:
    return (
        ROOT
        / "results/runs/plws/count_bias"
        / MODEL
        / dataset
        / f"seed_{seed}"
        / "shard_0.jsonl"
    )


def job_count(dataset: str, seed: int) -> int:
    path = jobs_path(dataset, seed)
    if not path.is_file():
        raise SystemExit(f"missing jobs {path}")
    return sum(1 for line in path.open() if line.strip())


def finished_count(dataset: str, seed: int) -> int:
    path = out_path(dataset, seed)
    found: set[str] = set()
    if not path.is_file():
        return 0
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("bias_schedule") != "count":
            continue
        if row.get("status") in {"ok", "too_long"} and row.get("uid"):
            found.add(str(row["uid"]))
    return len(found)


def cell_done(dataset: str, seed: int, expected: int) -> bool:
    return finished_count(dataset, seed) >= expected


def proc_state(pid: int) -> str | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return line.split()[1]
    except OSError:
        return None
    return None


def alive(pid: int) -> bool:
    state = proc_state(pid)
    return state is not None and not state.startswith("Z")


def environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        name, value = item.split(b"=", 1)
        out[name.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return out


def cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def arg(cmd: list[str], flag: str) -> str | None:
    if flag not in cmd:
        return None
    index = cmd.index(flag)
    if index + 1 >= len(cmd):
        return None
    return cmd[index + 1]


def discover() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if not alive(pid):
            continue
        cmd = cmdline(pid)
        if "score_leftover_suppress.py" not in " ".join(cmd):
            continue
        if arg(cmd, "--bias-schedule") != "count":
            continue
        if arg(cmd, "--model-tag") != MODEL:
            continue
        dataset = arg(cmd, "--dataset")
        seed_text = arg(cmd, "--seed")
        if dataset not in DATASETS or seed_text is None:
            continue
        seed = int(seed_text)
        if seed not in SEEDS:
            continue
        gpu = environ(pid).get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        found[key(dataset, seed)] = {
            "dataset": dataset,
            "seed": seed,
            "pid": pid,
            "gpu": gpu,
            "log": None,
            "adopted": True,
        }
    return found


def gpu_used() -> dict[str, int]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used: dict[str, int] = {}
    for line in text.splitlines():
        index, memory = [item.strip() for item in line.split(",")]
        used[index] = int(memory)
    return used


def workers_loading() -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
        except OSError:
            continue
        if comm.startswith("VLLM::") and proc_state(int(proc.name)) == "D":
            return True
    return False


def log_loaded(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    text = path.read_text(errors="replace")
    marker = text.rfind(" start gpus=")
    current = text[marker:] if marker >= 0 else text
    return any(item in current for item in LOADED)


def write_status(running: dict, pending: list[str], done: list[str]) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": MODEL,
        "seeds": SEEDS,
        "datasets": DATASETS,
        "gpus": GPUS,
        "pending": pending,
        "done": done,
        "running": [
            {
                "cell": name,
                "dataset": item["dataset"],
                "seed": item["seed"],
                "pid": item["pid"],
                "gpu": item["gpu"],
                "log": None if item["log"] is None else str(item["log"]),
                "adopted": item["adopted"],
            }
            for name, item in running.items()
        ],
    }
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATUS)


def launch(dataset: str, seed: int, gpu: str, attempt: int) -> dict:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{MODEL}__{dataset}__s{seed}.attempt_{attempt}.log"
    with log_path.open("ab") as handle:
        handle.write(
            f"\n# {time.strftime('%Y-%m-%dT%H:%M:%S%z')} start gpus={gpu}\n".encode()
        )
        process = subprocess.Popen(
            ["bash", str(ROOT / "scripts/run_count_bias_cell.sh")],
            cwd=ROOT,
            env={
                **os.environ,
                "PLWS_ROOT": str(ROOT),
                "MODEL_TAG": MODEL,
                "DATASET": dataset,
                "SEED": str(seed),
                "GPU": gpu,
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(
        f"start {dataset} seed={seed} gpu={gpu} pid={process.pid} attempt={attempt}",
        flush=True,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "pid": process.pid,
        "gpu": gpu,
        "log": log_path,
        "adopted": False,
    }


stopping = False


def request_stop(signum, _frame) -> None:
    global stopping
    stopping = True
    print(f"stop requested signal={signum}", flush=True)


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

expected = {(dataset, seed): job_count(dataset, seed) for dataset, seed in CELLS}
running = discover()
attempts = {key(dataset, seed): 0 for dataset, seed in CELLS}
for name, item in running.items():
    print(f"adopt {name} gpu={item['gpu']} pid={item['pid']}", flush=True)
done: list[str] = []

while True:
    if stopping:
        pending_now = [
            key(dataset, seed)
            for dataset, seed in CELLS
            if key(dataset, seed) not in running and key(dataset, seed) not in done
        ]
        write_status(running, pending_now, done)
        raise SystemExit(130)
    for name, item in list(running.items()):
        if alive(item["pid"]):
            continue
        del running[name]
        dataset, seed = item["dataset"], item["seed"]
        if cell_done(dataset, seed, expected[(dataset, seed)]):
            done.append(name)
            print(f"done {name}", flush=True)
        else:
            print(f"retry {name}", flush=True)
    pending: list[tuple[str, int]] = []
    for dataset, seed in CELLS:
        name = key(dataset, seed)
        if name in running or name in done:
            continue
        if cell_done(dataset, seed, expected[(dataset, seed)]):
            done.append(name)
            continue
        pending.append((dataset, seed))
    claimed = {item["gpu"] for item in running.values() if item["gpu"]}
    used = gpu_used()
    idle = [gpu for gpu in GPUS if gpu not in claimed and used.get(gpu, 10**9) < 800]
    still_loading = [
        name
        for name, item in running.items()
        if not item["adopted"] and not log_loaded(item["log"])
    ]
    if pending and idle and not still_loading and not workers_loading():
        dataset, seed = pending[0]
        name = key(dataset, seed)
        attempts[name] += 1
        running[name] = launch(dataset, seed, idle[0], attempts[name])
    write_status(
        running,
        [key(dataset, seed) for dataset, seed in pending],
        done,
    )
    if not pending and not running:
        print("queue finished", flush=True)
        raise SystemExit(0)
    time.sleep(5)
PY
