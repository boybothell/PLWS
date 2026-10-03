#!/usr/bin/env bash
# Dynamic count-bias queue. One cold load at a time. Does not write 窗后压 scores.
#
#   GPUS=2,3,4,5 MODELS=r1_7b SEEDS=42 \
#     DATASETS=math-500,aime25 RHOS=0.80,0.90,0.95,0.98 \
#     bash scripts/run_count_bias_queue.sh
#
# One rho per output directory. 32B takes two GPUs. 1.5B, 7B, and Llama-8B
# submit the whole remaining shard. 14B uses 32. 32B uses 16.
set -euo pipefail

ROOT="${PLWS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_TAG="${MODEL_TAG:-r1_1p5b}"
MODELS="${MODELS:-$MODEL_TAG}"
SEEDS="${SEEDS:-42,0,1}"
GPUS="${GPUS:?set GPUS, for example 2,5}"
RHOS="${RHOS:-${RHO:-}}"
if [[ -z "$RHOS" ]]; then
  echo "set RHOS, for example 0.80,0.90,0.95,0.98" >&2
  exit 1
fi
DATASETS="${DATASETS:-amc23,aime25,gpqa-diamond,math-500,olympiadbench}"
export PLWS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/.venv/bin/python" - "$ROOT" "$MODELS" "$SEEDS" "$GPUS" "$DATASETS" "$RHOS" <<'PY'
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

argv = __import__("sys").argv
ROOT = Path(argv[1])
MODELS = [item.strip() for item in argv[2].split(",") if item.strip()]
SEEDS = [int(item) for item in argv[3].split(",") if item.strip()]
GPUS = [item.strip() for item in argv[4].split(",") if item.strip()]
DATASETS = [item.strip() for item in argv[5].split(",") if item.strip()]
RHOS = [item.strip() for item in argv[6].split(",") if item.strip()]
for rho in RHOS:
    value = float(rho)
    if not 0 < value < 1:
        raise SystemExit(f"rho must be in (0, 1), got {rho}")
CELLS: list[tuple[str, str, int, str]] = []
# dataset: finish every rho of the early datasets before later ones.
# model: the default, rho then model then dataset.
order = os.environ.get("COUNT_BIAS_ORDER", "model")
def _append(rho: str, model: str, dataset: str, seed: int) -> None:
    CELLS.append((model, dataset, seed, rho))
if order == "dataset":
    for dataset in DATASETS:
        for rho in RHOS:
            for model in MODELS:
                for seed in SEEDS:
                    _append(rho, model, dataset, seed)
else:
    for rho in RHOS:
        for model in MODELS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    _append(rho, model, dataset, seed)
BATCH = {
    "r1_1p5b": "0",
    "r1_7b": "0",
    "r1_llama_8b": "0",
    "r1_14b": "32",
    "r1_32b": "16",
}
TP = {"r1_32b": 2}
LOG_ROOT = ROOT / "results" / "runs" / "plws" / "count_bias" / "queue_logs"
STATUS = Path(
    os.environ.get(
        "COUNT_BIAS_STATUS",
        str(ROOT / "results" / "runs" / "plws" / "count_bias" / "queue_status.json"),
    )
)
LOADED = (
    "Model loaded.",
    "init engine (profile, create kv cache, warmup model) took",
    "Running generation...",
    "official-batch step=",
)


def rho_tag(rho: str) -> str:
    return "rho_" + rho.replace(".", "p")


def key(model: str, dataset: str, seed: int, rho: str) -> str:
    return f"{rho_tag(rho)}__{model}__{dataset}__s{seed}"


def jobs_path(model: str, dataset: str, seed: int) -> Path:
    return (
        ROOT
        / "results/runs/plws/window_first/k_4/lexicon_core"
        / model
        / dataset
        / f"seed_{seed}"
        / "jobs/firstwin.jsonl"
    )


def out_path(model: str, dataset: str, seed: int, rho: str) -> Path:
    return (
        ROOT
        / "results/runs/plws/count_bias"
        / rho_tag(rho)
        / model
        / dataset
        / f"seed_{seed}"
        / "shard_0.jsonl"
    )


def job_count(model: str, dataset: str, seed: int) -> int:
    path = jobs_path(model, dataset, seed)
    if not path.is_file():
        raise SystemExit(f"missing jobs {path}")
    return sum(1 for line in path.open() if line.strip())


def finished_count(model: str, dataset: str, seed: int, rho: str) -> int:
    path = out_path(model, dataset, seed, rho)
    found: set[str] = set()
    if not path.is_file():
        return 0
    target = float(rho)
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("bias_schedule") != "count":
            continue
        if row.get("bias_formula") != "one_minus_rho_pow_n":
            continue
        if abs(float(row.get("bias_rho") or -1) - target) > 1e-9:
            continue
        if row.get("status") in {"ok", "too_long"} and row.get("uid"):
            found.add(str(row["uid"]))
    return len(found)


def cell_done(model: str, dataset: str, seed: int, rho: str, expected: int) -> bool:
    return finished_count(model, dataset, seed, rho) >= expected


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
        model = arg(cmd, "--model-tag")
        dataset = arg(cmd, "--dataset")
        seed_text = arg(cmd, "--seed")
        if model not in MODELS or dataset not in DATASETS or seed_text is None:
            continue
        seed = int(seed_text)
        rho = arg(cmd, "--bias-rho")
        if rho is None or (model, dataset, seed, rho) not in set(CELLS):
            continue
        raw_gpu = environ(pid).get("CUDA_VISIBLE_DEVICES", "")
        gpu = nvidia_index(raw_gpu)
        found[key(model, dataset, seed, rho)] = {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "rho": rho,
            "pid": pid,
            "gpu": gpu,
            "log": None,
            "adopted": True,
        }
    return found


def gpu_maps() -> tuple[dict[str, str], dict[str, str]]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    )
    uuid_of: dict[str, str] = {}
    index_of: dict[str, str] = {}
    for line in text.splitlines():
        index, uuid = [item.strip() for item in line.split(",", 1)]
        uuid_of[index] = uuid
        index_of[uuid] = index
    return uuid_of, index_of


UUID_OF, INDEX_OF = gpu_maps()


def cuda_visible(gpu: str) -> str:
    parts = []
    for item in gpu.split(","):
        item = item.strip()
        if not item:
            continue
        parts.append(item if item.startswith("GPU-") else UUID_OF[item])
    return ",".join(parts)


def nvidia_index(gpu: str) -> str:
    parts = []
    for item in gpu.split(","):
        item = item.strip()
        if not item:
            continue
        parts.append(INDEX_OF.get(item, item))
    return ",".join(parts)


def gpu_used() -> dict[str, int]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used: dict[str, int] = {}
    for line in text.splitlines():
        index, memory, util = [item.strip() for item in line.split(",")]
        # A card in "GPU requires reset" reports 0 MiB and util [N/A].
        # Treating that as free makes the next launch slide onto another card.
        if util == "[N/A]":
            used[index] = 10**9
        else:
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
        "models": MODELS,
        "seeds": SEEDS,
        "datasets": DATASETS,
        "rhos": RHOS,
        "gpus": GPUS,
        "pending": pending,
        "done": done,
        "running": [
            {
                "cell": name,
                "model": item["model"],
                "dataset": item["dataset"],
                "seed": item["seed"],
                "rho": item["rho"],
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


def launch(model: str, dataset: str, seed: int, rho: str, gpu: str, attempt: int) -> dict:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{key(model, dataset, seed, rho)}.attempt_{attempt}.log"
    out = out_path(model, dataset, seed, rho)
    cuda = cuda_visible(gpu)
    with log_path.open("ab") as handle:
        handle.write(
            f"\n# {time.strftime('%Y-%m-%dT%H:%M:%S%z')} start gpus={gpu} cuda={cuda} rho={rho}\n".encode()
        )
        process = subprocess.Popen(
            ["bash", str(ROOT / "scripts/run_count_bias_cell.sh")],
            cwd=ROOT,
            env={
                **os.environ,
                "PLWS_ROOT": str(ROOT),
                "MODEL_TAG": model,
                "DATASET": dataset,
                "SEED": str(seed),
                "RHO": rho,
                "OUT": str(out),
                "GPU": cuda,
                "BATCH_SIZE": BATCH.get(model, "0"),
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(
        f"start {model} {dataset} seed={seed} rho={rho} gpu={gpu} pid={process.pid} attempt={attempt}",
        flush=True,
    )
    return {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "rho": rho,
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

expected = {
    (model, dataset, seed, rho): job_count(model, dataset, seed)
    for model, dataset, seed, rho in CELLS
}
running = discover()
attempts = {key(model, dataset, seed, rho): 0 for model, dataset, seed, rho in CELLS}
for name, item in running.items():
    print(f"adopt {name} gpu={item['gpu']} pid={item['pid']}", flush=True)
done: list[str] = []

while True:
    if stopping:
        pending_now = [
            key(model, dataset, seed, rho)
            for model, dataset, seed, rho in CELLS
            if key(model, dataset, seed, rho) not in running
            and key(model, dataset, seed, rho) not in done
        ]
        write_status(running, pending_now, done)
        raise SystemExit(130)
    for name, item in list(running.items()):
        if alive(item["pid"]):
            continue
        del running[name]
        model, dataset, seed, rho = (
            item["model"],
            item["dataset"],
            item["seed"],
            item["rho"],
        )
        if cell_done(model, dataset, seed, rho, expected[(model, dataset, seed, rho)]):
            done.append(name)
            print(f"done {name}", flush=True)
        else:
            print(f"retry {name}", flush=True)
    pending: list[tuple[str, str, int, str]] = []
    for model, dataset, seed, rho in CELLS:
        name = key(model, dataset, seed, rho)
        if name in running or name in done:
            continue
        if cell_done(model, dataset, seed, rho, expected[(model, dataset, seed, rho)]):
            done.append(name)
            continue
        pending.append((model, dataset, seed, rho))
    claimed: set[str] = set()
    for item in running.values():
        for gpu in str(item["gpu"]).split(","):
            gpu = gpu.strip()
            if gpu:
                claimed.add(gpu)
    used = gpu_used()
    idle = [gpu for gpu in GPUS if gpu not in claimed and used.get(gpu, 10**9) < 800]
    still_loading = [
        name
        for name, item in running.items()
        if not item["adopted"] and not log_loaded(item["log"])
    ]
    if pending and not still_loading and not workers_loading():
        model, dataset, seed, rho = pending[0]
        need = TP.get(model, 1)
        if len(idle) >= need:
            name = key(model, dataset, seed, rho)
            attempts[name] += 1
            running[name] = launch(
                model,
                dataset,
                seed,
                rho,
                ",".join(idle[:need]),
                attempts[name],
            )
    write_status(
        running,
        [key(model, dataset, seed, rho) for model, dataset, seed, rho in pending],
        done,
    )
    if not pending and not running:
        print("queue finished", flush=True)
        raise SystemExit(0)
    time.sleep(5)
PY
