#!/usr/bin/env python3
"""6 卡混跑：密探齐一块就导出并压，空卡立刻接下一段。

不碰 GPU 6（s1）和 GPU 7。只认 seed_{N}/ 下的密探，不用 seed-42 平铺顶替。
占用显存的卡不抢；有别人在打的密探不重开。
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
from score_leftover_jump import done_uids  # noqa: E402

PY = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python")
VENV = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
LOG = AE / "results/window4_7b/logs"
JOBSNAP = AE / "results/window4_7b/jobs"
GPUS = tuple(int(x) for x in os.environ.get("WIN4_GPUS", "0,1,2,3,4,5").split(",") if x.strip())
TAG = "r1_7b"
SEEDS = (0, 1, 123)
DATASETS = ("math-500", "gpqa-diamond", "olympiadbench")
ALWAYS_DS = ("aime24", "aime25")
KINDS = ("low", "high", "mix")
BUSY_MIB = 2048


def nvidia_lib_path() -> str:
    extra = ":".join(
        sorted(str(path) for path in (VENV / ".venv" / "lib").glob("**/nvidia/*/lib") if path.is_dir())
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


def trial_out(ds: str, seed: int) -> Path:
    return AE / f"results/dense_G_{TAG}/{ds}/seed_{seed}/dense_puma/trial_answers.json"


def g_out(ds: str, seed: int) -> Path:
    return AE / f"results/dense_G_{TAG}/{ds}/seed_{seed}/per_sample.json"


def dense_done(ds: str, seed: int) -> bool:
    return trial_out(ds, seed).is_file() and g_out(ds, seed).is_file()


def jobs_path(seed: int, kind: str) -> Path:
    folder = AE / f"results/leftover_jump/{TAG}_s{seed}"
    return folder / ("jobs.jsonl" if kind == "low" else f"jobs_{kind}.jsonl")


def score_dir(seed: int, kind: str) -> Path:
    suf = "" if kind == "low" else f"_{kind}"
    return AE / "results/leftover_suppress_toend" / f"{TAG}_s{seed}_suppress{suf}"


def n_jobs(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def iter_cmds() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
        except OSError:
            continue
        if raw:
            out.append((int(entry.name), raw.decode("utf-8", "replace")))
    return out


def cmd_has(text: str, *need: str) -> bool:
    return all(token in text for token in need)


def dense_running(ds: str, seed: int) -> bool:
    seed_tok = f"seed_{seed}"
    for _, text in iter_cmds():
        if "gen_trial_answers.py" in text and ds in text and seed_tok in text:
            return True
        if "compute_dense_G.py" in text and ds in text and seed_tok in text:
            return True
        if "run_dense_trials_model.sh" in text and f"DATASET={ds}" in text and f"SEED={seed}" in text:
            return True
    return False


def gpu_mem() -> dict[int, int]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
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


def kill_legacy() -> None:
    my = os.getpid()
    for pid, text in iter_cmds():
        if pid == my:
            continue
        if "run_window4_7b_dyn6.py" not in text:
            continue
        if "python" not in text:
            continue
        print(f"stop legacy dyn6 pid={pid}", flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            print(f"legacy {pid} {exc}", flush=True)
    time.sleep(2)


def ready_datasets() -> list[str]:
    ready = list(ALWAYS_DS)
    for ds in DATASETS:
        if all(dense_done(ds, seed) for seed in SEEDS):
            ready.append(ds)
    return ready


def export_jobs(datasets: list[str]) -> None:
    print(f"export leftover jobs {datasets}", flush=True)
    subprocess.run(
        [
            str(PY),
            str(AE / "scripts/export_leftover_suppress_jobs.py"),
            "--model-tag",
            TAG,
            "--kinds",
            "all",
            "--seeds",
            "0,1,123",
            "--datasets",
            ",".join(datasets),
        ],
        cwd=str(AE),
        check=True,
    )


def run_dense(gpu: int, cell: dict) -> int:
    ds = cell["ds"]
    seed = cell["seed"]
    log = LOG / f"dense_{ds}_s{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    env["MODEL_TAG"] = TAG
    env["DATASET"] = ds
    env["GPUS"] = str(gpu)
    env["SEED"] = str(seed)
    env["TP"] = "1"
    env["VLLM_LENS_DISABLE"] = "1"
    env["LD_LIBRARY_PATH"] = nvidia_lib_path()
    print(f"gpu{gpu} dense {ds} s{seed} -> {log}", flush=True)
    started = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            ["bash", str(AE / "scripts/run_dense_trials_model.sh")],
            cwd=str(AE),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    print(
        f"gpu{gpu} dense {ds} s{seed} exit={proc.returncode} {time.time() - started:.0f}s",
        flush=True,
    )
    return proc.returncode


def run_suppress(gpu: int, task: dict) -> int:
    seed = task["seed"]
    kind = task["kind"]
    shard = task["shard"]
    nshard = task["nshard"]
    log = LOG / f"suppress_s{seed}_{kind}_sh{shard}_g{task['gen']}.log"
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
        TAG,
        "--lexicon",
        "core",
        "--seed",
        str(seed),
        "--run-kind",
        kind,
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
        str(AE / "results/leftover_suppress_toend"),
    ]
    print(f"gpu{gpu} suppress s{seed} {kind} shard {shard}/{nshard} g{task['gen']} -> {log}", flush=True)
    started = time.time()
    with log.open("w") as handle:
        proc = subprocess.run(cmd, cwd=str(AE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print(
        f"gpu{gpu} suppress s{seed} {kind} shard {shard}/{nshard} "
        f"exit={proc.returncode} {time.time() - started:.0f}s",
        flush=True,
    )
    return proc.returncode


class Mixer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dense_q: Queue = Queue()
        self.suppress_q: Queue = Queue()
        self.queued_dense: set[tuple[str, int]] = set()
        self.queued_suppress: set[tuple] = set()
        self.exported: list[str] = []
        self.gen = 0
        self.exporting = False
        self.stats = {"dense_fail": 0, "suppress_fail": 0, "done": 0}
        self.inflight = 0
        self.stop = False
        self.claimed: set[int] = set()

    def offer_dense(self) -> None:
        for seed in SEEDS:
            for ds in DATASETS:
                key = (ds, seed)
                with self.lock:
                    if key in self.queued_dense or dense_done(ds, seed) or dense_running(ds, seed):
                        continue
                    self.queued_dense.add(key)
                    self.dense_q.put({"kind_task": "dense", "ds": ds, "seed": seed})
                    print(f"queue dense {ds} s{seed}", flush=True)

    def offer_suppress(self) -> None:
        with self.lock:
            ready = ready_datasets()
            if ready == self.exported or (set(ready) <= set(self.exported) and self.exported):
                return
            if self.exporting:
                return
            self.exporting = True
            planned = list(ready)
        try:
            export_jobs(planned)
            self.gen += 1
            JOBSNAP.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                for kind in KINDS:
                    src = jobs_path(seed, kind)
                    if not src.is_file():
                        continue
                    snap = JOBSNAP / f"{TAG}_s{seed}_{kind}_g{self.gen}.jsonl"
                    shutil.copy2(src, snap)
                    n = n_jobs(snap)
                    if not n:
                        continue
                    already: set[str] = set()
                    folder = score_dir(seed, kind)
                    if folder.is_dir():
                        for path in folder.glob("scores_shard*.jsonl"):
                            already |= done_uids(path)
                    pending = n - len(already)
                    if pending <= 0:
                        print(f"skip suppress s{seed} {kind} done {n}", flush=True)
                        continue
                    shards = 6 if n >= 200 else (2 if n >= 20 else 1)
                    for shard in range(shards):
                        key = (seed, kind, shard, self.gen)
                        if key in self.queued_suppress:
                            continue
                        self.queued_suppress.add(key)
                        self.suppress_q.put(
                            {
                                "kind_task": "suppress",
                                "seed": seed,
                                "kind": kind,
                                "jobs": snap,
                                "n": n,
                                "nshard": shards,
                                "shard": shard,
                                "pending": pending,
                                "gen": self.gen,
                            }
                        )
                    print(
                        f"queue suppress s{seed} {kind} n={n} pending={pending} shards={shards} g{self.gen}",
                        flush=True,
                    )
            with self.lock:
                self.exported = planned
        finally:
            with self.lock:
                self.exporting = False

    def take(self) -> dict | None:
        try:
            return self.dense_q.get_nowait()
        except Empty:
            pass
        try:
            return self.suppress_q.get_nowait()
        except Empty:
            return None

    def finished(self) -> bool:
        return (
            all(dense_done(ds, seed) for ds in DATASETS for seed in SEEDS)
            and self.dense_q.empty()
            and self.suppress_q.empty()
            and self.inflight == 0
            and set(self.exported) >= set(ALWAYS_DS) | set(DATASETS)
        )

    def worker(self, gpu: int) -> None:
        print(f"worker gpu{gpu} start", flush=True)
        while not self.stop:
            self.offer_dense()
            self.offer_suppress()
            task = self.take()
            if task is None:
                if self.finished():
                    print(f"worker gpu{gpu} idle-exit", flush=True)
                    return
                time.sleep(4)
                continue
            with self.lock:
                self.inflight += 1
            try:
                if task["kind_task"] == "dense":
                    if dense_done(task["ds"], task["seed"]) or dense_running(task["ds"], task["seed"]):
                        print(f"gpu{gpu} skip dense {task['ds']} s{task['seed']}", flush=True)
                        code = 0
                    else:
                        code = run_dense(gpu, task)
                        if code != 0:
                            with self.lock:
                                self.stats["dense_fail"] += 1
                else:
                    code = run_suppress(gpu, task)
                    if code != 0:
                        with self.lock:
                            self.stats["suppress_fail"] += 1
            finally:
                with self.lock:
                    self.inflight -= 1
                    self.stats["done"] += 1
                    print(
                        f"queue done={self.stats['done']} inflight={self.inflight} "
                        f"fail_d={self.stats['dense_fail']} fail_s={self.stats['suppress_fail']}",
                        flush=True,
                    )
        print(f"worker gpu{gpu} stop", flush=True)

    def claim_gpus(self) -> None:
        used = gpu_mem()
        for gpu in GPUS:
            if gpu in self.claimed:
                continue
            if used.get(gpu, 0) >= BUSY_MIB:
                continue
            self.claimed.add(gpu)
            t = threading.Thread(target=self.worker, args=(gpu,), name=f"gpu{gpu}", daemon=True)
            t.start()
            print(f"claim gpu{gpu} mem={used.get(gpu, 0)}MiB", flush=True)


def main() -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    LOG.mkdir(parents=True, exist_ok=True)
    JOBSNAP.mkdir(parents=True, exist_ok=True)
    print(f"mixer gpus {list(GPUS)}", flush=True)
    kill_legacy()
    mix = Mixer()
    mix.offer_dense()
    mix.offer_suppress()
    mix.claim_gpus()
    while not mix.finished():
        mix.offer_dense()
        mix.offer_suppress()
        mix.claim_gpus()
        time.sleep(5)
    mix.stop = True
    time.sleep(1)
    print(
        f"window4 7b done fail_d={mix.stats['dense_fail']} fail_s={mix.stats['suppress_fail']}",
        flush=True,
    )
    if mix.stats["dense_fail"] or mix.stats["suppress_fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
