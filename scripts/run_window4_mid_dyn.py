#!/usr/bin/env python3
"""8B + 14B 窗后压四种子：密探齐一块就压，空卡立刻接下一段。

只占 0–3。6 留给 s1，不碰 7。14B 密探 TP=2。只认 seed_{N}/，不用 seed-42 平铺。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
from score_leftover_jump import done_uids  # noqa: E402

PY = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python")
VENV = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
LOG = AE / "results/window4_mid/logs"
JOBSNAP = AE / "results/window4_mid/jobs"
GPUS = tuple(int(x) for x in os.environ.get("WIN4_GPUS", "0,1,2,3").split(",") if x.strip())
SEEDS = (0, 1, 123)
DATASETS = ("math-500", "gpqa-diamond", "olympiadbench")
ALWAYS_DS = ("aime24", "aime25")
KINDS = ("low", "high", "mix")
MODELS = (
    {"tag": "nemotron_8b", "tp": 1},
    {"tag": "r1_14b", "tp": 2},
)


def nvidia_lib_path() -> str:
    extra = ":".join(
        sorted(str(path) for path in (VENV / ".venv" / "lib").glob("**/nvidia/*/lib") if path.is_dir())
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{extra}:{current}" if extra else current


def trial_out(tag: str, ds: str, seed: int) -> Path:
    return AE / f"results/dense_G_{tag}/{ds}/seed_{seed}/dense_puma/trial_answers.json"


def g_out(tag: str, ds: str, seed: int) -> Path:
    return AE / f"results/dense_G_{tag}/{ds}/seed_{seed}/per_sample.json"


def dense_done(tag: str, ds: str, seed: int) -> bool:
    return trial_out(tag, ds, seed).is_file() and g_out(tag, ds, seed).is_file()


def jobs_path(tag: str, seed: int, kind: str) -> Path:
    folder = AE / f"results/leftover_jump/{tag}_s{seed}"
    return folder / ("jobs.jsonl" if kind == "low" else f"jobs_{kind}.jsonl")


def score_dir(tag: str, seed: int, kind: str) -> Path:
    suf = "" if kind == "low" else f"_{kind}"
    return AE / "results/leftover_suppress_toend" / f"{tag}_s{seed}_suppress{suf}"


def n_jobs(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def iter_cmds() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
        except OSError:
            continue
        if raw:
            out.append((int(entry.name), raw.decode("utf-8", "replace")))
    return out


def dense_running(tag: str, ds: str, seed: int) -> bool:
    seed_tok = f"seed_{seed}"
    model_hint = "Nemotron" if tag == "nemotron_8b" else "Distill-Qwen-14B"
    for _, text in iter_cmds():
        if "gen_trial_answers.py" in text and ds in text and seed_tok in text:
            if tag == "nemotron_8b" and "Nemotron" not in text and "Nano-8B" not in text:
                continue
            if tag == "r1_14b" and "14B" not in text:
                continue
            return True
        if "compute_dense_G.py" in text and ds in text and seed_tok in text and tag in text:
            return True
        if "run_dense_trials_model.sh" in text and f"DATASET={ds}" in text and f"SEED={seed}" in text:
            if tag in text or model_hint in text:
                return True
    return False


def dense_prio(tag: str, ds: str) -> int:
    hard = 2 if ds == "olympiadbench" else 0
    big = 1 if tag == "r1_14b" else 0
    return hard * 10 + big


def ready_datasets(tag: str) -> list[str]:
    ready = list(ALWAYS_DS)
    for ds in DATASETS:
        if all(dense_done(tag, ds, seed) for seed in SEEDS):
            ready.append(ds)
    return ready


def leftover_has_datasets(tag: str, datasets: list[str]) -> bool:
    need = set(datasets)
    for seed in SEEDS:
        path = jobs_path(tag, seed, "low")
        if not path.is_file():
            return False
        have: set[str] = set()
        for line in path.open():
            if not line.strip():
                continue
            have.add(str(json.loads(line)["dataset"]))
        if not need <= have:
            return False
    return True


def export_jobs(tag: str, datasets: list[str]) -> None:
    if leftover_has_datasets(tag, datasets):
        print(f"skip export {tag} already {datasets}", flush=True)
        return
    print(f"export leftover jobs {tag} {datasets}", flush=True)
    subprocess.run(
        [
            str(PY),
            str(AE / "scripts/export_leftover_suppress_jobs.py"),
            "--model-tag",
            tag,
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


def _dense_env(gpus: list[int], cell: dict) -> dict[str, str]:
    env = os.environ.copy()
    env["MODEL_TAG"] = cell["tag"]
    env["DATASET"] = cell["ds"]
    env["GPUS"] = ",".join(str(g) for g in gpus)
    env["SEED"] = str(cell["seed"])
    env["TP"] = str(cell["tp"])
    env["VLLM_LENS_DISABLE"] = "1"
    env["LD_LIBRARY_PATH"] = nvidia_lib_path()
    return env


def run_dense_gpu(gpus: list[int], cell: dict) -> int:
    tag, ds, seed, tp = cell["tag"], cell["ds"], cell["seed"], cell["tp"]
    log = LOG / f"dense_{tag}_{ds}_s{seed}_gpu{gpus[0]}.log"
    env = _dense_env(gpus, cell)
    env["DENSE_GPU_ONLY"] = "1"
    print(f"gpu{','.join(map(str, gpus))} dense-gpu {tag} {ds} s{seed} tp={tp} -> {log}", flush=True)
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
        f"gpu{','.join(map(str, gpus))} dense-gpu {tag} {ds} s{seed} "
        f"exit={proc.returncode} {time.time() - started:.0f}s",
        flush=True,
    )
    return proc.returncode


def run_dense_g(cell: dict) -> int:
    tag, ds, seed = cell["tag"], cell["ds"], cell["seed"]
    out = trial_out(tag, ds, seed).parent
    dest = g_out(tag, ds, seed)
    log = LOG / f"dense_{tag}_{ds}_s{seed}_g.log"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"cpu dense-G {tag} {ds} s{seed} -> {log}", flush=True)
    started = time.time()
    with log.open("w") as handle:
        proc = subprocess.run(
            [
                str(PY),
                str(AE / "scripts/compute_dense_G.py"),
                "--dataset",
                ds,
                "--dense-root",
                str(out),
                "--out",
                str(dest),
                "--workers",
                "8",
            ],
            cwd=str(AE),
            env={**os.environ, "LD_LIBRARY_PATH": nvidia_lib_path()},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    print(
        f"cpu dense-G {tag} {ds} s{seed} exit={proc.returncode} {time.time() - started:.0f}s",
        flush=True,
    )
    return proc.returncode


def run_suppress(gpu: int, task: dict) -> int:
    tag = task["tag"]
    seed = task["seed"]
    kind = task["kind"]
    shard = task["shard"]
    nshard = task["nshard"]
    log = LOG / f"suppress_{tag}_s{seed}_{kind}_sh{shard}_g{task['gen']}.log"
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
    print(f"gpu{gpu} suppress {tag} s{seed} {kind} shard {shard}/{nshard} g{task['gen']} -> {log}", flush=True)
    started = time.time()
    with log.open("w") as handle:
        proc = subprocess.run(cmd, cwd=str(AE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print(
        f"gpu{gpu} suppress {tag} s{seed} {kind} shard {shard}/{nshard} "
        f"exit={proc.returncode} {time.time() - started:.0f}s",
        flush=True,
    )
    return proc.returncode


class Mixer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.free = set(GPUS)
        self.dense_pending: list[dict] = []
        self.suppress_pending: list[dict] = []
        self.queued_dense: set[tuple] = set()
        self.queued_suppress: set[tuple] = set()
        self.exported: dict[str, list[str]] = {}
        self.gen: dict[str, int] = {}
        self.exporting: set[str] = set()
        self.inflight = 0
        self.stats = {"dense_fail": 0, "suppress_fail": 0, "done": 0}

    def offer_dense(self) -> None:
        cells = []
        for spec in MODELS:
            tag, tp = spec["tag"], spec["tp"]
            for seed in SEEDS:
                for ds in DATASETS:
                    key = (tag, ds, seed)
                    if key in self.queued_dense or dense_done(tag, ds, seed) or dense_running(tag, ds, seed):
                        continue
                    cells.append(
                        {
                            "kind_task": "dense",
                            "tag": tag,
                            "ds": ds,
                            "seed": seed,
                            "tp": tp,
                            "prio": dense_prio(tag, ds),
                        }
                    )
                    self.queued_dense.add(key)
                    print(f"queue dense {tag} {ds} s{seed} tp={tp}", flush=True)
        cells.sort(key=lambda c: (c["prio"], c["tag"], c["ds"], c["seed"]))
        self.dense_pending.extend(cells)

    def offer_suppress(self) -> None:
        for spec in MODELS:
            tag = spec["tag"]
            ready = ready_datasets(tag)
            prev = self.exported.get(tag, [])
            if ready == prev or (set(ready) <= set(prev) and prev):
                continue
            if tag in self.exporting:
                continue
            self.exporting.add(tag)
            planned = list(ready)
            try:
                export_jobs(tag, planned)
                self.gen[tag] = self.gen.get(tag, 0) + 1
                gen = self.gen[tag]
                JOBSNAP.mkdir(parents=True, exist_ok=True)
                for seed in SEEDS:
                    for kind in KINDS:
                        src = jobs_path(tag, seed, kind)
                        if not src.is_file():
                            continue
                        snap = JOBSNAP / f"{tag}_s{seed}_{kind}_g{gen}.jsonl"
                        shutil.copy2(src, snap)
                        n = n_jobs(snap)
                        if not n:
                            continue
                        already: set[str] = set()
                        folder = score_dir(tag, seed, kind)
                        if folder.is_dir():
                            for path in folder.glob("scores_shard*.jsonl"):
                                already |= done_uids(path)
                        uids = []
                        for line in snap.read_text().splitlines():
                            if line.strip():
                                uids.append(json.loads(line)["uid"])
                        pending = sum(1 for uid in uids if uid not in already)
                        if pending <= 0:
                            print(f"skip suppress {tag} s{seed} {kind} done {n}", flush=True)
                            continue
                        shards = 6 if n >= 200 else (2 if n >= 20 else 1)
                        for shard in range(shards):
                            key = (tag, seed, kind, shard, gen)
                            if key in self.queued_suppress:
                                continue
                            self.queued_suppress.add(key)
                            self.suppress_pending.append(
                                {
                                    "kind_task": "suppress",
                                    "tag": tag,
                                    "seed": seed,
                                    "kind": kind,
                                    "jobs": snap,
                                    "n": n,
                                    "nshard": shards,
                                    "shard": shard,
                                    "pending": pending,
                                    "gen": gen,
                                }
                            )
                        print(
                            f"queue suppress {tag} s{seed} {kind} n={n} pending={pending} shards={shards} g{gen}",
                            flush=True,
                        )
                self.exported[tag] = planned
            finally:
                self.exporting.discard(tag)

    def take_gpus(self, n: int) -> list[int] | None:
        with self.lock:
            if len(self.free) < n:
                return None
            chosen = sorted(self.free)[:n]
            for gpu in chosen:
                self.free.remove(gpu)
            return chosen

    def release_gpus(self, gpus: list[int]) -> None:
        with self.lock:
            self.free.update(gpus)

    def pop_dense(self, nfree: int) -> dict | None:
        best_i = None
        for i, cell in enumerate(self.dense_pending):
            if cell["tp"] <= nfree:
                best_i = i
                break
        if best_i is None:
            return None
        return self.dense_pending.pop(best_i)

    def pop_suppress(self) -> dict | None:
        if not self.suppress_pending:
            return None
        return self.suppress_pending.pop(0)

    def finished(self) -> bool:
        dense_all = all(
            dense_done(spec["tag"], ds, seed)
            for spec in MODELS
            for ds in DATASETS
            for seed in SEEDS
        )
        exported_all = all(
            set(self.exported.get(spec["tag"], [])) >= set(ALWAYS_DS) | set(DATASETS)
            for spec in MODELS
        )
        return (
            dense_all
            and exported_all
            and not self.dense_pending
            and not self.suppress_pending
            and self.inflight == 0
        )

    def spawn(self, task: dict, gpus: list[int]) -> None:
        with self.lock:
            self.inflight += 1

        def _run() -> None:
            held = list(gpus)
            try:
                if task["kind_task"] == "dense":
                    tag, ds, seed = task["tag"], task["ds"], task["seed"]
                    code = 0
                    trials = trial_out(tag, ds, seed)
                    if dense_done(tag, ds, seed):
                        print(f"skip dense {tag} {ds} s{seed}", flush=True)
                    elif not trials.is_file() and dense_running(tag, ds, seed):
                        print(f"skip dense-gpu running {tag} {ds} s{seed}", flush=True)
                    else:
                        if not trials.is_file():
                            code = run_dense_gpu(held, task)
                        if held:
                            self.release_gpus(held)
                            held = []
                            print(f"release after dense-gpu {tag} {ds} s{seed}", flush=True)
                        if code == 0 and not g_out(tag, ds, seed).is_file():
                            code = run_dense_g(task)
                        if code != 0:
                            with self.lock:
                                self.stats["dense_fail"] += 1
                else:
                    code = run_suppress(held[0], task)
                    if code != 0:
                        with self.lock:
                            self.stats["suppress_fail"] += 1
            finally:
                if held:
                    self.release_gpus(held)
                with self.lock:
                    self.inflight -= 1
                    self.stats["done"] += 1
                    print(
                        f"queue done={self.stats['done']} inflight={self.inflight} "
                        f"free={sorted(self.free)} fail_d={self.stats['dense_fail']} "
                        f"fail_s={self.stats['suppress_fail']}",
                        flush=True,
                    )

        threading.Thread(target=_run, daemon=True).start()

    def dispatch_once(self) -> bool:
        with self.lock:
            nfree = len(self.free)
        cell = self.pop_dense(nfree)
        if cell is not None:
            gpus = self.take_gpus(cell["tp"])
            if gpus is None:
                self.dense_pending.insert(0, cell)
            else:
                self.spawn(cell, gpus)
                return True
        with self.lock:
            nfree = len(self.free)
        if nfree >= 1:
            task = self.pop_suppress()
            if task is not None:
                gpus = self.take_gpus(1)
                if gpus is None:
                    self.suppress_pending.insert(0, task)
                else:
                    self.spawn(task, gpus)
                    return True
        return False


def main() -> None:
    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_path()
    LOG.mkdir(parents=True, exist_ok=True)
    JOBSNAP.mkdir(parents=True, exist_ok=True)
    print(f"mixer 8b+14b gpus {list(GPUS)}", flush=True)
    mix = Mixer()
    mix.offer_dense()
    mix.offer_suppress()
    while not mix.finished():
        mix.offer_dense()
        mix.offer_suppress()
        if not mix.dispatch_once():
            time.sleep(4)
    print(
        f"window4 8b+14b done fail_d={mix.stats['dense_fail']} fail_s={mix.stats['suppress_fail']}",
        flush=True,
    )
    if mix.stats["dense_fail"] or mix.stats["suppress_fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
