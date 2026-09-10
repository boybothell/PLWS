#!/usr/bin/env python3
"""Dynamic 4-GPU scheduler for remaining H-A / H-B / H-C extracts.

Keeps the existing 4-way question shard map.  Idle GPUs pick the next
unfinished shard instead of waiting for a whole dataset/mode to finish.
CPU-only work (tokenizer gate, merge, analysis) does not occupy a GPU.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

AE = Path(__file__).resolve().parents[1]

VLLM_ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
PYTHON = VLLM_ROOT / ".venv/bin/python"
UNIVERSAL = AE / "scripts/score_confcal_universal.py"
KEYTOKEN = AE / "scripts/score_confcal_keytoken.py"
ANALYZE = AE / "scripts/analyze_confcal_v2.py"
ANALYZE_FALLBACK = AE / "scripts/analyze_confcal_v2_available.py"
SMALL = Path("/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-1.5B")
LOG_ROOT = AE / "results/_logs"
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond")
MODES = ("base", "instruct")
NUM_SHARDS = 4
DEFAULT_GPUS = "1,4,6,7"


def libraries() -> str:
    extra = ":".join(sorted(str(path) for path in (VLLM_ROOT / ".venv/lib").glob("**/nvidia/*/lib") if path.is_dir()))
    return f"{extra}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def dense_trial_path(dataset: str, seed: int = 42) -> Path:
    root = AE / f"results/dense_G_r1_7b/{dataset}"
    for path in (
        root / "dense_puma" / "trial_answers.json",
        root / f"seed_{seed}" / "dense_puma" / "trial_answers.json",
        root / f"s{seed}" / "dense_puma" / "trial_answers.json",
    ):
        if path.exists():
            return path
    raise FileNotFoundError(root)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def log_complete(log_path: Path, expected: int) -> bool:
    if expected <= 0 or not log_path.exists():
        return False
    with log_path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(handle.tell() - 8192, 0))
        tail = handle.read().decode(errors="replace")
    return f"[{expected}/{expected}]" in tail


def expected_shard_counts(dataset: str, seed: int = 42) -> list[int]:
    """Count trials per 4-way question shard without parsing reasoning text."""
    counts_by_qi: Counter[int] = Counter()
    leftover = b""
    pattern = re.compile(rb'"question_idx"\s*:\s*(\d+)')
    with dense_trial_path(dataset, seed).open("rb") as handle:
        while True:
            chunk = handle.read(8_000_000)
            if not chunk:
                break
            data = leftover + chunk
            keep = data[-40:]
            for match in pattern.finditer(data[:-40] if len(data) > 40 else data):
                counts_by_qi[int(match.group(1))] += 1
            leftover = keep
    for match in pattern.finditer(leftover):
        counts_by_qi[int(match.group(1))] += 1
    qids = sorted(counts_by_qi)
    shard_of = {qi: index % NUM_SHARDS for index, qi in enumerate(qids)}
    counts = [0] * NUM_SHARDS
    for qi, n_jobs in counts_by_qi.items():
        counts[shard_of[qi]] += n_jobs
    return counts


@dataclass
class Job:
    name: str
    kind: str
    dataset: str
    argv: list[str]
    out_path: Path
    log_path: Path
    expected: int
    priority: int
    needs_gpu: bool = True
    depends: tuple[str, ...] = ()
    proc: subprocess.Popen | None = field(default=None, repr=False)
    gpu: str | None = None
    retries: int = 0
    state: str = "pending"  # pending|running|done|failed

    def remaining(self) -> int:
        if self.kind in {"analyze", "gate", "merge"}:
            return 0 if self.is_complete() else 1
        if log_complete(self.log_path, self.expected):
            return 0
        if not self.out_path.exists():
            return self.expected
        # Huge keytoken dumps are not scanned; an unfinished phase log means work remains.
        if self.out_path.stat().st_size > 5_000_000:
            return 1
        return max(self.expected - line_count(self.out_path), 0)

    def is_complete(self) -> bool:
        if self.state == "done":
            return True
        if self.kind == "analyze":
            return False
        if self.kind == "gate":
            return self.out_path.exists() and "passed" in self.out_path.read_text()
        if self.kind == "merge":
            shard = self.out_path.name.split("shard")[-1].split(".")[0]
            parent = self.out_path.parent
            solver_log = LOG_ROOT / "confcal_keytoken" / f"{self.dataset}_s42_solver" / f"shard{shard}.log"
            small_log = LOG_ROOT / "confcal_keytoken" / f"{self.dataset}_s42_small" / f"shard{shard}.log"
            parents_done = log_complete(solver_log, self.expected) and log_complete(small_log, self.expected)
            return parents_done and self.out_path.exists()
        return self.remaining() <= 0

    def runnable(self, done: set[str]) -> bool:
        return self.state == "pending" and not self.is_complete() and all(name in done for name in self.depends)


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    expected: dict[str, list[int]] = {dataset: expected_shard_counts(dataset) for dataset in DATASETS}
    for dataset in DATASETS:
        ds_rank = DATASETS.index(dataset)
        for mode in MODES:
            mode_rank = 0 if mode == "base" else 1
            for shard in range(NUM_SHARDS):
                out = AE / "results/confcal_judge/v2/universal_v2b" / f"{dataset}_s42_{mode}" / f"scores_shard{shard}.jsonl"
                log = LOG_ROOT / "confcal_universal" / f"{dataset}_s42_{mode}" / f"shard{shard}.log"
                jobs.append(
                    Job(
                        name=f"hbhc:{dataset}:{mode}:{shard}",
                        kind="universal",
                        dataset=dataset,
                        argv=[
                            str(PYTHON), str(UNIVERSAL), "--dataset", dataset, "--mode", mode,
                            "--shard-id", str(shard), "--num-shards", str(NUM_SHARDS),
                        ],
                        out_path=out,
                        log_path=log,
                        expected=expected[dataset][shard],
                        priority=20 + 10 * ds_rank + 5 * mode_rank,
                    )
                )
        gate_out = AE / "results/confcal_judge/v2/keytoken" / f"{dataset}_s42_tokenizer_gate.json"
        gate_name = f"ha:gate:{dataset}"
        jobs.append(
            Job(
                name=gate_name,
                kind="gate",
                dataset=dataset,
                argv=[str(PYTHON), str(KEYTOKEN), "--phase", "gate", "--dataset", dataset, "--small-model", str(SMALL)],
                out_path=gate_out,
                log_path=LOG_ROOT / "confcal_keytoken" / f"{dataset}_s42_gate" / "gate.log",
                expected=1,
                        priority=5 + 2 * ds_rank,
                needs_gpu=False,
            )
        )
        for phase, phase_rank in (("solver", 0), ("small", 1), ("merge", 2)):
            for shard in range(NUM_SHARDS):
                out_dir = AE / "results/confcal_judge/v2/keytoken" / f"{dataset}_s42"
                prefix = "scores" if phase == "merge" else phase
                depends: tuple[str, ...] = (gate_name,)
                if phase == "merge":
                    depends = (f"ha:solver:{dataset}:{shard}", f"ha:small:{dataset}:{shard}")
                jobs.append(
                    Job(
                        name=f"ha:{phase}:{dataset}:{shard}",
                        kind=phase,
                        dataset=dataset,
                        argv=[
                            str(PYTHON), str(KEYTOKEN), "--phase", phase, "--dataset", dataset,
                            "--small-model", str(SMALL), "--shard-id", str(shard),
                            "--num-shards", str(NUM_SHARDS),
                        ],
                        out_path=out_dir / f"{prefix}_shard{shard}.jsonl",
                        log_path=LOG_ROOT / "confcal_keytoken" / f"{dataset}_s42_{phase}" / f"shard{shard}.log",
                        expected=expected[dataset][shard],
                        priority={"solver": 11, "small": 16, "merge": 90}[phase] + 2 * ds_rank,
                        needs_gpu=phase != "merge",
                        depends=depends,
                    )
                )
    jobs.append(
        Job(
            name="analyze",
            kind="analyze",
            dataset="all",
            argv=[sys.executable, str(ANALYZE if ANALYZE.exists() else ANALYZE_FALLBACK)],
            out_path=AE / "results/confcal_judge/v2/analysis.json",
            log_path=LOG_ROOT / "confcal_v2_analyze.log",
            expected=1,
            priority=1000,
            needs_gpu=False,
            depends=tuple(job.name for job in jobs),
        )
    )
    return jobs


def cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
        return [part.decode() for part in raw if part]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []


def environ_map(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\x00")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return {}
    out: dict[str, str] = {}
    for item in raw:
        if b"=" in item:
            key, value = item.decode(errors="replace").split("=", 1)
            out[key] = value
    return out


def is_serial_parent(cmd: list[str]) -> bool:
    if not cmd:
        return False
    if cmd[0].endswith("python") and any(part.endswith("run_confcal_universal_serial.py") for part in cmd):
        return True
    return len(cmd) >= 2 and cmd[0].endswith("bash") and cmd[1].endswith("run_confcal_v2_resume.sh")


def stop_serial_parents() -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        cmd = cmdline(int(entry.name))
        if is_serial_parent(cmd):
            print(f"stop serial parent pid={entry.name} cmd={' '.join(cmd[:6])}", flush=True)
            try:
                os.kill(int(entry.name), signal.SIGTERM)
            except ProcessLookupError:
                pass


def parse_job_name(cmd: list[str]) -> str | None:
    text = " ".join(cmd)
    if "score_confcal_universal.py" in text:
        dataset = mode = shard = None
        for index, part in enumerate(cmd):
            if part == "--dataset":
                dataset = cmd[index + 1]
            elif part == "--mode":
                mode = cmd[index + 1]
            elif part == "--shard-id":
                shard = cmd[index + 1]
        if dataset and mode and shard is not None:
            return f"hbhc:{dataset}:{mode}:{shard}"
    if "score_confcal_keytoken.py" in text:
        dataset = phase = shard = None
        for index, part in enumerate(cmd):
            if part == "--dataset":
                dataset = cmd[index + 1]
            elif part == "--phase":
                phase = cmd[index + 1]
            elif part == "--shard-id":
                shard = cmd[index + 1]
        if dataset and phase == "gate":
            return f"ha:gate:{dataset}"
        if dataset and phase and shard is not None:
            return f"ha:{phase}:{dataset}:{shard}"
    return None


@dataclass
class Adopted:
    pid: int


def adopt_running(jobs: dict[str, Job], gpus: list[str]) -> dict[str, Job]:
    busy: dict[str, Job] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = cmdline(pid)
        name = parse_job_name(cmd)
        if not name or name not in jobs:
            continue
        job = jobs[name]
        job.state = "running"
        job.proc = Adopted(pid)  # type: ignore[assignment]
        gpu = environ_map(pid).get("CUDA_VISIBLE_DEVICES")
        if gpu and gpu in gpus and job.needs_gpu:
            job.gpu = gpu
            busy[gpu] = job
            print(f"adopt {job.name} pid={pid} gpu={gpu} remaining={job.remaining()}", flush=True)
        else:
            print(f"adopt {job.name} pid={pid} cpu remaining={job.remaining()}", flush=True)
    return busy


def launch(job: Job, gpu: str | None) -> None:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "VLLM_LENS_DISABLE": "1", "LD_LIBRARY_PATH": libraries()}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
        job.gpu = gpu
    handle = job.log_path.open("a")
    handle.write(f"\n==== dynamic launch {time.strftime('%F %T')} gpu={gpu} ====\n")
    handle.flush()
    print(f"launch {job.name} gpu={gpu} remaining={job.remaining()}", flush=True)
    job.proc = subprocess.Popen(job.argv, cwd=AE, env=env, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    job.state = "running"


def alive(job: Job) -> bool:
    proc = job.proc
    if proc is None:
        return False
    if isinstance(proc, Adopted):
        return Path(f"/proc/{proc.pid}").exists()
    return proc.poll() is None


def exit_code(job: Job) -> int:
    proc = job.proc
    if proc is None:
        return 1
    if isinstance(proc, Adopted):
        return 0 if not Path(f"/proc/{proc.pid}").exists() else 1
    return int(proc.returncode or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--poll", type=float, default=20.0)
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) != NUM_SHARDS:
        raise ValueError(f"need {NUM_SHARDS} GPUs to keep the existing shard map, got {gpus}")
    print(f"planning jobs on gpus={gpus}", flush=True)
    stop_serial_parents()
    job_list = build_jobs()
    jobs = {job.name: job for job in job_list}
    for job in job_list:
        if job.kind != "analyze" and job.is_complete():
            job.state = "done"
    done = {job.name for job in job_list if job.state == "done"}
    print(f"already done: {len(done)} / {len(job_list)}", flush=True)
    busy = adopt_running(jobs, gpus)
    cpu_running: list[Job] = [job for job in job_list if job.state == "running" and not job.needs_gpu]
    failed: list[str] = []
    while True:
        for gpu, job in list(busy.items()):
            if not alive(job):
                leftover = job.remaining()
                code = exit_code(job)
                if leftover <= 0 and code == 0:
                    job.state = "done"
                    done.add(job.name)
                    print(f"complete {job.name} gpu={gpu}", flush=True)
                elif leftover <= 0:
                    job.state = "done"
                    done.add(job.name)
                    print(f"complete {job.name} gpu={gpu} leftover=0 code={code}", flush=True)
                elif job.retries < 1:
                    job.retries += 1
                    job.state = "pending"
                    job.proc = None
                    print(f"retry {job.name} leftover={leftover} code={code}", flush=True)
                else:
                    job.state = "failed"
                    failed.append(job.name)
                    print(f"fail {job.name} leftover={leftover} code={code}", flush=True)
                busy.pop(gpu, None)
        still_cpu: list[Job] = []
        for job in cpu_running:
            if alive(job):
                still_cpu.append(job)
                continue
            leftover = job.remaining()
            code = exit_code(job)
            if leftover <= 0 or job.kind == "analyze":
                job.state = "done"
                done.add(job.name)
                print(f"complete {job.name} cpu code={code}", flush=True)
            elif job.retries < 1:
                job.retries += 1
                job.state = "pending"
                job.proc = None
                print(f"retry {job.name} leftover={leftover} code={code}", flush=True)
            else:
                job.state = "failed"
                failed.append(job.name)
                print(f"fail {job.name} leftover={leftover} code={code}", flush=True)
        cpu_running = still_cpu
        runnable = sorted((job for job in job_list if job.runnable(done)), key=lambda job: (job.priority, job.name))
        for job in runnable:
            if job.needs_gpu:
                free = next((gpu for gpu in gpus if gpu not in busy), None)
                if free is None:
                    continue
                launch(job, free)
                busy[free] = job
            elif len(cpu_running) < 2:
                launch(job, None)
                cpu_running.append(job)
        pending = [job for job in job_list if job.state in {"pending", "running"}]
        if not pending:
            break
        print(
            f"tick busy={ {gpu: job.name for gpu, job in busy.items()} } "
            f"cpu={[job.name for job in cpu_running]} pending={len(pending)} done={len(done)}",
            flush=True,
        )
        time.sleep(args.poll)
    if failed:
        raise RuntimeError("dynamic scheduler failures: " + ", ".join(failed))
    print("all v2 extracts complete", flush=True)


if __name__ == "__main__":
    main()
