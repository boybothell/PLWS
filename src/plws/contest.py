"""Contest-set PUMA + PLWS lane. Not the four-model main table."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from plws.artifacts import load_jsonl
from plws.matrix import (
    FIRSTWIN,
    FROZEN_FULLCOT,
    dense_complete,
    job_rows,
    jobs_present,
    needs_cpu_export,
    plws_score_read_dirs,
    puma_complete,
)
from plws.paths import PLWSPaths
from plws.protocol import (
    FULLCOT_GENERATION_TOKENS,
    PROMPT_RESERVE_TOKENS,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)

CONTEST_MODELS = ("qwen3_30b_a3b",)
FOLLOWON_MODELS = ("r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b")
QUEUE_MODELS = CONTEST_MODELS + FOLLOWON_MODELS
TWO_GPU_MODELS = frozenset({"qwen3_30b_a3b", "qwq_32b", "qwen3_32b"})
MODELS = {
    "qwen3_30b_a3b": "/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507",
    "qwq_32b": "/mnt/d/lsj/models/QwQ-32B",
    "qwen3_32b": "/mnt/d/lsj/models/Qwen3-32B",
    "r1_7b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B",
    "nemotron_8b": "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1",
    "r1_14b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B",
    "qwen3_4b": "/mnt/d/lsj/models/Qwen3-4B",
}
ALIGN_CONF = {
    "qwen3_30b_a3b": "Q30B-T.conf",
    "qwq_32b": "DS-32B.conf",
    "qwen3_32b": "DS-32B.conf",
    "r1_7b": "DS-7B.conf",
    "nemotron_8b": "Nemotron.conf",
    "r1_14b": "DS-14B.conf",
    "qwen3_4b": "DS-7B.conf",
}
DATASETS = ("brumo25", "hmmt25", "aime24", "aime25", "aime26")
SEEDS = (42, 0, 1, 123, 7)
FIFTH_SEED = 7
PHASES = (
    "contest_prereq",
    "contest_plws",
    "followon_prereq",
    "followon_plws",
)
BATCH_SIZE = 8
GPU_COUNT = 2


@dataclass(frozen=True, slots=True)
class ContestProtocol:
    protocol_id: str
    generation_tokens: int
    answer_fix_tokens: int
    prompt_reserve_tokens: int = PROMPT_RESERVE_TOKENS

    @property
    def max_model_len(self) -> int:
        return (
            self.generation_tokens
            + self.prompt_reserve_tokens
            + self.answer_fix_tokens
        )


MAIN_PROTOCOL = ContestProtocol(
    PROTOCOL_ID,
    FULLCOT_GENERATION_TOKENS,
    TRUNCATED_ANSWER_FIX_TOKENS,
)


def protocol_for(_model: str) -> ContestProtocol:
    return MAIN_PROTOCOL


def sample_meta_path(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> Path:
    return paths.root / "samples" / model / dataset / f"seed_{seed}" / "sample_meta.json"


def sample_matches_protocol(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> bool:
    path = sample_meta_path(paths, model, dataset, seed)
    if not path.is_file():
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = protocol_for(model)
    if int(meta.get("max_tokens") or 0) != protocol.generation_tokens:
        return False
    if int(meta.get("answer_fix_max_tokens") or 0) != protocol.answer_fix_tokens:
        return False
    if meta.get("prompt_version") != "default":
        return False
    protocol_id = meta.get("protocol_id")
    if protocol_id not in {None, protocol.protocol_id}:
        return False
    reserve = meta.get("prompt_reserve_tokens")
    if reserve is not None and int(reserve) != protocol.prompt_reserve_tokens:
        return False
    context = meta.get("max_model_len")
    if context is not None and int(context) != protocol.max_model_len:
        return False
    return True


def contest_reusable_score(row: Mapping[str, Any], protocol: ContestProtocol) -> bool:
    return (
        row.get("status") in {"ok", "too_long"}
        and row.get("uid")
        and row.get("protocol_id") == protocol.protocol_id
        and row.get("max_model_len") == protocol.max_model_len
        and int(row.get("truncated_answer_fix_tokens") or 0)
        == protocol.answer_fix_tokens
        and int(row.get("fullcot_generation_tokens") or 0)
        == protocol.generation_tokens
    )


@dataclass(frozen=True, slots=True)
class ContestTask:
    task_id: str
    phase: str
    gpu_count: int
    kind: str
    model: str
    dataset: str
    seed: int
    run_kind: str = FIRSTWIN
    shard_id: int = 0
    num_shards: int = 1

    @property
    def priority(self) -> int:
        return PHASES.index(self.phase)


def align_conf(model: str) -> str:
    return ALIGN_CONF[model]


def batch_size(model: str) -> int:
    if model == "r1_14b":
        return 4
    if model in TWO_GPU_MODELS:
        return BATCH_SIZE
    return 64


def gpu_count(model: str) -> int:
    return 2 if model in TWO_GPU_MODELS else 1


def model_rank(model: str) -> int:
    try:
        return QUEUE_MODELS.index(model)
    except ValueError:
        return len(QUEUE_MODELS)


def model_phases(model: str) -> tuple[str, str]:
    if model in FOLLOWON_MODELS:
        return "followon_prereq", "followon_plws"
    return "contest_prereq", "contest_plws"


def sample_answers_path(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> Path:
    return paths.root / "samples" / model / dataset / f"seed_{seed}" / "answers.json"


def may_sample_fullcot(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> bool:
    if model not in FROZEN_FULLCOT:
        return True
    return not sample_answers_path(paths, model, dataset, seed).is_file()


def contest_task_ready(paths: PLWSPaths, task: ContestTask) -> bool:
    if task.kind != "plws":
        return True
    return jobs_present(paths, task.model, task.dataset, task.seed)


def contest_blocked_phases(
    pending: Iterable[ContestTask],
    running_phases: Iterable[str] = (),
) -> frozenset[str]:
    phases = {task.phase for task in pending} | set(running_phases)
    if any(phase.startswith("contest_") for phase in phases):
        return frozenset({"followon_prereq", "followon_plws"})
    return frozenset()


ENGINE_LOADED_MARKERS = (
    "Model loaded.",
    "init engine (profile, create kv cache, warmup model) took",
    "Running generation...",
    "official-batch step=",
)


def log_shows_engine_loaded(text: str) -> bool:
    return any(marker in text for marker in ENGINE_LOADED_MARKERS)


def is_task_log_name(task_id: str, name: str) -> bool:
    return name == f"{task_id}.log" or name.startswith(f"{task_id}.")


def engine_loaded_from_logs(
    log_paths: Iterable[Path], *, tail_bytes: int = 262144
) -> bool:
    for path in log_paths:
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - tail_bytes))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        if log_shows_engine_loaded(text):
            return True
    return False


def select_cold_starts(
    candidates: Iterable[ContestTask],
    *,
    any_loading: bool,
) -> list[ContestTask]:
    """At most one new engine may start while another is still loading."""

    if any_loading:
        return []
    for task in candidates:
        return [task]
    return []


def fill_contest_dispatch(
    pending: list[ContestTask],
    free_gpus: int,
    paths: PLWSPaths,
    blocked_phases: Iterable[str] = (),
) -> list[ContestTask]:
    remaining = list(pending)
    taken: list[ContestTask] = []
    free = free_gpus
    blocked = set(blocked_phases)
    while True:
        ready = [
            task
            for task in remaining
            if task.gpu_count <= free
            and task.phase not in blocked
            and contest_task_ready(paths, task)
        ]
        if not ready:
            break
        fit = min(ready, key=lambda task: (task.priority, model_rank(task.model), task.task_id))
        remaining.remove(fit)
        taken.append(fit)
        free -= fit.gpu_count
    return taken


def fill_leftover_dispatch(
    pending: list[ContestTask],
    free_gpus: int,
    paths: PLWSPaths,
    blocked_phases: Iterable[str] = (),
) -> list[ContestTask]:
    """Prefer 30B contest work; leftover GPUs too small for TP=2 go to follow-on."""

    taken = fill_contest_dispatch(pending, free_gpus, paths, blocked_phases)
    rest = free_gpus - sum(task.gpu_count for task in taken)
    if rest < 1:
        return taken
    leftover_pending = [task for task in pending if task not in taken]
    taken.extend(fill_contest_dispatch(leftover_pending, rest, paths, blocked_phases=()))
    return taken


def _cmd_flag(cmd: list[str], flag: str) -> str | None:
    try:
        return cmd[cmd.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def is_eight_olympiad_leftover_cmd(cmd: list[str]) -> bool:
    if not any(part.endswith("score_leftover_suppress.py") for part in cmd):
        return False
    return (
        _cmd_flag(cmd, "--model-tag") == "qwen3_8b"
        and _cmd_flag(cmd, "--dataset") == "olympiadbench"
        and _cmd_flag(cmd, "--seed") == "42"
    )


def is_contest_queue_cmd(cmd: list[str]) -> bool:
    return any(part.endswith("run_contest_queue.py") for part in cmd) and "--dry-run" not in cmd


def _iter_cmdlines() -> Iterable[list[str]]:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if cmd:
            yield cmd


def eight_olympiad_leftover_alive() -> bool:
    return any(is_eight_olympiad_leftover_cmd(cmd) for cmd in _iter_cmdlines())


def contest_queue_alive() -> bool:
    return any(is_contest_queue_cmd(cmd) for cmd in _iter_cmdlines())


def leftover_gpus_from_proc(cmd: list[str], env: Mapping[str, str]) -> tuple[str, ...]:
    if not is_eight_olympiad_leftover_cmd(cmd):
        return ()
    raw = env.get("CUDA_VISIBLE_DEVICES") or env.get("GPU") or ""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _proc_environ(pid: str) -> dict[str, str]:
    try:
        raw = Path("/proc") / pid / "environ"
        data = raw.read_bytes()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for item in data.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def leftover_8b_gpus() -> set[str]:
    held: set[str] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        held.update(leftover_gpus_from_proc(cmd, _proc_environ(proc.name)))
    return held


def gpu_used_mib() -> dict[str, int]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    used: dict[str, int] = {}
    for line in output.splitlines():
        if "," not in line:
            continue
        index, memory = line.split(",", 1)
        used[index.strip()] = int(float(memory.strip()))
    return used


def idle_contest_gpus(
    pool: Iterable[str],
    claimed: Iterable[str] = (),
    *,
    max_used_mib: int = 800,
    leftover: Iterable[str] | None = None,
    used_mib: Mapping[str, int] | None = None,
) -> list[str]:
    blocked = set(claimed)
    blocked.update(leftover if leftover is not None else leftover_8b_gpus())
    memory = dict(used_mib) if used_mib is not None else gpu_used_mib()
    idle: list[str] = []
    for gpu in pool:
        if gpu in blocked:
            continue
        if memory.get(gpu, 10**9) >= max_used_mib:
            continue
        idle.append(gpu)
    return idle


def contest_needs_prereq(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> bool:
    if not sample_matches_protocol(paths, model, dataset, seed):
        return True
    if not puma_complete(paths, model, dataset, seed):
        return True
    if jobs_present(paths, model, dataset, seed):
        return False
    return not dense_complete(paths, model, dataset, seed)


def cells_needing_cpu_export(
    paths: PLWSPaths, models: Iterable[str]
) -> list[tuple[str, str, int]]:
    found: list[tuple[str, str, int]] = []
    for model in models:
        for dataset in DATASETS:
            for seed in SEEDS:
                if needs_cpu_export(paths, model, dataset, seed):
                    found.append((model, dataset, seed))
    return found


def prereq_task(model: str, dataset: str, seed: int) -> ContestTask:
    prereq_phase, _ = model_phases(model)
    return ContestTask(
        task_id=f"prereq__{model}__{dataset}__s{seed}",
        phase=prereq_phase,
        gpu_count=gpu_count(model),
        kind="prereq",
        model=model,
        dataset=dataset,
        seed=seed,
    )


def plws_task(model: str, dataset: str, seed: int) -> ContestTask:
    _, plws_phase = model_phases(model)
    return ContestTask(
        task_id=f"plws__{model}__{dataset}__s{seed}",
        phase=plws_phase,
        gpu_count=gpu_count(model),
        kind="plws",
        model=model,
        dataset=dataset,
        seed=seed,
        run_kind=FIRSTWIN,
    )


def build_tasks(
    paths: PLWSPaths,
    *,
    models: Iterable[str] | None = None,
) -> list[ContestTask]:
    chosen = tuple(models) if models is not None else CONTEST_MODELS
    tasks: list[ContestTask] = []
    for model in chosen:
        if model not in MODELS:
            raise ValueError(f"unknown contest model {model}")
        for dataset in DATASETS:
            for seed in SEEDS:
                if contest_needs_prereq(paths, model, dataset, seed):
                    tasks.append(prereq_task(model, dataset, seed))
                tasks.append(plws_task(model, dataset, seed))
    return sorted(
        tasks,
        key=lambda task: (task.priority, model_rank(task.model), task.task_id),
    )


def task_complete(paths: PLWSPaths, task: ContestTask) -> tuple[bool, str]:
    if task.kind == "prereq":
        if contest_needs_prereq(paths, task.model, task.dataset, task.seed):
            return False, "windows or PUMA missing"
        if jobs_present(paths, task.model, task.dataset, task.seed):
            return True, "jobs ready"
        return True, "prereq no longer required"
    if task.kind == "plws":
        return contest_plws_complete(
            paths,
            task.model,
            task.dataset,
            task.seed,
            shard_id=task.shard_id,
            num_shards=task.num_shards,
        )
    return False, f"unknown task kind {task.kind}"


def first_open_phase(
    tasks: Iterable[ContestTask],
    finished: Mapping[str, str],
) -> str | None:
    remaining = {task.phase for task in tasks if task.task_id not in finished}
    for phase in PHASES:
        if phase in remaining:
            return phase
    return None


def contest_plws_complete(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    *,
    shard_id: int = 0,
    num_shards: int = 1,
) -> tuple[bool, str]:
    if not jobs_present(paths, model, dataset, seed):
        return False, "missing jobs"
    protocol = protocol_for(model)
    jobs = [
        str(row["uid"])
        for index, row in enumerate(job_rows(paths, model, dataset, seed))
        if index % num_shards == shard_id
    ]
    if not jobs:
        return True, "empty window shard"
    done = {
        str(row["uid"])
        for directory in plws_score_read_dirs(paths, model, dataset, seed)
        for path in (
            *sorted(directory.glob("shard_*.jsonl")),
            *sorted(directory.glob("scores_shard*.jsonl")),
            directory / "scores.jsonl",
        )
        for row in load_jsonl(path)
        if contest_reusable_score(row, protocol)
    }
    missing = [uid for uid in jobs if uid not in done]
    have = len(jobs) - len(missing)
    if missing:
        return False, f"done {have}/{len(jobs)}"
    return True, f"done {have}/{len(jobs)}"


def summarize(tasks: Iterable[ContestTask]) -> dict[str, int]:
    counts: dict[str, int] = {phase: 0 for phase in PHASES}
    counts["total"] = 0
    for task in tasks:
        counts[task.phase] += 1
        counts["total"] += 1
    return counts
