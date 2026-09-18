"""Contest-set PUMA + PLWS lane. Not the four-model main table."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from plws.artifacts import load_jsonl
from plws.grading import has_gold
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
from plws.puma_grading import puma_grades_verified
from plws.protocol import (
    FULLCOT_GENERATION_TOKENS,
    PROMPT_RESERVE_TOKENS,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)
from plws.runtime import model_path

CONTEST_MODELS = ("qwen3_30b_a3b",)
FOLLOWON_MODELS = ("r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b")
QUEUE_MODELS = CONTEST_MODELS + FOLLOWON_MODELS
FILL_MODELS = (
    "r1_7b",
    "nemotron_8b",
    "qwen3_4b",
    "qwen3_8b",
    "r1_14b",
    "r1_1p5b",
    "r1_llama_8b",
    "r1_32b",
    "qwen3_30b_a3b",
    "qwen3_32b",
    "qwq_32b",
)
FILL_FOLLOWON = frozenset({"qwen3_8b", "r1_1p5b", "r1_llama_8b", "r1_32b"})
TWO_GPU_MODELS = frozenset({"qwen3_30b_a3b", "qwq_32b", "qwen3_32b", "r1_32b"})
MODELS = {
    tag: str(model_path(tag))
    for tag in (
        "qwen3_30b_a3b",
        "qwq_32b",
        "qwen3_32b",
        "r1_7b",
        "nemotron_8b",
        "r1_14b",
        "r1_1p5b",
        "r1_llama_8b",
        "r1_32b",
        "qwen3_4b",
        "qwen3_8b",
    )
}
ALIGN_CONF = {
    "qwen3_30b_a3b": "Q30B-T.conf",
    "qwq_32b": "DS-32B.conf",
    "qwen3_32b": "DS-32B.conf",
    "r1_7b": "DS-7B.conf",
    "nemotron_8b": "Nemotron.conf",
    "r1_14b": "DS-14B.conf",
    "r1_1p5b": "DS-7B.conf",
    "r1_llama_8b": "DS-7B.conf",
    "r1_32b": "DS-32B.conf",
    "qwen3_4b": "DS-7B.conf",
    "qwen3_8b": "DS-7B.conf",
}
DATASETS = ("brumo25", "hmmt25", "aime24", "aime25", "aime26")
FILL_DATASETS = DATASETS + ("amc23",)
SEEDS = (42, 0, 1, 123, 7)
# Official new-run scope. Historical fill still uses FILL_DATASETS / SEEDS.
OFFICIAL_NEW_DATASETS = (
    "math-500",
    "olympiadbench",
    "gpqa-diamond",
    "aime25",
    "hmmt25",
    "amc23",
)
OFFICIAL_FIRST_SEEDS = (42, 0, 1)
OFFICIAL_LATER_SEEDS = (123, 7)
NO_NEW_WORK_DATASETS = ("aime24", "aime26", "brumo25")
PUBLISHED_LEGACY_PLWS = frozenset(
    ("r1_7b", "amc23", seed) for seed in (42, 0, 1, 123)
)
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
        and "gt" in row
        and has_gold(row.get("gt"))
        and "gold_error" in row
        and not row.get("gold_error")
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
    if model not in TWO_GPU_MODELS:
        return 1
    override = os.environ.get("PLWS_LARGE_TP") or os.environ.get("PLWS_TP")
    if override:
        value = int(override)
        if value < 1:
            raise ValueError(f"PLWS_LARGE_TP must be positive, got {value}")
        return value
    return 2


def model_rank(model: str) -> int:
    try:
        return QUEUE_MODELS.index(model)
    except ValueError:
        return len(QUEUE_MODELS)


def model_phases(model: str) -> tuple[str, str]:
    if model in FOLLOWON_MODELS or model in FILL_FOLLOWON:
        return "followon_prereq", "followon_plws"
    return "contest_prereq", "contest_plws"


def fill_model_rank(model: str) -> int:
    try:
        return FILL_MODELS.index(model)
    except ValueError:
        return len(FILL_MODELS)


def fill_dataset_rank(dataset: str) -> int:
    try:
        return FILL_DATASETS.index(dataset)
    except ValueError:
        return len(FILL_DATASETS)


def fill_kind_rank(kind: str) -> int:
    return 0 if kind == "plws" else 1


def fill_sort_key(task: ContestTask) -> tuple[int, int, int, int, str]:
    return (
        fill_kind_rank(task.kind),
        fill_model_rank(task.model),
        fill_dataset_rank(task.dataset),
        int(task.seed),
        task.task_id,
    )


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


def log_after_latest_start(text: str) -> str:
    """Keep only the current attempt. Older appends stay in the same file."""

    marker = text.rfind(" start gpus=")
    if marker >= 0:
        return text[marker:]
    return text


def log_loaded_after_latest_start(text: str) -> bool:
    """Ignore Model loaded. markers from an earlier append to the same log."""

    return log_shows_engine_loaded(log_after_latest_start(text))


GPU_RELEASED_MARKERS = (
    "[dense-model] GPU phase done",
    "[contest-prereq] gpu released",
    "[matrix-prereq] gpu released",
)
DENSE_ENGINE_RELEASED = "releasing engine before CPU postprocess"


def log_shows_gpu_released(text: str) -> bool:
    """True after the last GPU phase; PUMA trial shutdown must not match."""

    text = log_after_latest_start(text)
    if any(marker in text for marker in GPU_RELEASED_MARKERS):
        return True
    dense_at = text.rfind("[dense-model]")
    if dense_at < 0:
        return False
    return DENSE_ENGINE_RELEASED in text[dense_at:]


def read_log_tail(path: Path, *, tail_bytes: int = 262144) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - tail_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _rfind_bytes(path: Path, needle: bytes, *, chunk: int = 1024 * 1024) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        return -1
    overlap = max(0, len(needle) - 1)
    pos = size
    while pos > 0:
        start = max(0, pos - chunk)
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                data = handle.read(pos - start + (overlap if start else 0))
        except OSError:
            return -1
        found = data.rfind(needle)
        if found >= 0:
            return start + found
        pos = start
    return -1


def _find_bytes_from(
    path: Path, needle: bytes, start: int, *, chunk: int = 1024 * 1024
) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        return -1
    pos = max(0, start)
    carry = b""
    while pos < size:
        try:
            with path.open("rb") as handle:
                handle.seek(pos)
                data = carry + handle.read(min(chunk, size - pos))
        except OSError:
            return -1
        found = data.find(needle)
        if found >= 0:
            return pos - len(carry) + found
        carry = data[-(len(needle) - 1) :] if len(needle) > 1 else b""
        pos += min(chunk, size - pos)
    return -1


def log_path_shows_gpu_released(path: Path) -> bool:
    """Same as log_shows_gpu_released, but dense headers may have scrolled off the tail."""

    if not path.is_file():
        return False
    tail = read_log_tail(path)
    if log_shows_gpu_released(tail):
        return True
    if DENSE_ENGINE_RELEASED not in tail and not any(
        marker in tail for marker in GPU_RELEASED_MARKERS
    ):
        return False
    start_at = _rfind_bytes(path, b" start gpus=")
    search_from = start_at if start_at >= 0 else 0
    dense_at = _find_bytes_from(path, b"[dense-model]", search_from)
    if dense_at < 0:
        return False
    if _find_bytes_from(path, DENSE_ENGINE_RELEASED.encode(), dense_at) >= 0:
        return True
    return any(
        _find_bytes_from(path, marker.encode(), dense_at) >= 0
        for marker in GPU_RELEASED_MARKERS
    )


def lease_ready_to_drop(
    *,
    has_workers: bool,
    used_mib: Mapping[str, int],
    gpus: Iterable[str],
    max_used_mib: int = 800,
) -> bool:
    held = tuple(gpus)
    if not held or has_workers:
        return False
    return not any(used_mib.get(gpu, 10**9) >= max_used_mib for gpu in held)


def should_release_gpu_lease(
    gpus: Iterable[str],
    *,
    log_text: str = "",
    log_path: Path | None = None,
    has_workers: bool,
    used_mib: Mapping[str, int],
    max_used_mib: int = 800,
) -> bool:
    """Drop the lease only when dense GPU work is gone and cards are empty.

    CPU postprocess of gen_trial_answers still holds the wrapper PID. The
    queue must not keep those cards claimed, or the pool sits idle.
    """

    if not lease_ready_to_drop(
        has_workers=has_workers,
        used_mib=used_mib,
        gpus=gpus,
        max_used_mib=max_used_mib,
    ):
        return False
    if log_path is not None:
        return log_path_shows_gpu_released(log_path)
    return log_shows_gpu_released(log_text)


def engine_ready_for_next_cold_start(
    *,
    workers_in_d: bool,
    log_loaded: bool,
) -> bool:
    """Workers existing is not loaded. D-state or a fresh start still blocks."""

    return log_loaded and not workers_in_d


def is_vllm_loading_comm(comm: str) -> bool:
    """TP workers and EngineCore both load weights. comm is 15 chars in /proc."""

    return comm.startswith("VLLM::Worker") or comm.startswith("VLLM::EngineCor")


def vllm_workers_loading(exclude_pids: Iterable[int] = ()) -> bool:
    """True when a VLLM worker or EngineCore is still in uninterruptible load (D)."""

    blocked = {int(pid) for pid in exclude_pids}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid in blocked:
            continue
        try:
            comm = (proc / "comm").read_text().strip()
        except OSError:
            continue
        if not is_vllm_loading_comm(comm):
            continue
        try:
            for line in (proc / "status").read_text().splitlines():
                if line.startswith("State:"):
                    if line.split()[1].startswith("D"):
                        return True
                    break
        except OSError:
            continue
    return False


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
        if log_loaded_after_latest_start(text):
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


def dual_lane_cap(pool_size: int | None) -> int:
    """How many TP=2 jobs this pool may run at once.

    A 2-card pool (0+1 or 3+4 after dropping 2,5,6) holds one pair. A 4-card
    pool may run two pairs. None keeps the old unbounded extra-pair loop.
    """

    if pool_size is None:
        return 10**9
    return pool_size // 2


def fill_dispatch(
    pending: list[ContestTask],
    free_gpus: int,
    paths: PLWSPaths,
    *,
    running_gpu_counts: Iterable[int] = (),
    pool_size: int | None = None,
) -> list[ContestTask]:
    """Run the 1-GPU and 2-GPU lanes side by side.

    Leftover PLWS still wins inside the 1-GPU lane. A ready TP=2 job keeps a
    pair reserved instead of letting 1-GPU work occupy the whole 3-card pool.
    Concurrent TP=2 lanes cannot exceed pool_size // 2, so a 2-card leftover
    pool does not invent a second pair.
    """

    ready = [
        task
        for task in pending
        if contest_task_ready(paths, task)
    ]
    ones = sorted(
        (task for task in ready if task.gpu_count == 1),
        key=fill_sort_key,
    )
    twos = sorted(
        (task for task in ready if task.gpu_count >= 2),
        key=fill_sort_key,
    )
    taken: list[ContestTask] = []
    free = free_gpus
    dual_running = sum(1 for count in running_gpu_counts if count >= 2)
    max_dual = dual_lane_cap(pool_size)
    can_pair = True if pool_size is None else pool_size >= 2

    if twos and free >= 2 and dual_running < max_dual:
        taken.append(twos.pop(0))
        free -= taken[-1].gpu_count
        dual_running += 1
    elif twos and dual_running < max_dual and can_pair and free < 2:
        return []

    while ones and free >= 1:
        taken.append(ones.pop(0))
        free -= 1
    while twos and free >= 2 and dual_running < max_dual:
        taken.append(twos.pop(0))
        free -= taken[-1].gpu_count
        dual_running += 1
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


def is_contest_fill_queue_cmd(cmd: list[str]) -> bool:
    return (
        any(part.endswith("run_contest_fill_queue.py") for part in cmd)
        and "--dry-run" not in cmd
    )


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


def contest_fill_queue_alive(*, exclude_pid: int | None = None) -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        if exclude_pid is not None and int(proc.name) == exclude_pid:
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if cmd and is_contest_fill_queue_cmd(cmd):
            return True
    return False


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
    paths: PLWSPaths,
    models: Iterable[str],
    datasets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[tuple[str, str, int]]:
    chosen = tuple(datasets) if datasets is not None else DATASETS
    chosen_seeds = tuple(seeds) if seeds is not None else SEEDS
    found: list[tuple[str, str, int]] = []
    for model in models:
        for dataset in chosen:
            for seed in chosen_seeds:
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


def _legacy_leftover_uids(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> set[str]:
    done: set[str] = set()
    for kind in ("low", "mix", "high"):
        for directory in paths.legacy_score_dirs(
            model, seed, "suppress", kind, k=4, lexicon="core"
        ):
            if not directory.is_dir():
                continue
            for path in (
                *sorted(directory.glob("shard_*.jsonl")),
                *sorted(directory.glob("scores*.jsonl")),
            ):
                for row in load_jsonl(path):
                    if row.get("status") not in {"ok", "too_long"}:
                        continue
                    rec_dataset = row.get("dataset")
                    uid = str(row.get("uid") or "")
                    if rec_dataset and rec_dataset != dataset:
                        continue
                    if not rec_dataset and uid:
                        parts = uid.split(":")
                        if len(parts) > 1 and parts[1] != dataset:
                            continue
                    if uid:
                        done.add(uid)
    return done


def fill_legacy_plws_complete(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> tuple[bool, str]:
    if (model, dataset, int(seed)) not in PUBLISHED_LEGACY_PLWS:
        return False, "not published leftover"
    done = _legacy_leftover_uids(paths, model, dataset, seed)
    if jobs_present(paths, model, dataset, seed):
        jobs = [
            str(row["uid"])
            for row in job_rows(paths, model, dataset, seed)
            if row.get("uid")
        ]
        missing = [uid for uid in jobs if uid not in done]
        have = len(jobs) - len(missing)
        if missing:
            return False, f"legacy leftover {have}/{len(jobs)}"
        return True, f"legacy leftover {have}/{len(jobs)}"
    if done:
        return True, f"legacy leftover n={len(done)}"
    return False, "missing legacy leftover"


def fill_needs_prereq(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> bool:
    if fill_legacy_plws_complete(paths, model, dataset, seed)[0]:
        return False
    if puma_complete(paths, model, dataset, seed) and not puma_grades_verified(
        paths.puma_statistics_path(model, dataset, seed)
    ):
        return True
    if jobs_present(paths, model, dataset, seed) and puma_complete(
        paths, model, dataset, seed
    ):
        return False
    if not may_sample_fullcot(paths, model, dataset, seed):
        if not sample_matches_protocol(paths, model, dataset, seed):
            return False
        if not puma_complete(paths, model, dataset, seed):
            return True
        if jobs_present(paths, model, dataset, seed):
            return False
        return not dense_complete(paths, model, dataset, seed)
    return contest_needs_prereq(paths, model, dataset, seed)


def fill_plws_complete(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    *,
    shard_id: int = 0,
    num_shards: int = 1,
) -> tuple[bool, str]:
    ok, reason = contest_plws_complete(
        paths,
        model,
        dataset,
        seed,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    if ok:
        return ok, reason
    legacy_ok, legacy_reason = fill_legacy_plws_complete(
        paths, model, dataset, seed
    )
    if legacy_ok:
        return True, legacy_reason
    return ok, reason


def build_fill_tasks(
    paths: PLWSPaths,
    *,
    models: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[ContestTask]:
    chosen = tuple(models) if models is not None else FILL_MODELS
    chosen_ds = tuple(datasets) if datasets is not None else FILL_DATASETS
    chosen_seeds = tuple(seeds) if seeds is not None else SEEDS
    tasks: list[ContestTask] = []
    for model in chosen:
        if model not in MODELS:
            raise ValueError(f"unknown contest fill model {model}")
        for dataset in chosen_ds:
            for seed in chosen_seeds:
                if fill_needs_prereq(paths, model, dataset, seed):
                    tasks.append(prereq_task(model, dataset, seed))
                tasks.append(plws_task(model, dataset, seed))
    return sorted(tasks, key=fill_sort_key)


def filter_fill_tasks(
    tasks: Iterable[ContestTask],
    task_ids: Iterable[str],
) -> list[ContestTask]:
    wanted = [item.strip() for item in task_ids if str(item).strip()]
    if not wanted:
        return list(tasks)
    known = {task.task_id: task for task in tasks}
    missing = [task_id for task_id in wanted if task_id not in known]
    if missing:
        raise ValueError(f"unknown fill task ids: {missing}")
    return sorted((known[task_id] for task_id in wanted), key=fill_sort_key)


def fullcot_only_tasks(tasks: Iterable[ContestTask]) -> list[ContestTask]:
    """Keep prereq cells only. Full-CoT-only fill never launches PLWS."""

    return sorted(
        (task for task in tasks if task.kind == "prereq"),
        key=fill_sort_key,
    )


def fill_fullcot_complete(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> tuple[bool, str]:
    if not sample_answers_path(paths, model, dataset, seed).is_file():
        return False, "fullcot answers missing"
    if not sample_matches_protocol(paths, model, dataset, seed):
        return False, "fullcot protocol mismatch"
    return True, "fullcot ready"


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


def fill_task_complete(paths: PLWSPaths, task: ContestTask) -> tuple[bool, str]:
    if task.kind == "prereq":
        if fill_needs_prereq(paths, task.model, task.dataset, task.seed):
            return False, "windows or PUMA missing"
        if jobs_present(paths, task.model, task.dataset, task.seed):
            return True, "jobs ready"
        return True, "prereq no longer required"
    if task.kind == "plws":
        return fill_plws_complete(
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
