"""Five-model then 30B PLWS-core + unified-host DEER matrix.

Planning and completion only. The GPU queue lives in
``scripts/run_matrix_queue.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from plws.artifacts import atomic_write_jsonl, done_uids, load_jsonl
from plws.paths import PLWSPaths
from plws.protocol import (
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)

FIVE_MODELS = ("r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b", "qwen3_8b")
EIGHT_MODEL = "qwen3_8b"
THIRTY_MODEL = "qwen3_30b_a3b"
ALL_MODELS = (*FIVE_MODELS, THIRTY_MODEL)
FROZEN_FULLCOT = frozenset({"r1_7b", "nemotron_8b", "r1_14b"})
QWEN_MODELS = frozenset({"qwen3_4b", EIGHT_MODEL, THIRTY_MODEL})
DATASETS = (
    "math-500",
    "olympiadbench",
    "gpqa-diamond",
    "aime24",
    "aime25",
)
SEEDS = (42, 0, 1, 123)
FIRSTWIN = "firstwin"
TIER_KINDS = ("low", "mix", "high")
LEXICON = "core"
K = 4
SAMPLING_SEED = 20260904
PHASES = (
    "five_prereq",
    "five_plws",
    "five_deer",
    "eight_prereq",
    "eight_plws",
    "eight_deer",
    "thirty_prereq",
    "thirty_plws",
    "thirty_deer",
    "late_deer",
)
DEER_EXCLUDED = frozenset()
DEER_DEFERRED = frozenset({("nemotron_8b", "gpqa-diamond")})
LARGE_DATASETS = frozenset({"math-500", "olympiadbench"})
AIME4S_MODELS = frozenset({"qwen3_4b", "qwen3_8b"})
AIME_DATASETS = frozenset({"aime24", "aime25"})

MODELS = {
    "r1_7b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-7B",
    "nemotron_8b": "/mnt/d/lsj/models/Llama-3.1-Nemotron-Nano-8B-v1",
    "r1_14b": "/mnt/d/lsj/models/DeepSeek-R1-Distill-Qwen-14B",
    "qwen3_4b": "/mnt/d/lsj/models/Qwen3-4B",
    "qwen3_8b": "/mnt/d/lsj/models/Qwen3-8B",
    "qwen3_30b_a3b": "/mnt/d/lsj/models/Qwen3-30B-A3B-Thinking-2507",
}

BLOCKER_STATUS = (
    "results/experiments/lexicon_ablation/pre/queue/status.json",
    "results/experiments/lexicon_ablation/aime4s/qwen3_4b/queue/status.json",
    "results/experiments/lexicon_ablation/aime4s/qwen3_8b/queue/status.json",
)


@dataclass(frozen=True, slots=True)
class MatrixTask:
    task_id: str
    phase: str
    gpu_count: int
    kind: str
    model: str
    dataset: str
    seed: int
    run_kind: str = ""
    shard_id: int = 0
    num_shards: int = 1

    @property
    def priority(self) -> int:
        return PHASES.index(self.phase)


def puma_repo(plws_root: Path) -> Path:
    return Path(plws_root).resolve().parent / "PUMA"


def puma_data_path(plws_root: Path, dataset: str) -> Path:
    return puma_repo(plws_root) / "data" / f"{dataset}_test.jsonl"


def model_lane(model: str) -> str:
    if model == THIRTY_MODEL:
        return "thirty"
    if model == EIGHT_MODEL:
        return "eight"
    return "five"


def gpu_count(model: str) -> int:
    return 2 if model == THIRTY_MODEL else 1


def batch_size(model: str) -> int:
    if model == "r1_14b":
        return 4
    if model == THIRTY_MODEL:
        return 8
    return 64


def shard_count(dataset: str) -> int:
    return 2 if dataset in LARGE_DATASETS else 1


def puma_complete(paths: PLWSPaths, model: str, dataset: str, seed: int) -> bool:
    directory = paths.puma_output_dir(model, dataset, seed)
    stats = paths.puma_statistics_path(model, dataset, seed)
    return stats.is_file() and (directory / "prefixed_answers.json").is_file()


def dense_complete(paths: PLWSPaths, model: str, dataset: str, seed: int) -> bool:
    return paths.dense_trial_path(model, dataset, seed).is_file()


def jobs_path(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    kind: str = FIRSTWIN,
) -> Path:
    return paths.jobs_path(
        model, dataset, seed, kind, k=K, lexicon=LEXICON
    )


def _tier_job_files(paths: PLWSPaths, model: str, dataset: str, seed: int) -> list[Path]:
    return [jobs_path(paths, model, dataset, seed, kind) for kind in TIER_KINDS]


def jobs_present(paths: PLWSPaths, model: str, dataset: str, seed: int) -> bool:
    if jobs_path(paths, model, dataset, seed).is_file():
        return True
    return all(path.is_file() for path in _tier_job_files(paths, model, dataset, seed))


def job_rows(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    if kind:
        return [
            row
            for row in load_jsonl(jobs_path(paths, model, dataset, seed, kind))
            if row.get("uid")
        ]
    firstwin = jobs_path(paths, model, dataset, seed)
    if firstwin.is_file():
        return [row for row in load_jsonl(firstwin) if row.get("uid")]
    rows: list[dict[str, Any]] = []
    for path in _tier_job_files(paths, model, dataset, seed):
        rows.extend(row for row in load_jsonl(path) if row.get("uid"))
    return rows


def needs_prereq(paths: PLWSPaths, model: str, dataset: str, seed: int) -> bool:
    if model in FROZEN_FULLCOT or model not in QWEN_MODELS:
        return False
    if not puma_complete(paths, model, dataset, seed):
        return True
    if jobs_present(paths, model, dataset, seed):
        return False
    return not dense_complete(paths, model, dataset, seed)


def needs_cpu_export(paths: PLWSPaths, model: str, dataset: str, seed: int) -> bool:
    return (
        not jobs_present(paths, model, dataset, seed)
        and dense_complete(paths, model, dataset, seed)
    )


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


def ensure_firstwin_jobs(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> bool:
    """Write firstwin.jsonl from already-split leftover files when needed."""

    dest = jobs_path(paths, model, dataset, seed)
    if dest.is_file():
        return False
    sources = _tier_job_files(paths, model, dataset, seed)
    if not all(path.is_file() for path in sources):
        return False
    rows = [row for path in sources for row in load_jsonl(path) if row.get("uid")]
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(dest, rows)
    return dest.is_file()


def plws_score_dir(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    kind: str = FIRSTWIN,
) -> Path:
    return paths.score_dir(
        model, dataset, seed, kind, k=K, lexicon=LEXICON
    )


def plws_score_read_dirs(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> list[Path]:
    return [
        plws_score_dir(paths, model, dataset, seed, kind)
        for kind in (FIRSTWIN, *TIER_KINDS)
    ]


def plws_complete(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    kind: str = FIRSTWIN,
    *,
    shard_id: int = 0,
    num_shards: int = 1,
) -> tuple[bool, str]:
    if not jobs_present(paths, model, dataset, seed):
        if model in FROZEN_FULLCOT:
            return True, "no window jobs; Full-CoT fallback"
        return False, "missing jobs"
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
        if reusable_score(row)
    }
    missing = [uid for uid in jobs if uid not in done]
    have = len(jobs) - len(missing)
    if missing:
        return False, f"done {have}/{len(jobs)}"
    return True, f"done {have}/{len(jobs)}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def deer_output_dir(paths: PLWSPaths, model: str, dataset: str, seed: int) -> Path:
    return (
        paths.results
        / "baselines"
        / "deer"
        / "puma_fullcot_32k_v2"
        / model
        / dataset
        / f"seed_{seed}"
    )


def deer_expected_policy(model: str) -> dict[str, Any]:
    qwen = model.startswith("qwen3_")
    return {
        "family_policy": "qwen3" if qwen else "standard",
        "think_ratio": 0.8 if qwen else 0.6,
        "confidence_policy": "avg2" if qwen else "avg1",
        "require_probe_think_close": qwen,
    }


def deer_complete(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    *,
    expected: int | None = None,
) -> tuple[bool, str]:
    output = deer_output_dir(paths, model, dataset, seed)
    result = output / "deer.jsonl"
    manifest_path = output / "manifest.json"
    if not result.is_file() or not manifest_path.is_file():
        return False, "missing deer.jsonl or manifest"
    try:
        manifest = _load_json(manifest_path)
        rows = load_jsonl(result)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return False, f"invalid DEER output: {exc}"
    if expected is None:
        data = puma_data_path(paths.root, dataset)
        if data.is_file():
            expected = sum(1 for line in data.read_text().splitlines() if line.strip())
        else:
            expected = 0
    policy = deer_expected_policy(model)
    deer = manifest.get("deer") or {}
    valid = (
        manifest.get("method") == "deer"
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("model_tag") == model
        and manifest.get("dataset") == dataset
        and _optional_int(manifest.get("seed")) == int(seed)
        and manifest.get("fullcot_generation_tokens") == FULLCOT_GENERATION_TOKENS
        and manifest.get("truncated_answer_fix_tokens")
        == TRUNCATED_ANSWER_FIX_TOKENS
        and manifest.get("max_model_len") == MAX_MODEL_LEN
        and deer.get("family_policy") == policy["family_policy"]
        and deer.get("think_ratio") == policy["think_ratio"]
        and deer.get("confidence_policy") == policy["confidence_policy"]
        and deer.get("require_probe_think_close")
        is policy["require_probe_think_close"]
        and len(rows) == expected
        and expected > 0
        and all(row.get("protocol_id") == PROTOCOL_ID for row in rows)
        and all("Error:" not in str(row.get("generated_text", "")) for row in rows)
    )
    if not valid:
        return False, f"identity/count mismatch actual={len(rows)} expected={expected}"
    return True, f"{len(rows)} records"


def aime4s_score_files(paths: PLWSPaths, model: str, seed: int) -> list[Path]:
    folder = (
        paths.results
        / "experiments"
        / "lexicon_ablation"
        / "aime4s"
        / model
        / f"s{seed}"
        / LEXICON
    )
    files = [
        *sorted(folder.glob("scores.jsonl")),
        *sorted(folder.glob("scores_shard*.jsonl")),
        *sorted(folder.glob("shard_*.jsonl")),
    ]
    return [path for path in files if path.is_file()]


def reusable_score(row: Mapping[str, Any]) -> bool:
    return (
        row.get("status") in {"ok", "too_long"}
        and row.get("uid")
        and row.get("protocol_id") == PROTOCOL_ID
        and row.get("max_model_len") == MAX_MODEL_LEN
        and int(row.get("truncated_answer_fix_tokens") or 0)
        == TRUNCATED_ANSWER_FIX_TOKENS
    )


def import_aime4s_core(paths: PLWSPaths, model: str, seed: int) -> dict[str, int]:
    """Copy finished AIME four-seed core scores into canonical firstwin cells."""

    if model not in AIME4S_MODELS:
        return {}
    by_uid = {
        str(row["uid"]): dict(row)
        for path in aime4s_score_files(paths, model, seed)
        for row in load_jsonl(path)
        if reusable_score(row)
    }
    copied: dict[str, int] = {}
    for dataset in AIME_DATASETS:
        wanted = {
            str(row["uid"]) for row in job_rows(paths, model, dataset, seed)
        }
        rows = [by_uid[uid] for uid in wanted if uid in by_uid]
        if not rows:
            continue
        score_dir = plws_score_dir(paths, model, dataset, seed)
        target = score_dir / "shard_0.jsonl"
        existing = {
            str(row["uid"]): dict(row)
            for directory in plws_score_read_dirs(paths, model, dataset, seed)
            for path in (directory / "shard_0.jsonl", directory / "scores.jsonl")
            for row in load_jsonl(path)
            if reusable_score(row)
        }
        merged = dict(existing)
        added = 0
        for row in rows:
            uid = str(row["uid"])
            if uid in merged:
                continue
            merged[uid] = row
            added += 1
        if added or (rows and not target.is_file()):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "".join(
                    json.dumps(merged[uid], ensure_ascii=False) + "\n"
                    for uid in sorted(merged)
                ),
                encoding="utf-8",
            )
        copied[f"{dataset}:{FIRSTWIN}"] = added
    return copied


def prereq_task(model: str, dataset: str, seed: int) -> MatrixTask:
    phase = f"{model_lane(model)}_prereq"
    return MatrixTask(
        task_id=f"prereq__{model}__{dataset}__s{seed}",
        phase=phase,
        gpu_count=gpu_count(model),
        kind="prereq",
        model=model,
        dataset=dataset,
        seed=seed,
    )


def plws_task(
    model: str,
    dataset: str,
    seed: int,
    shard_id: int,
    num_shards: int,
    kind: str = FIRSTWIN,
) -> MatrixTask:
    phase = f"{model_lane(model)}_plws"
    shard = f"__s{shard_id}of{num_shards}" if num_shards > 1 else ""
    return MatrixTask(
        task_id=f"plws__{model}__{dataset}__s{seed}{shard}",
        phase=phase,
        gpu_count=gpu_count(model),
        kind="plws",
        model=model,
        dataset=dataset,
        seed=seed,
        run_kind=kind,
        shard_id=shard_id,
        num_shards=num_shards,
    )


def deer_task(model: str, dataset: str, seed: int) -> MatrixTask:
    phase = (
        "late_deer"
        if (model, dataset) in DEER_DEFERRED
        else f"{model_lane(model)}_deer"
    )
    return MatrixTask(
        task_id=f"deer__{model}__{dataset}__s{seed}",
        phase=phase,
        gpu_count=gpu_count(model),
        kind="deer",
        model=model,
        dataset=dataset,
        seed=seed,
    )


def build_tasks(
    paths: PLWSPaths,
    *,
    include_thirty: bool = True,
    models: Iterable[str] | None = None,
) -> list[MatrixTask]:
    chosen = tuple(models) if models is not None else (
        ALL_MODELS if include_thirty else FIVE_MODELS
    )
    tasks: list[MatrixTask] = []
    for model in chosen:
        if model not in ALL_MODELS:
            raise ValueError(f"unknown model {model}")
        for dataset in DATASETS:
            for seed in SEEDS:
                if needs_prereq(paths, model, dataset, seed):
                    tasks.append(prereq_task(model, dataset, seed))
                shards = shard_count(dataset)
                for shard_id in range(shards):
                    tasks.append(
                        plws_task(model, dataset, seed, shard_id, shards)
                    )
                tasks.append(deer_task(model, dataset, seed))
    return sorted(tasks, key=lambda task: (task.priority, task.task_id))


def task_complete(paths: PLWSPaths, task: MatrixTask) -> tuple[bool, str]:
    if task.kind == "prereq":
        if needs_prereq(paths, task.model, task.dataset, task.seed):
            return False, "windows or PUMA missing"
        if jobs_present(paths, task.model, task.dataset, task.seed):
            return True, "jobs ready"
        return True, "prereq no longer required"
    if task.kind == "plws":
        return plws_complete(
            paths,
            task.model,
            task.dataset,
            task.seed,
            shard_id=task.shard_id,
            num_shards=task.num_shards,
        )
    if task.kind == "deer":
        return deer_complete(paths, task.model, task.dataset, task.seed)
    return False, f"unknown task kind {task.kind}"


def first_open_phase(
    tasks: Iterable[MatrixTask],
    finished: Mapping[str, str],
) -> str | None:
    remaining = {task.phase for task in tasks if task.task_id not in finished}
    for phase in PHASES:
        if phase in remaining:
            return phase
    return None


def task_ready(paths: PLWSPaths, task: MatrixTask) -> bool:
    if task.kind != "plws":
        return True
    if jobs_present(paths, task.model, task.dataset, task.seed):
        return True
    return task.model in FROZEN_FULLCOT


def dispatch_blocked_phases(
    pending: Iterable[MatrixTask],
    running_phases: Iterable[str] = (),
) -> frozenset[str]:
    phases = {task.phase for task in pending} | set(running_phases)
    if any(phase.startswith("thirty_") for phase in phases):
        return frozenset({"late_deer"})
    return frozenset()


def next_dispatchable(
    pending: Iterable[MatrixTask],
    free_gpus: int,
    paths: PLWSPaths,
    blocked_phases: Iterable[str] = (),
) -> MatrixTask | None:
    blocked = set(blocked_phases)
    ready = [
        task
        for task in pending
        if task.gpu_count <= free_gpus
        and task.phase not in blocked
        and task_ready(paths, task)
    ]
    if not ready:
        return None
    return min(ready, key=lambda task: (task.priority, task.task_id))


def fill_dispatch(
    pending: list[MatrixTask],
    free_gpus: int,
    paths: PLWSPaths,
    blocked_phases: Iterable[str] = (),
) -> list[MatrixTask]:
    """Pick as many ready tasks as fit. Earlier phases first; skip unready PLWS."""

    remaining = list(pending)
    taken: list[MatrixTask] = []
    free = free_gpus
    while True:
        fit = next_dispatchable(
            remaining, free, paths, blocked_phases=blocked_phases
        )
        if fit is None:
            break
        remaining.remove(fit)
        taken.append(fit)
        free -= fit.gpu_count
    return taken


def _pid_alive(pid: int) -> bool:
    try:
        state = ""
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                state = line.split()[1]
                break
        if not state or state.startswith("Z"):
            return False
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError) as exc:
        return isinstance(exc, PermissionError)
    return True


def blockers_idle(paths: PLWSPaths) -> tuple[bool, str]:
    busy: list[str] = []
    for relative in BLOCKER_STATUS:
        path = paths.root / relative
        if not path.is_file():
            continue
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            busy.append(f"{relative}: unreadable")
            continue
        live = [
            item
            for item in payload.get("running") or []
            if isinstance(item, dict) and _pid_alive(int(item.get("pid") or 0))
        ]
        pending = list(payload.get("pending") or [])
        if live or pending:
            busy.append(
                f"{relative}: running={len(live)} pending={len(pending)}"
            )
    if busy:
        return False, "; ".join(busy)
    return True, "current pre/AIME queues idle"


def summarize(tasks: Iterable[MatrixTask]) -> dict[str, int]:
    counts: dict[str, int] = {phase: 0 for phase in PHASES}
    counts["total"] = 0
    for task in tasks:
        counts[task.phase] += 1
        counts["total"] += 1
    return counts
