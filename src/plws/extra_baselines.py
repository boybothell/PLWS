"""Canonical Answer Convergence / Dynasor cell identity and completeness."""

from __future__ import annotations

import json
from pathlib import Path

from plws.contest import MODELS, OFFICIAL_FIRST_SEEDS, OFFICIAL_NEW_DATASETS
from plws.protocol import (
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROMPT_RESERVE_TOKENS,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)

EXTRA_BASELINE_MODELS = ("r1_1p5b", "r1_7b", "r1_14b")
EXTRA_BASELINE_METHODS = ("answer_convergence", "dynasor")
# Short cells first so two idle cards keep turning over.
EXTRA_BASELINE_DATASETS = (
    "amc23",
    "aime25",
    "hmmt25",
    "gpqa-diamond",
    "math-500",
    "olympiadbench",
)
AC_THRESHOLD = 10
DYNASOR_EFFORT = "mid"
DYNASOR_CHUNK_SIZE = 64
DYNASOR_CERTAINTY_THRESHOLD = 3


def extra_baseline_output_dir(
    root: Path, method: str, model_tag: str, dataset: str, seed: int
) -> Path:
    return (
        root
        / "results"
        / "baselines"
        / method
        / PROTOCOL_ID
        / model_tag
        / dataset
        / f"seed_{seed}"
    )


def extra_baseline_sample_dir(
    root: Path, model_tag: str, dataset: str, seed: int
) -> Path:
    return root / "samples" / model_tag / dataset / f"seed_{seed}"


def extra_baseline_expected_count(
    root: Path, model_tag: str, dataset: str, seed: int, *, limit: int = 0
) -> int:
    sample = extra_baseline_sample_dir(root, model_tag, dataset, seed) / "answers.json"
    rows = json.loads(sample.read_text(encoding="utf-8"))
    return min(len(rows), limit) if limit else len(rows)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extra_baseline_complete(
    root: Path,
    method: str,
    model_tag: str,
    dataset: str,
    seed: int,
    *,
    limit: int = 0,
) -> tuple[bool, str]:
    output_dir = extra_baseline_output_dir(root, method, model_tag, dataset, seed)
    manifest_path = output_dir / "manifest.json"
    final_path = output_dir / "final_answers.jsonl"
    if not manifest_path.is_file() or not final_path.is_file():
        return False, "missing manifest or final_answers.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = _load_jsonl(final_path)
        expected = extra_baseline_expected_count(
            root, model_tag, dataset, seed, limit=limit
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid output: {exc}"
    common = (
        manifest.get("method") == method
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("model_tag") == model_tag
        and manifest.get("dataset") == dataset
        and manifest.get("seed") == seed
        and manifest.get("fullcot_generation_tokens") == FULLCOT_GENERATION_TOKENS
        and manifest.get("prompt_reserve_tokens") == PROMPT_RESERVE_TOKENS
        and manifest.get("truncated_answer_fix_tokens") == TRUNCATED_ANSWER_FIX_TOKENS
        and manifest.get("max_model_len") == MAX_MODEL_LEN
        and manifest.get("expected_records") == expected
        and len(rows) == expected
        and all(row.get("protocol_id") == PROTOCOL_ID for row in rows)
    )
    if method == "answer_convergence":
        common = common and manifest.get("threshold") == AC_THRESHOLD
    elif method == "dynasor":
        method_cfg = manifest.get("dynasor") or {}
        common = common and (
            method_cfg.get("effort") == DYNASOR_EFFORT
            and method_cfg.get("chunk_size") == DYNASOR_CHUNK_SIZE
            and method_cfg.get("certainty_threshold") == DYNASOR_CERTAINTY_THRESHOLD
        )
    else:
        return False, f"unknown method {method}"
    return (True, f"{len(rows)} records") if common else (
        False,
        f"identity/count mismatch count={len(rows)} expected={expected}",
    )


def extra_baseline_task_id(method: str, model_tag: str, dataset: str, seed: int) -> str:
    prefix = "ansconv" if method == "answer_convergence" else method
    return f"{prefix}__{model_tag}__{dataset}__s{seed}"


def extra_baseline_sort_key(method: str, model_tag: str, dataset: str, seed: int):
    return (
        EXTRA_BASELINE_DATASETS.index(dataset)
        if dataset in EXTRA_BASELINE_DATASETS
        else len(EXTRA_BASELINE_DATASETS),
        EXTRA_BASELINE_MODELS.index(model_tag)
        if model_tag in EXTRA_BASELINE_MODELS
        else len(EXTRA_BASELINE_MODELS),
        EXTRA_BASELINE_METHODS.index(method)
        if method in EXTRA_BASELINE_METHODS
        else len(EXTRA_BASELINE_METHODS),
        OFFICIAL_FIRST_SEEDS.index(seed)
        if seed in OFFICIAL_FIRST_SEEDS
        else len(OFFICIAL_FIRST_SEEDS),
    )


def build_extra_baseline_plan(
    *,
    models: tuple[str, ...] = EXTRA_BASELINE_MODELS,
    datasets: tuple[str, ...] = EXTRA_BASELINE_DATASETS,
    seeds: tuple[int, ...] = OFFICIAL_FIRST_SEEDS,
    methods: tuple[str, ...] = EXTRA_BASELINE_METHODS,
) -> list[tuple[str, str, str, int]]:
    official = set(OFFICIAL_NEW_DATASETS)
    unknown_ds = [item for item in datasets if item not in official]
    if unknown_ds:
        raise ValueError(f"datasets must be official six: {unknown_ds}")
    unknown_models = [item for item in models if item not in MODELS]
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    unknown_methods = [item for item in methods if item not in EXTRA_BASELINE_METHODS]
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")
    tasks = [
        (method, model_tag, dataset, seed)
        for dataset in datasets
        for model_tag in models
        for method in methods
        for seed in seeds
    ]
    tasks.sort(key=lambda item: extra_baseline_sort_key(*item))
    return tasks
