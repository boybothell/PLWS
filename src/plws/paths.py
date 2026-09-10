"""Repository discovery and canonical PLWS paths with legacy read fallbacks."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEEDED_DENSE_DATASETS = frozenset(
    {"aime24", "aime25", "aime26", "amc23", "brumo25", "gsm8k", "hmmt25"}
)


def datasets_for_jobs(
    jobs: Iterable[Mapping[str, Any]],
    *,
    requested: str | None = None,
    allow_mixed: bool = False,
) -> tuple[str, ...]:
    """Resolve cell datasets and reject ambiguous canonical output."""

    datasets = sorted(
        {str(job["dataset"]) for job in jobs if job.get("dataset")}
    )
    if requested:
        datasets = [requested]
    if len(datasets) > 1 and not allow_mixed:
        raise ValueError(
            "mixed-dataset jobs need an explicit output or a dataset filter"
        )
    if not datasets and not allow_mixed:
        raise ValueError("empty jobs need a dataset to locate the canonical cell")
    return tuple(datasets)


def repository_root(start: str | Path | None = None) -> Path:
    """Return the PLWS repository root.

    ``PLWS_ROOT`` is authoritative. Otherwise walk upward from ``start`` (or
    this module) until the repository's ``src/plws`` package is found.
    """

    configured = os.environ.get("PLWS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    origin = Path(start or __file__).expanduser().resolve()
    current = origin if origin.is_dir() else origin.parent
    for candidate in (current, *current.parents):
        if (candidate / "src" / "plws").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate PLWS repository root from {origin}")


@dataclass(frozen=True, slots=True)
class PLWSPaths:
    """Canonical write paths plus protected historical read locations."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "PLWSPaths":
        return cls(repository_root(start))

    @property
    def results(self) -> Path:
        return self.root / "results"

    def window_run_root(self, *, k: int, lexicon: str = "core") -> Path:
        return (
            self.results
            / "runs"
            / "plws"
            / "window_first"
            / f"k_{int(k)}"
            / f"lexicon_{lexicon}"
        )

    def cell_dir(
        self,
        model: str,
        dataset: str,
        seed: int,
        *,
        k: int,
        lexicon: str = "core",
    ) -> Path:
        return (
            self.window_run_root(k=k, lexicon=lexicon)
            / model
            / dataset
            / f"seed_{seed}"
        )

    def jobs_dir(
        self,
        model: str,
        dataset: str,
        seed: int,
        *,
        k: int,
        lexicon: str = "core",
    ) -> Path:
        return self.cell_dir(
            model, dataset, seed, k=k, lexicon=lexicon
        ) / "jobs"

    def jobs_path(
        self,
        model: str,
        dataset: str,
        seed: int,
        kind: str,
        *,
        k: int,
        lexicon: str = "core",
    ) -> Path:
        return (
            self.jobs_dir(model, dataset, seed, k=k, lexicon=lexicon)
            / f"{kind}.jsonl"
        )

    def work_jobs_path(
        self,
        model: str,
        seed: int,
        kind: str,
        *,
        k: int,
        lexicon: str = "core",
        stem: str = "",
    ) -> Path:
        base = stem or "jobs"
        name = f"{base}.jsonl" if kind == "low" else f"{base}_{kind}.jsonl"
        return (
            self.window_run_root(k=k, lexicon=lexicon)
            / "work"
            / f"{model}_s{seed}"
            / name
        )

    def legacy_jobs_path(
        self, model: str, seed: int, kind: str, *, stem: str = ""
    ) -> Path:
        base = stem or "jobs"
        name = f"{base}.jsonl" if kind == "low" else f"{base}_{kind}.jsonl"
        return (
            self.results
            / "archive"
            / "legacy_layout"
            / "plws"
            / "leftover_jump"
            / f"{model}_s{seed}"
            / name
        )

    def resolve_jobs_path(
        self,
        model: str,
        dataset: str,
        seed: int,
        kind: str,
        *,
        k: int,
        lexicon: str = "core",
    ) -> Path:
        canonical = self.jobs_path(
            model, dataset, seed, kind, k=k, lexicon=lexicon
        )
        legacy = self.legacy_jobs_path(model, seed, kind)
        return canonical if canonical.is_file() else legacy

    def score_dir(
        self,
        model: str,
        dataset: str,
        seed: int,
        kind: str,
        *,
        k: int,
        lexicon: str = "core",
    ) -> Path:
        return (
            self.cell_dir(model, dataset, seed, k=k, lexicon=lexicon)
            / "scores"
            / kind
        )

    def score_path(
        self,
        model: str,
        dataset: str,
        seed: int,
        mode: str,
        kind: str,
        shard: int,
        *,
        k: int,
        lexicon: str = "core",
    ) -> Path:
        prefix = "shard_free_" if mode == "free" else "shard_"
        return (
            self.score_dir(
                model, dataset, seed, kind, k=k, lexicon=lexicon
            )
            / f"{prefix}{shard}.jsonl"
        )

    def legacy_score_dirs(
        self,
        model: str,
        seed: int,
        mode: str,
        kind: str,
        *,
        k: int,
        lexicon: str = "core",
    ) -> tuple[Path, ...]:
        """Return old score directories compatible with this exact config."""

        if lexicon != "core":
            return ()
        kind_suffix = "" if kind == "low" else f"_{kind}"
        run_name = f"{model}_s{seed}_{mode}{kind_suffix}"
        legacy_root = (
            self.results / "archive" / "legacy_layout" / "plws"
        )
        if int(k) == 4:
            return (
                legacy_root / "leftover_suppress_toend" / run_name,
                legacy_root / "leftover_suppress" / run_name,
            )
        return (
            legacy_root
            / "leftover_suppress_kablate"
            / f"k{int(k)}"
            / run_name,
            legacy_root
            / "leftover_suppress_kablate"
            / f"k_{int(k)}"
            / run_name,
        )

    def score_read_dirs(
        self,
        model: str,
        dataset: str,
        seed: int,
        mode: str,
        kind: str,
        *,
        k: int,
        lexicon: str = "core",
    ) -> tuple[Path, ...]:
        return (
            self.score_dir(
                model, dataset, seed, kind, k=k, lexicon=lexicon
            ),
            *self.legacy_score_dirs(
                model, seed, mode, kind, k=k, lexicon=lexicon
            ),
        )

    def puma_output_dir(self, model: str, dataset: str, seed: int) -> Path:
        root = self.results / "baselines" / "puma"
        if int(seed) == 42:
            return root / f"puma_offline_{model}" / dataset
        return root / f"puma_offline_{model}_s{seed}" / dataset

    def puma_statistics_path(self, model: str, dataset: str, seed: int) -> Path:
        candidates: list[Path] = []
        if model == "r1_7b" and dataset == "math-500" and int(seed) == 42:
            candidates.append(
                self.results
                / "baselines"
                / "official"
                / "math500_official"
                / "puma_ds7b"
                / "statistics.json"
            )
        candidates.append(
            self.puma_output_dir(model, dataset, seed) / "statistics.json"
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    def dense_trial_path(self, model: str, dataset: str, seed: int) -> Path:
        base = (
            self.results
            / "upstream"
            / "dense_trials"
            / f"dense_G_{model}"
            / dataset
        )
        seeded = base / f"seed_{seed}" / "dense_puma" / "trial_answers.json"
        flat = base / "dense_puma" / "trial_answers.json"
        if seeded.is_file():
            return seeded
        if int(seed) == 42 and dataset not in SEEDED_DENSE_DATASETS and flat.is_file():
            return flat
        return seeded if dataset in SEEDED_DENSE_DATASETS or int(seed) != 42 else flat

    def dense_g_path(self, model: str, dataset: str, seed: int) -> Path:
        base = (
            self.results
            / "upstream"
            / "dense_trials"
            / f"dense_G_{model}"
            / dataset
        )
        seeded = base / f"seed_{seed}" / "per_sample.json"
        flat = base / "per_sample.json"
        if seeded.is_file():
            return seeded
        if int(seed) == 42 and dataset not in SEEDED_DENSE_DATASETS and flat.is_file():
            return flat
        return seeded if dataset in SEEDED_DENSE_DATASETS or int(seed) != 42 else flat
