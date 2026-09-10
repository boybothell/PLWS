"""Canonical PLWS run paths and atomic metadata writes."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _path_component(value: str, label: str) -> str:
    if not value or value in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty path component")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must not contain path separators")
    return value


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Paths for ``results/runs/<method>/<experiment>/<run_id>/``."""

    results_root: Path
    method: str
    experiment: str
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_root", Path(self.results_root))
        _path_component(self.method, "method")
        _path_component(self.experiment, "experiment")
        _path_component(self.run_id, "run_id")

    @property
    def directory(self) -> Path:
        return self.results_root / "runs" / self.method / self.experiment / self.run_id

    @property
    def manifest(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def status(self) -> Path:
        return self.directory / "status.json"

    @property
    def artifacts(self) -> Path:
        return self.directory / "artifacts"


def run_paths(
    results_root: str | Path,
    *,
    method: str,
    experiment: str,
    run_id: str,
) -> RunPaths:
    """Build canonical paths without creating directories or migrating data."""

    return RunPaths(Path(results_root), method, experiment, run_id)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Stable identity and inputs for one machine run."""

    run_id: str
    method: str
    experiment: str
    model: str
    dataset: str
    seed: int
    config: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RunStatus:
    """Mutable-in-time state stored separately from the immutable manifest."""

    state: RunState
    updated_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    message: str | None = None
    exit_code: int | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace a text file and fsync its contents."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
) -> None:
    """Atomically replace a JSON file using a temporary sibling file."""

    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )
    atomic_write_text(path, text + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically replace a JSONL file."""

    text = "".join(
        json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows
    )
    atomic_write_text(path, text)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def done_uids(paths: str | Path | Iterable[str | Path]) -> set[str]:
    """Collect terminal UIDs from one file, directory, or many of either."""

    if isinstance(paths, (str, Path)):
        sources = (Path(paths),)
    else:
        sources = tuple(Path(path) for path in paths)
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(sorted(source.glob("scores_shard*.jsonl")))
            files.extend(sorted(source.glob("shard_*.jsonl")))
            files.extend(sorted(source.glob("scores.jsonl")))
        else:
            files.append(source)
    completed: set[str] = set()
    for path in files:
        for row in load_jsonl(path):
            if row.get("status") in {"ok", "too_long"} and row.get("uid") is not None:
                completed.add(str(row["uid"]))
    return completed
