#!/usr/bin/env python3
"""Build the PLWS result layout without changing any legacy artifact.

The default is a read-only dry run.  ``--apply`` creates only new directories,
relative symlinks, manifests, and stream-split JSONL copies.  Existing files
are accepted only when their bytes match the planned output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
RUNS_REL = Path("runs/plws/window_first")
SCHEMA_VERSION = 1
TIERS = ("high", "mix", "low")
DIR_RE = re.compile(
    r"^(?P<model>.+)_s(?P<seed>\d+)"
    r"(?:_(?:suppress(?:_(?:high|mix))?|free))?$"
)
K_RE = re.compile(r"(?:^|_)k(?P<k>\d+)(?:_|$)")
SHARD_RE = re.compile(r"scores_shard(?P<shard>\d+)\.jsonl$")


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class Cell:
    k: int
    model: str
    dataset: str
    seed: int

    def relative(self) -> Path:
        return (
            RUNS_REL
            / f"k_{self.k}"
            / "lexicon_core"
            / self.model
            / self.dataset
            / f"seed_{self.seed}"
        )


@dataclass
class SourceUse:
    path: str
    sha256: str
    size: int
    rows: int = 0
    duplicate_rows: int = 0
    destination_rows: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SeenRow:
    source: str
    offset: int
    line_number: int


@dataclass
class OutputPlan:
    relative: Path
    rows: int = 0
    size: int = 0
    digest: Any = field(default_factory=hashlib.sha256, repr=False)
    sources: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    duplicate_sources: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()

    def add(self, raw: bytes, source: str) -> None:
        line = raw if raw.endswith(b"\n") else raw + b"\n"
        self.rows += 1
        self.size += len(line)
        self.digest.update(line)
        self.sources[source] += 1

    def add_duplicate(self, source: str) -> None:
        self.duplicate_sources[source] += 1


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def file_sha_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def relative_string(path: Path, results: Path) -> str:
    try:
        return str(path.relative_to(results))
    except ValueError:
        return str(path)


def parse_json_line(raw: bytes, source: Path, line_number: int) -> dict[str, Any]:
    try:
        row = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise MigrationError(f"{source}:{line_number}: expected a JSON object")
    return row


def uid_parts(row: dict[str, Any]) -> tuple[str | None, str | None, int | None]:
    uid = row.get("uid")
    if not isinstance(uid, str):
        return None, None, None
    parts = uid.split(":")
    if len(parts) < 4:
        return None, None, None
    try:
        seed = int(parts[2])
    except (TypeError, ValueError):
        seed = None
    return parts[0] or None, parts[1] or None, seed


def identify_row(
    row: dict[str, Any],
    source: Path,
    fallback_model: str,
    fallback_seed: int,
) -> tuple[str, str, int]:
    uid_model, uid_dataset, uid_seed = uid_parts(row)
    model = row.get("model") or uid_model or fallback_model
    dataset = row.get("dataset") or uid_dataset
    seed = row.get("seed", uid_seed if uid_seed is not None else fallback_seed)
    if not isinstance(model, str) or not model:
        raise MigrationError(f"{source}: cannot determine model")
    if not isinstance(dataset, str) or not dataset:
        raise MigrationError(f"{source}: cannot determine dataset (status row retained as error)")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise MigrationError(f"{source}: cannot determine seed") from exc
    if uid_model and uid_model != model:
        raise MigrationError(f"{source}: model disagrees with uid: {model!r} != {uid_model!r}")
    if uid_dataset and uid_dataset != dataset:
        raise MigrationError(f"{source}: dataset disagrees with uid: {dataset!r} != {uid_dataset!r}")
    if uid_seed is not None and uid_seed != seed:
        raise MigrationError(f"{source}: seed disagrees with uid: {seed} != {uid_seed}")
    return model, dataset, seed


def source_identity(path: Path) -> tuple[str, int]:
    match = DIR_RE.fullmatch(path.parent.name)
    if not match:
        raise MigrationError(f"{path}: cannot infer model/seed from parent directory")
    return match.group("model"), int(match.group("seed"))


def tier_from_path(path: Path) -> str:
    if path.stem.endswith("_high") or path.parent.name.endswith("_high"):
        return "high"
    if path.stem.endswith("_mix") or path.parent.name.endswith("_mix"):
        return "mix"
    return "low"


def normalize_tier(value: Any, fallback: str, source: Path) -> str:
    tier = str(value or fallback).lower()
    if tier == "mixed":
        tier = "mix"
    if tier not in TIERS:
        raise MigrationError(f"{source}: unsupported tier/kind {tier!r}")
    if value is not None and tier != fallback:
        raise MigrationError(f"{source}: row kind {tier!r} disagrees with path tier {fallback!r}")
    return tier


def normalized_line(raw: bytes) -> bytes:
    return raw if raw.endswith(b"\n") else raw + b"\n"


def job_source_sort_key(path: Path) -> tuple[str, int, str]:
    canonical = {"jobs.jsonl", "jobs_high.jsonl", "jobs_mix.jsonl"}
    return str(path.parent), 0 if path.name in canonical else 1, path.name


class Planner:
    def __init__(self, results: Path) -> None:
        self.results = results
        self.outputs: dict[Path, OutputPlan] = {}
        self.source_uses: dict[str, SourceUse] = {}
        self.cells: set[Cell] = set()
        self.input_rows = {"jobs": 0, "scores": 0}
        self.output_rows = {"jobs": 0, "scores": 0}
        self.identical_duplicates = {"jobs": 0, "scores": 0}
        self.job_seen: dict[tuple[Path, str], SeenRow] = {}
        self.score_seen: dict[tuple[Path, str], SeenRow] = {}
        self.skip_rows: set[tuple[str, int]] = set()
        self.duplicate_pairs: dict[tuple[str, str, str], int] = defaultdict(int)

    def output(self, relative: Path) -> OutputPlan:
        if relative not in self.outputs:
            self.outputs[relative] = OutputPlan(relative)
        return self.outputs[relative]

    def read_seen_row(self, seen: SeenRow) -> dict[str, Any]:
        source = Path(seen.source)
        if not source.is_absolute():
            source = self.results / source
        with source.open("rb") as handle:
            handle.seek(seen.offset)
            raw = handle.readline()
        return parse_json_line(raw, source, seen.line_number)

    def scan_source(
        self,
        source: Path,
        router: Any,
        source_kind: str,
    ) -> None:
        source_rel = relative_string(source, self.results)
        digest = hashlib.sha256()
        size = rows = duplicate_rows = 0
        destinations: dict[str, int] = defaultdict(int)
        with source.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                line_number += 1
                digest.update(raw)
                size += len(raw)
                if not raw.strip():
                    continue
                row = parse_json_line(raw, source, line_number)
                cell, relative = router(row, source)
                self.cells.add(cell)
                output = self.output(relative)
                uid = row.get("uid")
                keep = True
                if source_kind == "jobs":
                    if not isinstance(uid, str) or not uid:
                        raise MigrationError(f"{source}:{line_number}: jobs row has no usable uid")
                    key = (relative, uid)
                    previous = self.job_seen.get(key)
                    if previous is None:
                        self.job_seen[key] = SeenRow(source_rel, offset, line_number)
                    else:
                        previous_row = self.read_seen_row(previous)
                        if row != previous_row:
                            raise MigrationError(
                                f"{source}:{line_number}: conflicting jobs rows for uid {uid!r}; "
                                f"first seen at {previous.source}:{previous.line_number}"
                            )
                        keep = False
                        duplicate_rows += 1
                        self.identical_duplicates["jobs"] += 1
                        self.skip_rows.add((source_rel, line_number))
                        output.add_duplicate(source_rel)
                        self.duplicate_pairs[
                            (str(relative), previous.source, source_rel)
                        ] += 1
                elif isinstance(uid, str) and uid:
                    key = (relative, uid)
                    previous = self.score_seen.get(key)
                    if previous is not None:
                        raise MigrationError(
                            f"{source}:{line_number}: duplicate score uid {uid!r} in target "
                            f"{relative}; first seen at {previous.source}:{previous.line_number}"
                        )
                    self.score_seen[key] = SeenRow(source_rel, offset, line_number)
                if keep:
                    output.add(raw, source_rel)
                rows += 1
                destinations[str(relative)] += 1
        self.source_uses[source_rel] = SourceUse(
            path=source_rel,
            sha256=digest.hexdigest(),
            size=size,
            rows=rows,
            duplicate_rows=duplicate_rows,
            destination_rows=dict(sorted(destinations.items())),
        )
        self.input_rows[source_kind] += rows
        self.output_rows[source_kind] += rows - duplicate_rows

    def scan_jobs(self) -> None:
        root = self.results / "leftover_jump"
        if not root.is_dir():
            return
        for source in sorted(root.glob("*/*.jsonl"), key=job_source_sort_key):
            if not source.name.startswith("jobs"):
                continue
            fallback_model, fallback_seed = source_identity(source)
            path_tier = tier_from_path(source)
            file_k_match = K_RE.search(source.stem)
            file_k = int(file_k_match.group("k")) if file_k_match else 4

            def route(
                row: dict[str, Any],
                current: Path,
                *,
                model: str = fallback_model,
                seed: int = fallback_seed,
                tier: str = path_tier,
                k: int = file_k,
            ) -> tuple[Cell, Path]:
                actual_model, dataset, actual_seed = identify_row(row, current, model, seed)
                row_k = int(row.get("k", k))
                if row_k != k:
                    raise MigrationError(f"{current}: row k={row_k} disagrees with filename k={k}")
                actual_tier = normalize_tier(row.get("kind"), tier, current)
                cell = Cell(row_k, actual_model, dataset, actual_seed)
                return cell, cell.relative() / "jobs" / f"{actual_tier}.jsonl"

            self.scan_source(source, route, "jobs")

    def scan_scores_root(self, root: Path, k_from_path: bool) -> None:
        if not root.is_dir():
            return
        for source in sorted(root.glob("**/scores_shard*.jsonl")):
            match = SHARD_RE.fullmatch(source.name)
            if not match:
                continue
            fallback_model, fallback_seed = source_identity(source)
            path_tier = tier_from_path(source)
            if k_from_path:
                k_match = next((K_RE.search(part) for part in source.parts if K_RE.search(part)), None)
                if k_match is None:
                    raise MigrationError(f"{source}: cannot determine ablation k")
                source_k = int(k_match.group("k"))
            else:
                source_k = 4
            shard = int(match.group("shard"))
            shard_name = (
                f"shard_free_{shard}.jsonl"
                if source.parent.name.endswith("_free")
                else f"shard_{shard}.jsonl"
            )

            def route(
                row: dict[str, Any],
                current: Path,
                *,
                model: str = fallback_model,
                seed: int = fallback_seed,
                tier: str = path_tier,
                k: int = source_k,
                output_name: str = shard_name,
            ) -> tuple[Cell, Path]:
                actual_model, dataset, actual_seed = identify_row(row, current, model, seed)
                if "k" in row and int(row["k"]) != k:
                    raise MigrationError(f"{current}: row k={row['k']} disagrees with source k={k}")
                actual_tier = normalize_tier(row.get("kind"), tier, current)
                cell = Cell(k, actual_model, dataset, actual_seed)
                relative = cell.relative() / "scores" / actual_tier / output_name
                return cell, relative

            self.scan_source(source, route, "scores")

    def finish_empty_jobs(self) -> None:
        for cell in self.cells:
            for tier in TIERS:
                self.output(cell.relative() / "jobs" / f"{tier}.jsonl")


def iter_regular_files(root: Path) -> Iterator[Path]:
    for base, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        for name in sorted(files):
            path = Path(base) / name
            if path.is_file() and not path.is_symlink():
                yield path


def upstream_sources(results: Path) -> dict[str, list[Path]]:
    top = list(results.iterdir()) if results.is_dir() else []
    puma = sorted(path for path in top if path.is_dir() and path.name.startswith("puma_offline_"))
    official = results / "math500_official" / "puma_ds7b"
    if official.is_dir():
        puma.append(official)
    dense = sorted(path for path in top if path.is_dir() and path.name.startswith("dense_G_"))
    deer = sorted(path for path in top if path.is_dir() and path.name.startswith("deer"))
    math_official = results / "math500_official"
    if math_official.is_dir():
        deer.extend(
            sorted(
                path
                for path in math_official.iterdir()
                if path.is_dir() and path.name.startswith("deer")
            )
        )
    return {"puma_official": puma, "dense_trials": dense, "deer": deer}


def upstream_link_name(source: Path, results: Path) -> str:
    if source.parent == results:
        return source.name
    return "__".join(source.relative_to(results).parts)


def build_upstream_manifest(category: str, sources: Iterable[Path], results: Path) -> dict[str, Any]:
    records = []
    protected = category in {"puma_official", "deer"}
    for source in sources:
        files = []
        for path in iter_regular_files(source):
            sha256, size = file_sha_size(path)
            files.append(
                {
                    "path": str(path.relative_to(source)),
                    "sha256": sha256,
                    "size": size,
                }
            )
        records.append(
            {
                "legacy_path": relative_string(source, results),
                "link_name": upstream_link_name(source, results),
                "files": files,
                "files_count": len(files),
                "size": sum(item["size"] for item in files),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "protected": protected,
        "delete_ready": False,
        "legacy_sources": records,
    }


def output_record(plan: OutputPlan) -> dict[str, Any]:
    return {
        "path": str(plan.relative),
        "rows": plan.rows,
        "sha256": plan.sha256,
        "size": plan.size,
        "legacy_sources": [
            {"path": source, "rows": rows} for source, rows in sorted(plan.sources.items())
        ],
        "duplicate_sources": [
            {"path": source, "rows": rows}
            for source, rows in sorted(plan.duplicate_sources.items())
        ],
    }


def cell_documents(planner: Planner) -> dict[Path, bytes]:
    documents: dict[Path, bytes] = {}
    by_cell: dict[Cell, list[OutputPlan]] = defaultdict(list)
    for output in planner.outputs.values():
        for cell in planner.cells:
            try:
                output.relative.relative_to(cell.relative())
            except ValueError:
                continue
            by_cell[cell].append(output)
            break
    for cell in sorted(planner.cells):
        outputs = sorted(by_cell[cell], key=lambda item: str(item.relative))
        source_names = sorted(
            {
                name
                for output in outputs
                for name in (*output.sources, *output.duplicate_sources)
            }
        )
        sources = [planner.source_uses[name] for name in source_names]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "method": "plws",
            "policy": "window_first",
            "k": cell.k,
            "lexicon": "core",
            "model": cell.model,
            "dataset": cell.dataset,
            "seed": cell.seed,
            "delete_ready": False,
            "legacy_sources": [
                {
                    "path": source.path,
                    "sha256": source.sha256,
                    "size": source.size,
                    "rows": source.rows,
                    "duplicate_rows": source.duplicate_rows,
                    "destination_rows": {
                        path: rows
                        for path, rows in source.destination_rows.items()
                        if path.startswith(str(cell.relative()) + "/")
                    },
                }
                for source in sources
            ],
            "outputs": [output_record(output) for output in outputs],
        }
        status = {
            "schema_version": SCHEMA_VERSION,
            "state": "migrated",
            "complete": True,
            "row_counts": {
                "jobs": sum(item.rows for item in outputs if "/jobs/" in f"/{item.relative}"),
                "scores": sum(item.rows for item in outputs if "/scores/" in f"/{item.relative}"),
            },
            "legacy_source_count": len(sources),
        }
        documents[cell.relative() / "manifest.json"] = canonical_json(manifest)
        documents[cell.relative() / "status.json"] = canonical_json(status)
    return documents


def same_bytes(path: Path, expected_sha: str, expected_size: int) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    actual_sha, actual_size = file_sha_size(path)
    return actual_sha == expected_sha and actual_size == expected_size


def preflight_file(path: Path, expected: bytes | OutputPlan) -> str:
    if isinstance(expected, OutputPlan):
        sha256, size = expected.sha256, expected.size
    else:
        sha256, size = hashlib.sha256(expected).hexdigest(), len(expected)
    if not path.exists() and not path.is_symlink():
        return "create"
    if same_bytes(path, sha256, size):
        return "skip"
    raise MigrationError(f"refusing to overwrite inconsistent existing path: {path}")


def preflight_link(link: Path, source: Path) -> str:
    expected = os.path.relpath(source, link.parent)
    if not link.exists() and not link.is_symlink():
        return "create"
    if link.is_symlink() and os.readlink(link) == expected:
        return "skip"
    raise MigrationError(f"refusing to replace inconsistent existing path: {link}")


def required_cell_directories(cells: Iterable[Cell]) -> list[Path]:
    directories: set[Path] = set()
    for cell in cells:
        root = cell.relative()
        directories.update(
            {
                root,
                root / "jobs",
                root / "metrics",
                root / "scores",
                *(root / "scores" / tier for tier in TIERS),
            }
        )
    return sorted(directories)


def preflight_directory(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "create"
    if path.is_dir() and not path.is_symlink():
        return "skip"
    raise MigrationError(f"refusing to replace non-directory path: {path}")


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


class StagedWriters:
    def __init__(self, results: Path, plans: dict[Path, OutputPlan], needed: set[Path]) -> None:
        self.results = results
        self.plans = plans
        self.needed = needed
        self.paths: dict[Path, Path] = {}
        self.handles: OrderedDict[Path, Any] = OrderedDict()

    def write(self, relative: Path, raw: bytes) -> None:
        if relative not in self.needed:
            return
        if relative not in self.paths:
            target = self.results / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".part", dir=target.parent
            )
            os.close(descriptor)
            self.paths[relative] = Path(temporary)
        if relative in self.handles:
            handle = self.handles.pop(relative)
        else:
            handle = self.paths[relative].open("ab")
        self.handles[relative] = handle
        handle.write(normalized_line(raw))
        if len(self.handles) > 64:
            _, old = self.handles.popitem(last=False)
            old.close()

    def finish(self) -> None:
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self.handles.clear()
        for relative in self.needed:
            if relative not in self.paths:
                target = self.results / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".part", dir=target.parent
                )
                os.close(descriptor)
                self.paths[relative] = Path(temporary)
            temporary = self.paths[relative]
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            plan = self.plans[relative]
            if not same_bytes(temporary, plan.sha256, plan.size):
                raise MigrationError(f"staged output failed checksum validation: {relative}")

    def commit(self) -> None:
        for relative in sorted(self.needed):
            os.replace(self.paths[relative], self.results / relative)
        self.paths.clear()

    def cleanup(self) -> None:
        for handle in self.handles.values():
            handle.close()
        for path in self.paths.values():
            path.unlink(missing_ok=True)


def stream_outputs(planner: Planner, writers: StagedWriters) -> None:
    for source_name, source_use in sorted(planner.source_uses.items()):
        source = planner.results / source_name
        destinations = set(source_use.destination_rows)
        with source.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                if (source_name, line_number) in planner.skip_rows:
                    continue
                row = parse_json_line(raw, source, line_number)
                candidates = []
                for relative in destinations:
                    plan = planner.outputs[Path(relative)]
                    if source_name in plan.sources:
                        candidates.append(plan)
                matched = []
                uid = row.get("uid")
                dataset = row.get("dataset") or uid_parts(row)[1]
                kind = str(row.get("kind", "")).lower()
                for plan in candidates:
                    text = str(plan.relative)
                    if dataset and f"/{dataset}/" not in f"/{text}/":
                        continue
                    if kind == "mixed":
                        kind = "mix"
                    if kind and f"/{kind}" not in text:
                        continue
                    matched.append(plan)
                if len(matched) != 1:
                    raise MigrationError(
                        f"{source}:{line_number}: could not uniquely replay planned destination"
                    )
                writers.write(matched[0].relative, raw)


def create_relative_link(link: Path, source: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, link.parent)
    temporary = link.parent / f".{link.name}.{os.getpid()}.part"
    if temporary.exists() or temporary.is_symlink():
        raise MigrationError(f"temporary link already exists: {temporary}")
    os.symlink(target, temporary)
    os.replace(temporary, link)


def build_report(
    planner: Planner,
    upstream: dict[str, dict[str, Any]],
    file_states: dict[str, str],
) -> dict[str, Any]:
    accounting_matches = all(
        planner.input_rows[kind]
        == planner.output_rows[kind] + planner.identical_duplicates[kind]
        for kind in planner.input_rows
    )
    if not accounting_matches:
        raise MigrationError(
            "row accounting failed: "
            f"input={planner.input_rows}, unique_output={planner.output_rows}, "
            f"identical_duplicates={planner.identical_duplicates}"
        )
    def with_total(counts: dict[str, int]) -> dict[str, int]:
        return {**counts, "total": sum(counts.values())}
    duplicate_sources = [
        {
            "target": target,
            "kept_source": kept_source,
            "duplicate_source": duplicate_source,
            "rows": rows,
        }
        for (target, kept_source, duplicate_source), rows in sorted(
            planner.duplicate_pairs.items()
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "safe_copy_only": True,
        "legacy_files_modified": 0,
        "protected_categories": ["puma_official", "deer"],
        "row_accounting": {
            "equation": "input = unique_output + identical_duplicates",
            "input": with_total(planner.input_rows),
            "unique_output": with_total(planner.output_rows),
            "identical_duplicates": with_total(planner.identical_duplicates),
            "matches": accounting_matches,
        },
        "duplicate_rows": with_total(planner.identical_duplicates),
        "duplicate_sources": duplicate_sources,
        "cells": len(planner.cells),
        "jsonl_outputs": len(planner.outputs),
        "legacy_jsonl_sources": len(planner.source_uses),
        "upstream": {
            name: {
                "protected": manifest["protected"],
                "legacy_sources": len(manifest["legacy_sources"]),
                "files": sum(item["files_count"] for item in manifest["legacy_sources"]),
                "size": sum(item["size"] for item in manifest["legacy_sources"]),
            }
            for name, manifest in upstream.items()
        },
        "planned_paths": sorted(file_states),
        "errors": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Create the planned compatibility layout.")
    mode.add_argument("--dry-run", action="store_true", help="Read and validate only (the default).")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = args.results_root.resolve()
    if not results.is_dir():
        raise SystemExit(f"results root not found: {results}")

    planner = Planner(results)
    try:
        planner.scan_jobs()
        planner.scan_scores_root(results / "leftover_suppress_toend", k_from_path=False)
        planner.scan_scores_root(results / "leftover_suppress_kablate", k_from_path=True)
        planner.finish_empty_jobs()

        upstream = {
            category: build_upstream_manifest(category, sources, results)
            for category, sources in upstream_sources(results).items()
        }
        documents = cell_documents(planner)
        for category, manifest in upstream.items():
            documents[Path("upstream") / category / "manifest.json"] = canonical_json(manifest)

        states: dict[str, str] = {}
        needed_outputs: set[Path] = set()
        directories = required_cell_directories(planner.cells)
        for relative in directories:
            states[str(relative) + "/"] = preflight_directory(results / relative)
        for relative, plan in sorted(planner.outputs.items()):
            state = preflight_file(results / relative, plan)
            states[str(relative)] = state
            if state == "create":
                needed_outputs.add(relative)
        for relative, data in sorted(documents.items()):
            states[str(relative)] = preflight_file(results / relative, data)
        links: list[tuple[Path, Path, str]] = []
        for category, sources in upstream_sources(results).items():
            for source in sources:
                relative = Path("upstream") / category / upstream_link_name(source, results)
                state = preflight_link(results / relative, source)
                states[str(relative)] = state
                links.append((results / relative, source, state))

        report = build_report(planner, upstream, states)
        report_data = canonical_json(report)
        report_path = results / "migration_report.json"
        report_state = preflight_file(report_path, report_data)
        action_counts = {
            "create": sum(state == "create" for state in states.values())
            + (report_state == "create"),
            "skip": sum(state == "skip" for state in states.values())
            + (report_state == "skip"),
        }

        if not args.apply:
            print(
                json.dumps(
                    {"mode": "dry-run", "actions": action_counts, **report},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        for relative in directories:
            (results / relative).mkdir(parents=True, exist_ok=True)
        writers = StagedWriters(results, planner.outputs, needed_outputs)
        try:
            stream_outputs(planner, writers)
            writers.finish()
            writers.commit()
        except BaseException:
            writers.cleanup()
            raise
        for relative, data in sorted(documents.items()):
            if states[str(relative)] == "create":
                write_atomic(results / relative, data)
        for link, source, state in links:
            if state == "create":
                create_relative_link(link, source)
        if report_state == "create":
            write_atomic(report_path, report_data)
        print(
            json.dumps(
                {"mode": "apply", "actions": action_counts, **report},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    except MigrationError as exc:
        raise SystemExit(f"migration refused: {exc}") from exc


if __name__ == "__main__":
    main()
