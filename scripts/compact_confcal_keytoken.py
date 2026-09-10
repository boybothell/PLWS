#!/usr/bin/env python3
"""Safely compact historical keytoken top-k dumps.

Raw solver/small files are read in lockstep with bounded memory.  This tool
never deletes raw data.  It emits a deletion manifest only after every raw
pair has a complete, verified compact artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_confcal_keytoken import merge_row, score_from_window  # noqa: E402

RAW_ROOT = ROOT / "results/confcal_judge/v2/keytoken"
STAGING_ROOT = ROOT / "results/archive/confcal/keytoken_compact_staging"


def key(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row["question_idx"]), int(row["decision_step"]), str(row["answer"])


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def same_identity(recorded: dict[str, Any], current: dict[str, Any]) -> bool:
    try:
        same_path = Path(recorded["path"]).resolve() == Path(current["path"]).resolve()
    except (KeyError, OSError):
        return False
    return (
        same_path
        and recorded.get("bytes") == current.get("bytes")
        and recorded.get("mtime_ns") == current.get("mtime_ns")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_pairs(raw_root: Path) -> list[tuple[str, int, Path, Path]]:
    pairs: list[tuple[str, int, Path, Path]] = []
    for dataset_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        for solver in sorted(dataset_dir.glob("solver_shard*.jsonl")):
            shard = int(solver.stem.removeprefix("solver_shard"))
            small = dataset_dir / f"small_shard{shard}.jsonl"
            if not small.is_file():
                raise RuntimeError(f"missing small pair for {solver}")
            pairs.append((dataset_dir.name, shard, solver, small))
        unmatched = {
            int(path.stem.removeprefix("small_shard"))
            for path in dataset_dir.glob("small_shard*.jsonl")
        } - {
            shard for dataset, shard, _, _ in pairs if dataset == dataset_dir.name
        }
        if unmatched:
            raise RuntimeError(f"missing solver pairs in {dataset_dir}: {sorted(unmatched)}")
    return pairs


def parsed_lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {error}") from error
            yield line_number, row


def parsed_headers(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Parse metadata preceding the huge final ``tokens`` field."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            prefix = line.split(', "tokens"', 1)[0]
            if prefix != line:
                prefix = prefix.rstrip().rstrip(",") + "}"
            try:
                row = json.loads(prefix)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid metadata JSON at {path}:{line_number}: {error}"
                ) from error
            yield line_number, row


def header_from_raw(raw: bytes, path: Path, line_number: int) -> dict[str, Any]:
    prefix = raw.split(b', "tokens"', 1)[0]
    if prefix != raw:
        prefix = prefix.rstrip().rstrip(b",") + b"}"
    try:
        return json.loads(prefix)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid metadata JSON at {path}:{line_number}: {error}"
        ) from error


def header_index(path: Path) -> dict[tuple[int, int, str], tuple[int, int, str, int]]:
    """Index key -> (offset, byte length, status, line number)."""
    indexed: dict[tuple[int, int, str], tuple[int, int, str, int]] = {}
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            header = header_from_raw(raw, path, line_number)
            row_key = key(header)
            if row_key in indexed:
                raise RuntimeError(f"duplicate raw key at {path}:{line_number}")
            indexed[row_key] = (
                handle.tell() - len(raw),
                len(raw),
                str(header.get("status") or ""),
                line_number,
            )
    return indexed


def adopt_legacy(
    dataset: str,
    shard: int,
    solver_path: Path,
    small_path: Path,
    staging_root: Path,
) -> Path:
    """Adopt an existing canonical compact shard after raw-key validation."""
    legacy = solver_path.parent / f"scores_shard{shard}.jsonl"
    if not legacy.is_file():
        raise RuntimeError(f"no legacy compact shard to adopt: {legacy}")
    destination_dir = staging_root / dataset
    destination = destination_dir / legacy.name
    metadata_path = destination.with_suffix(".meta.json")
    if destination.exists() or metadata_path.exists():
        raise RuntimeError(f"refusing to overwrite existing compact artifact: {destination}")

    print(f"indexing solver headers for {dataset} shard{shard}", flush=True)
    solver_index = header_index(solver_path)
    print(f"indexing small headers for {dataset} shard{shard}", flush=True)
    small_index = header_index(small_path)
    if solver_index.keys() != small_index.keys():
        raise RuntimeError(
            f"raw key-set mismatch: solver={len(solver_index)} small={len(small_index)} "
            f"missing_small={len(solver_index.keys() - small_index.keys())} "
            f"missing_solver={len(small_index.keys() - solver_index.keys())}"
        )
    raw_keys = {
        row_key
        for row_key, solver_meta in solver_index.items()
        if solver_meta[2] == "ok" and small_index[row_key][2] == "ok"
    }
    input_rows = len(solver_index)
    skipped_rows = input_rows - len(raw_keys)

    legacy_keys = set(compact_index(legacy))
    if raw_keys != legacy_keys:
        raise RuntimeError(
            f"legacy compact coverage mismatch: raw_ok={len(raw_keys)} "
            f"legacy={len(legacy_keys)} missing={len(raw_keys - legacy_keys)} "
            f"extra={len(legacy_keys - raw_keys)}"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise RuntimeError(f"stale partial output must be inspected first: {temporary}")
    try:
        with legacy.open("rb") as source, temporary.open("xb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    atomic_json(
        metadata_path,
        {
            "schema_version": 1,
            "dataset": dataset,
            "shard": shard,
            "adopted_legacy": True,
            "solver": file_identity(solver_path),
            "small": file_identity(small_path),
            "input_rows": input_rows,
            "output_rows": len(raw_keys),
            "skipped_rows": skipped_rows,
            "legacy": file_identity(legacy),
            "output": {**file_identity(destination), "sha256": sha256(destination)},
        },
    )
    return destination


def compact_one(
    dataset: str,
    shard: int,
    solver_path: Path,
    small_path: Path,
    staging_root: Path,
    window: int,
) -> Path:
    destination_dir = staging_root / dataset
    destination = destination_dir / f"scores_shard{shard}.jsonl"
    metadata_path = destination.with_suffix(".meta.json")
    if destination.exists() or metadata_path.exists():
        raise RuntimeError(f"refusing to overwrite existing compact artifact: {destination}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise RuntimeError(f"stale partial output must be inspected first: {temporary}")

    input_rows = output_rows = skipped_rows = 0
    print(f"indexing small headers for {dataset} shard{shard}", flush=True)
    small_index = header_index(small_path)
    seen: set[tuple[int, int, str]] = set()
    try:
        with (
            temporary.open("x", encoding="utf-8") as output,
            solver_path.open("rb") as solver_handle,
            small_path.open("rb") as small_handle,
        ):
            for left_line, raw in enumerate(solver_handle, 1):
                if not raw.strip():
                    continue
                left_header = header_from_raw(raw, solver_path, left_line)
                left_key = key(left_header)
                if left_key in seen:
                    raise RuntimeError(f"duplicate raw key at {solver_path}:{left_line}")
                seen.add(left_key)
                input_rows += 1
                small_meta = small_index.get(left_key)
                if small_meta is None:
                    raise RuntimeError(
                        f"solver key missing from small raw at {solver_path}:{left_line}: "
                        f"{left_key!r}"
                    )
                if left_header.get("status") != "ok" or small_meta[2] != "ok":
                    skipped_rows += 1
                    continue
                left = json.loads(raw)
                small_handle.seek(small_meta[0])
                right_raw = small_handle.read(small_meta[1])
                right = json.loads(right_raw)
                summary = merge_row(
                    left["tokens"],
                    right["tokens"],
                    int(left.get("boxed_offset") or 0),
                    window,
                )
                payload = {
                    "question_idx": left_key[0],
                    "decision_step": left_key[1],
                    "answer": left_key[2],
                    "geo_conf": left.get("geo_conf"),
                    "status": "ok",
                    "keytoken": summary,
                    "score_big_js0": score_from_window(summary, 0.0, "big"),
                    "score_small_js0": score_from_window(summary, 0.0, "small"),
                    "score_big_js025": score_from_window(summary, 0.25, "big"),
                    "score_small_js025": score_from_window(summary, 0.25, "small"),
                }
                output.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")
                output_rows += 1
                if input_rows % 100 == 0:
                    print(
                        f"{dataset} shard{shard}: input={input_rows} "
                        f"output={output_rows} skipped={skipped_rows}",
                        flush=True,
                    )
            missing_solver = small_index.keys() - seen
            if missing_solver:
                raise RuntimeError(
                    f"{len(missing_solver)} small keys are missing from solver raw; "
                    f"first={next(iter(missing_solver))!r}"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "shard": shard,
        "window": window,
        "solver": file_identity(solver_path),
        "small": file_identity(small_path),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "skipped_rows": skipped_rows,
        "output": {
            **file_identity(destination),
            "sha256": sha256(destination),
        },
    }
    atomic_json(metadata_path, metadata)
    return destination


def equal_value(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equal_value(left[name], right[name], tolerance) for name in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            equal_value(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        a, b = float(left), float(right)
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def compact_index(path: Path) -> dict[tuple[int, int, str], str]:
    indexed: dict[tuple[int, int, str], str] = {}
    for line_number, row in parsed_lines(path):
        row_key = key(row)
        if row_key in indexed:
            raise RuntimeError(f"duplicate compact key at {path}:{line_number}")
        canonical = json.dumps(
            row, ensure_ascii=False, allow_nan=True, sort_keys=True, separators=(",", ":")
        ).encode()
        indexed[row_key] = hashlib.sha256(canonical).hexdigest()
    return indexed


def compare_compact(left: Path, right: Path) -> tuple[bool, str]:
    right_offsets: dict[tuple[int, int, str], tuple[int, int]] = {}
    with right.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row_key = key(json.loads(raw))
            if row_key in right_offsets:
                raise RuntimeError(f"duplicate compact key at {right}:{line_number}")
            right_offsets[row_key] = (handle.tell() - len(raw), len(raw))

    seen: set[tuple[int, int, str]] = set()
    mismatched: list[tuple[int, int, str]] = []
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        for line_number, raw in enumerate(left_handle, 1):
            if not raw.strip():
                continue
            left_row = json.loads(raw)
            row_key = key(left_row)
            if row_key in seen:
                raise RuntimeError(f"duplicate compact key at {left}:{line_number}")
            seen.add(row_key)
            location = right_offsets.get(row_key)
            if location is None:
                continue
            right_handle.seek(location[0])
            right_row = json.loads(right_handle.read(location[1]))
            if not equal_value(left_row, right_row):
                mismatched.append(row_key)
    if seen != right_offsets.keys():
        missing = len(right_offsets.keys() - seen)
        extra = len(seen - right_offsets.keys())
        return False, f"key-set mismatch missing={missing} extra={extra}"
    if mismatched:
        return False, f"value mismatch for {len(mismatched)} keys, first={mismatched[0]!r}"
    return True, f"{len(seen)} rows equal within tolerance"


def verify(
    raw_root: Path,
    staging_root: Path,
    pairs: list[tuple[str, int, Path, Path]] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    raw_delete_candidates: list[str] = []
    all_valid = True
    selected_pairs = raw_pairs(raw_root) if pairs is None else pairs
    for dataset, shard, solver, small in selected_pairs:
        compact = staging_root / dataset / f"scores_shard{shard}.jsonl"
        metadata_path = compact.with_suffix(".meta.json")
        row: dict[str, Any] = {
            "dataset": dataset,
            "shard": shard,
            "compact": str(compact),
            "valid": False,
        }
        if not compact.is_file() or not metadata_path.is_file():
            row["reason"] = "compact or metadata missing"
            all_valid = False
            results.append(row)
            continue
        metadata = json.loads(metadata_path.read_text())
        if not same_identity(metadata.get("solver") or {}, file_identity(solver)) or not same_identity(
            metadata.get("small") or {}, file_identity(small)
        ):
            row["reason"] = "raw file identity changed after compaction"
            all_valid = False
            results.append(row)
            continue
        if metadata.get("output", {}).get("sha256") != sha256(compact):
            row["reason"] = "compact sha256 mismatch"
            all_valid = False
            results.append(row)
            continue
        legacy = raw_root / dataset / f"scores_shard{shard}.jsonl"
        if legacy.is_file():
            same, reason = compare_compact(compact, legacy)
            row["legacy_comparison"] = reason
            if not same:
                row["reason"] = "legacy compact comparison failed"
                all_valid = False
                results.append(row)
                continue
        row["valid"] = True
        row["reason"] = "validated"
        raw_delete_candidates.extend((str(solver), str(small)))
        results.append(row)

    manifest = {
        "schema_version": 1,
        "raw_root": str(raw_root),
        "staging_root": str(staging_root),
        "delete_ready": all_valid and bool(results),
        "raw_files": raw_delete_candidates if all_valid else [],
        "verification": results,
        "note": "This manifest does not delete files. Deletion requires a separate explicit step.",
    }
    datasets = sorted({dataset for dataset, _, _, _ in selected_pairs})
    manifest_path = (
        staging_root / datasets[0] / "deletion_manifest.json"
        if len(datasets) == 1
        else staging_root / "deletion_manifest.json"
    )
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "compact", "adopt", "verify"))
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)
    parser.add_argument("--dataset")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--window", type=int, default=128)
    args = parser.parse_args()
    pairs = raw_pairs(args.raw_root)
    selected = [
        pair
        for pair in pairs
        if (args.dataset is None or pair[0] == args.dataset)
        and (args.shard is None or pair[1] == args.shard)
    ]
    if args.command == "plan":
        print(
            json.dumps(
                [
                    {
                        "dataset": dataset,
                        "shard": shard,
                        "solver": file_identity(solver),
                        "small": file_identity(small),
                    }
                    for dataset, shard, solver, small in selected
                ],
                indent=2,
            )
        )
        return
    if args.command == "verify":
        manifest = verify(args.raw_root, args.staging_root, selected)
        print(json.dumps(manifest, indent=2))
        raise SystemExit(0 if manifest["delete_ready"] else 1)
    if not args.all and (args.dataset is None or args.shard is None):
        parser.error(f"{args.command} requires --all or both --dataset and --shard")
    for dataset, shard, solver, small in selected:
        if args.command == "adopt":
            output = adopt_legacy(dataset, shard, solver, small, args.staging_root)
            print(f"adopted {dataset} shard{shard} -> {output}", flush=True)
        else:
            output = compact_one(dataset, shard, solver, small, args.staging_root, args.window)
            print(f"compacted {dataset} shard{shard} -> {output}", flush=True)


if __name__ == "__main__":
    main()
