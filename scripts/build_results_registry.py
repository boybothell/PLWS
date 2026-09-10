#!/usr/bin/env python3
"""Create a read-only inventory of ``results/``.

This script never moves, rewrites, or deletes experiment artifacts.  Every
row is intentionally emitted with ``delete_ready: false``; deletion needs a
separate, specific validation manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def classify(name: str) -> tuple[str, bool, bool]:
    categories = {
        "baselines": ("frozen_baselines", True, False),
        "upstream": ("frozen_inputs", True, False),
        "runs": ("experiment_runs", True, False),
        "reports": ("derived_reports", True, True),
        "archive": ("protected_archive", True, False),
        "cache": ("rebuildable_cache", False, True),
        "registry": ("artifact_registry", True, True),
        "README.md": ("documentation", True, True),
    }
    return categories.get(name, ("unclassified", True, False))


def directory_stats(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        stat = directory.lstat()
        return {
            "bytes": 0 if directory.is_symlink() else stat.st_size,
            "files": 0 if directory.is_symlink() else 1,
            "directories": 0,
            "symlinks": 1 if directory.is_symlink() else 0,
            "newest_mtime": timestamp(stat.st_mtime),
        }

    total = files = directories = links = 0
    newest = directory.lstat().st_mtime
    for base, child_directories, child_files in os.walk(directory, followlinks=False):
        directories += len(child_directories)
        for filename in child_files:
            path = Path(base) / filename
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            newest = max(newest, stat.st_mtime)
            if path.is_symlink():
                links += 1
            else:
                files += 1
                total += stat.st_size
    return {
        "bytes": total,
        "files": files,
        "directories": directories,
        "symlinks": links,
        "newest_mtime": timestamp(newest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing registry files.")
    args = parser.parse_args()
    results = args.results_root.resolve()
    if not results.is_dir():
        raise SystemExit(f"results root not found: {results}")

    rows: list[dict[str, Any]] = []
    for path in sorted(results.iterdir(), key=lambda item: item.name):
        role, protected, regenerable = classify(path.name)
        rows.append(
            {
                "schema_version": 1,
                "path": str(path.relative_to(ROOT)),
                "role": role,
                "protected": protected,
                "regenerable": regenerable,
                "delete_ready": False,
                "delete_manifest": None,
                **directory_stats(path),
            }
        )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results),
        "entries": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "protected_bytes": sum(row["bytes"] for row in rows if row["protected"]),
        "delete_ready_bytes": 0,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    registry = results / "registry"
    atomic_write(registry / "runs.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    atomic_write(registry / "storage_report.json", json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
