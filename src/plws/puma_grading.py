"""Verify PUMA correctness flags with PLWS's guarded grader."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plws.artifacts import atomic_write_json, utc_now
from plws.grading import grade_many, has_gold, require_grader

VERIFICATION_VERSION = "plws-unified-grader-v1"
BACKUP_SUFFIX = ".bak_grader_audit"
FLAGS = (
    ("original_correct", "original_answer"),
    ("compressed_correct", "compressed_answer"),
)


class PumaGradeVerificationError(RuntimeError):
    """PUMA statistics cannot safely be consumed as scored output."""


@dataclass(frozen=True, slots=True)
class Verification:
    total_rows: int
    promoted: int
    marker: Path | None


def marker_path(statistics: str | Path) -> Path:
    return Path(statistics).with_name("statistics.grader.json")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _marker_payload(statistics: Path, source: bytes) -> dict[str, Any]:
    stat = statistics.stat()
    return {
        "schema_version": 1,
        "verification_version": VERIFICATION_VERSION,
        "statistics": statistics.name,
        "statistics_sha256": _sha256(source),
        "statistics_size": stat.st_size,
        "statistics_mtime_ns": stat.st_mtime_ns,
        "verified_at": utc_now(),
    }


def puma_grades_verified(statistics: str | Path) -> bool:
    """Return whether the current statistics bytes have a fresh verifier marker."""

    statistics = Path(statistics)
    marker = marker_path(statistics)
    if not statistics.is_file() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        stat = statistics.stat()
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        payload.get("verification_version") == VERIFICATION_VERSION
        and payload.get("statistics") == statistics.name
        and int(payload.get("statistics_size", -1)) == stat.st_size
        and int(payload.get("statistics_mtime_ns", -1)) == stat.st_mtime_ns
    )


def write_puma_verification_marker(statistics: str | Path) -> Path:
    """Stamp statistics that the caller has just fully verified."""

    statistics = Path(statistics)
    source = statistics.read_bytes()
    marker = marker_path(statistics)
    atomic_write_json(marker, _marker_payload(statistics, source))
    return marker


def verify_puma_statistics(
    statistics: str | Path,
    *,
    fix: bool,
    workers: int = 16,
) -> Verification:
    """Re-grade both PUMA flags, safely promote misses, and stamp the file.

    False-to-True changes are safe positive evidence. A stored True that now
    grades False is never demoted automatically; it blocks the pipeline for
    review. Missing gold and grader errors also block the cell.
    """

    require_grader()
    statistics = Path(statistics)
    try:
        source = statistics.read_bytes()
        rows = json.loads(source)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PumaGradeVerificationError(
            f"cannot read PUMA statistics {statistics}: {exc}"
        ) from exc
    if not isinstance(rows, list) or not rows:
        raise PumaGradeVerificationError(
            f"PUMA statistics must be a non-empty list: {statistics}"
        )

    indexed: list[tuple[int, str]] = []
    pairs: list[tuple[Any, Any]] = []
    problems: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index} is not an object")
            continue
        where = row.get("question_idx", index)
        gold = row.get("ground_truth")
        if not has_gold(gold):
            problems.append(f"q{where} missing ground_truth")
            continue
        for flag, answer_key in FLAGS:
            if flag not in row:
                problems.append(f"q{where} missing {flag}")
                continue
            if answer_key not in row:
                problems.append(f"q{where} missing {answer_key}")
                continue
            indexed.append((index, flag))
            pairs.append((row.get(answer_key), gold))
    if problems:
        shown = "; ".join(problems[:8])
        raise PumaGradeVerificationError(
            f"invalid PUMA statistics {statistics}: {shown}"
        )

    promotions: list[tuple[int, str]] = []
    reviews: list[str] = []
    errors: list[str] = []
    for (index, flag), (fresh, error) in zip(
        indexed, grade_many(pairs, workers=workers)
    ):
        row = rows[index]
        where = row.get("question_idx", index)
        stored = bool(row.get(flag))
        if error:
            errors.append(f"q{where} {flag}: {error}")
        elif fresh and not stored:
            promotions.append((index, flag))
        elif stored and not fresh:
            reviews.append(f"q{where} {flag}")
    if errors or reviews:
        details = []
        if errors:
            details.append("grader errors: " + "; ".join(errors[:8]))
        if reviews:
            details.append("stored True regraded False: " + ", ".join(reviews[:8]))
        raise PumaGradeVerificationError(
            f"PUMA grade verification blocked for {statistics}: "
            + "; ".join(details)
        )
    if promotions and not fix:
        raise PumaGradeVerificationError(
            f"{statistics} has {len(promotions)} safely promotable stale flag(s)"
        )

    if promotions:
        for index, flag in promotions:
            rows[index][flag] = True
        if statistics.read_bytes() != source:
            raise PumaGradeVerificationError(
                f"PUMA statistics changed during verification: {statistics}"
            )
        backup = statistics.with_name(statistics.name + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(statistics, backup)
        atomic_write_json(statistics, rows, indent=None)

    marker = write_puma_verification_marker(statistics) if fix else None
    return Verification(
        total_rows=len(rows),
        promoted=len(promotions),
        marker=marker,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    try:
        result = verify_puma_statistics(
            args.statistics,
            fix=args.fix,
            workers=args.workers,
        )
    except PumaGradeVerificationError as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(
        f"[puma-grader] verified n={result.total_rows} "
        f"promoted={result.promoted} marker={result.marker or 'not written'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
