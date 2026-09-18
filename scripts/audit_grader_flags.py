#!/usr/bin/env python3
"""Re-grade stored correctness flags and report or repair the stale ones.

Two failure modes put wrong flags on disk: a job exported before PUMA could
supply gold, so it graded against nothing, and a machine whose grader lacked the
LaTeX backend, so every symbolic answer graded wrong. Both are indistinguishable
from a genuine miss once written, so the only way to trust a cell is to grade it
again.

Repair is one-directional on purpose. ``check_is_correct`` returning True is
positive evidence that two answers agree, while False can still be the
3-second timeout in its symbolic step. So a stored False may be promoted to
True, and a stored True is only reported for review, never demoted.

    python scripts/audit_grader_flags.py
    python scripts/audit_grader_flags.py --models r1_32b --datasets math-500
    python scripts/audit_grader_flags.py --models r1_32b,qwen3_32b --fix
    python scripts/audit_grader_flags.py \\
        --models r1_7b,nemotron_8b,r1_14b,r1_1p5b,r1_llama_8b,qwen3_4b,qwen3_8b

See docs/GRADER_ACCURACY.md before --fix on shared disks or historical cells.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_HINT = Path(os.environ.get("PLWS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT_HINT / "src"))

from plws.artifacts import load_jsonl  # noqa: E402
from plws.contest import OFFICIAL_FIRST_SEEDS, OFFICIAL_NEW_DATASETS  # noqa: E402
from plws.grading import grade_many, has_gold, require_grader  # noqa: E402
from plws.matrix import job_rows  # noqa: E402
from plws.paths import PLWSPaths  # noqa: E402
from plws.puma_grading import write_puma_verification_marker  # noqa: E402

DEFAULT_MODELS = ("qwen3_30b_a3b", "r1_32b", "qwen3_32b", "qwq_32b")
BACKUP_SUFFIX = ".bak_grader_audit"

# (flag field, predicted answer field) pairs inside a PUMA statistics row.
PUMA_FLAGS = (
    ("original_correct", "original_answer"),
    ("compressed_correct", "compressed_answer"),
)


class Finding:
    """What one audited artifact wants to change."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = total
        self.promote = 0
        self.review: list[str] = []
        self.no_gold = 0
        self.errors: list[str] = []

    @property
    def clean(self) -> bool:
        return not (self.promote or self.review or self.no_gold or self.errors)

    def line(self, fixed: bool) -> str:
        if self.clean:
            return f"OK    {self.label} n={self.total}"
        verb = "promoted" if fixed else "promotable"
        bits = [f"{verb}={self.promote}"]
        if self.review:
            bits.append(f"review={len(self.review)}")
        if self.no_gold:
            bits.append(f"no_gold={self.no_gold}")
        if self.errors:
            bits.append(f"grader_err={len(self.errors)}")
        return f"STALE {self.label} n={self.total} " + " ".join(bits)


def backup_once(path: Path) -> None:
    target = path.with_name(path.name + BACKUP_SUFFIX)
    if not target.exists():
        shutil.copy2(path, target)


def audit_records(
    label: str,
    records: list[dict],
    flag: str,
    answer_key: str,
    gold_of,
    workers: int,
) -> tuple[Finding, bool]:
    """Re-grade one flag across records. Returns the finding and whether records changed."""

    finding = Finding(label, len(records))
    indexed: list[int] = []
    pairs: list[tuple[Any, Any]] = []
    for index, record in enumerate(records):
        gold = gold_of(record)
        if not has_gold(gold):
            finding.no_gold += 1
            continue
        indexed.append(index)
        pairs.append((record.get(answer_key), gold))
    if not pairs:
        return finding, False

    changed = False
    for index, (fresh, error) in zip(indexed, grade_many(pairs, workers=workers)):
        record = records[index]
        stored = bool(record.get(flag))
        where = record.get("question_idx", record.get("uid", index))
        if error:
            finding.errors.append(f"q{where}: {error}")
            continue
        if fresh and not stored:
            record[flag] = True
            finding.promote += 1
            changed = True
        elif stored and not fresh:
            finding.review.append(f"q{where}")
    return finding, changed


def write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".audit")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".audit")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def leftover_shard_complete(shard: Path) -> bool:
    """Refuse to rewrite a leftover shard that is still being generated."""

    stem = shard.name
    if not (stem.startswith("shard_") and stem.endswith(".jsonl")):
        return True
    index = stem[len("shard_") : -len(".jsonl")]
    status_path = shard.with_name(f"status_shard_{index}.json")
    if not status_path.is_file():
        return True
    try:
        state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
    except (OSError, json.JSONDecodeError):
        return False
    return state == "succeeded"


def leftover_has_reuse_evidence(records: list[dict]) -> bool:
    return all(
        has_gold(record.get("gt"))
        and "gold_error" in record
        and not record.get("gold_error")
        for record in records
    )


def audit_plws_records(
    label: str,
    records: list[dict],
    gold_of,
    *,
    fix: bool,
    workers: int,
) -> tuple[Finding, bool]:
    """Re-grade PLWS rows and add the evidence required by the reuse gate."""

    finding, flags_changed = audit_records(
        label,
        records,
        "new_gold_ok",
        "new_answer",
        gold_of,
        workers,
    )
    blocked = bool(finding.review or finding.no_gold or finding.errors)
    if not fix or blocked:
        return finding, False

    changed = flags_changed
    for record in records:
        gold = gold_of(record)
        if not has_gold(gold):
            continue
        old = (record.get("gt"), record.get("gold_error"))
        record["gt"] = gold
        record["gold_error"] = ""
        changed = changed or old != (record["gt"], record["gold_error"])
    return finding, changed


def audit_puma_file(
    label: str,
    path: Path,
    *,
    fix: bool,
    workers: int,
) -> tuple[list[Finding], dict[int, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = rows.get("rows", [])
    gold_by_q = {int(row["question_idx"]): row.get("ground_truth") for row in rows}
    findings: list[Finding] = []
    dirty = False
    for flag, answer_key in PUMA_FLAGS:
        finding, changed = audit_records(
            f"{label}/{flag}",
            rows,
            flag,
            answer_key,
            lambda row: row.get("ground_truth"),
            workers,
        )
        findings.append(finding)
        dirty = dirty or changed
    if dirty and fix:
        backup_once(path)
        write_json(path, rows)
    return findings, gold_by_q


def audit_cell(
    paths: PLWSPaths,
    model: str,
    dataset: str,
    seed: int,
    *,
    fix: bool,
    workers: int,
    plws_evidence_only: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    gold_by_q: dict[int, Any] = {}
    stats_path = paths.puma_statistics_path(model, dataset, seed)
    if stats_path.is_file():
        if plws_evidence_only:
            rows = json.loads(stats_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                rows = rows.get("rows", [])
            gold_by_q = {
                int(row["question_idx"]): row.get("ground_truth") for row in rows
            }
        else:
            extra, gold_by_q = audit_puma_file(
                f"{model} {dataset} s{seed} puma",
                stats_path,
                fix=fix,
                workers=workers,
            )
            findings.extend(extra)
            blocked = any(
                finding.review or finding.no_gold or finding.errors
                for finding in extra
            )
            if fix and not blocked:
                write_puma_verification_marker(stats_path)

    # The main table prefers this file for PUMA Acc when it exists.
    backfill = (
        paths.results
        / "baselines"
        / "puma"
        / "backfill"
        / model
        / dataset
        / f"seed_{seed}"
        / "statistics.json"
    )
    if (
        not plws_evidence_only
        and backfill.is_file()
        and backfill.resolve() != stats_path.resolve()
    ):
        extra, backfill_gold = audit_puma_file(
            f"{model} {dataset} s{seed} puma_backfill",
            backfill,
            fix=fix,
            workers=workers,
        )
        findings.extend(extra)
        gold_by_q = {**gold_by_q, **backfill_gold}
        blocked = any(
            finding.review or finding.no_gold or finding.errors
            for finding in extra
        )
        if fix and not blocked:
            write_puma_verification_marker(backfill)

    score_dir = paths.score_dir(model, dataset, seed, "firstwin", k=4, lexicon="core")
    shards = sorted(score_dir.glob("shard_*.jsonl"))
    if shards:
        jobs_gold = {
            int(job["question_idx"]): job.get("gt")
            for job in job_rows(paths, model, dataset, seed)
            if job.get("uid")
        }

        def plws_gold(record: dict) -> Any:
            question = int(record.get("question_idx", -1))
            gold = jobs_gold.get(question)
            return gold if has_gold(gold) else gold_by_q.get(question)

        for shard in shards:
            records = load_jsonl(shard)
            if plws_evidence_only and leftover_has_reuse_evidence(records):
                continue
            finding, changed = audit_plws_records(
                f"{model} {dataset} s{seed} plws/{shard.name}/new_gold_ok",
                records,
                plws_gold,
                fix=fix,
                workers=workers,
            )
            findings.append(finding)
            if changed and fix:
                if not leftover_shard_complete(shard):
                    print(f"SKIP write {shard}: leftover still running", flush=True)
                    continue
                backup_once(shard)
                write_jsonl(shard, records)

    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(OFFICIAL_NEW_DATASETS))
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in OFFICIAL_FIRST_SEEDS)
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Promote stored False to True in place, keeping one backup per file.",
    )
    parser.add_argument(
        "--plws-evidence-only",
        action="store_true",
        help="Skip PUMA and only repair PLWS shards missing reusable grade evidence.",
    )
    parser.add_argument(
        "--exclude-cells",
        default="",
        help="Comma-separated model:dataset:seed cells that are currently being written.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--quiet", action="store_true", help="Only print cells that need attention."
    )
    args = parser.parse_args()

    require_grader()
    paths = PLWSPaths.discover(ROOT_HINT)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    excluded = {
        tuple(item.strip().rsplit(":", 2))
        for item in args.exclude_cells.split(",")
        if item.strip()
    }

    findings: list[Finding] = []
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                if (model, dataset, str(seed)) in excluded:
                    continue
                for finding in audit_cell(
                    paths,
                    model,
                    dataset,
                    seed,
                    fix=args.fix,
                    workers=args.workers,
                    plws_evidence_only=args.plws_evidence_only,
                ):
                    findings.append(finding)
                    if not args.quiet or not finding.clean:
                        print(finding.line(args.fix), flush=True)

    stale = [item for item in findings if not item.clean]
    print()
    print(f"audited {len(findings)} artifact flag(s), {len(stale)} stale")
    print(f"promotable={sum(item.promote for item in stale)}")
    print(f"review(stored True, regraded False)={sum(len(item.review) for item in stale)}")
    print(f"no_gold={sum(item.no_gold for item in stale)}")
    print(f"grader_errors={sum(len(item.errors) for item in stale)}")
    for item in stale:
        for error in item.errors[:3]:
            print(f"  ERR {item.label} {error}")
    unsafe = [
        item
        for item in stale
        if item.review or item.no_gold or item.errors
    ]
    if unsafe or (stale and not args.fix):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
