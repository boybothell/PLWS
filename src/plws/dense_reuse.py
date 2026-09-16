"""Reuse PUMA trial answers when completing dense every-step trials.

PUMA and dense use the same trial-answer generator and decoding. PUMA only
generates steps admitted by its embedding filter; dense needs every step.
This module freezes a resume-safe plan that reuses PUMA's generated rows and
creates shard inputs containing only the steps still missing from dense.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from plws.artifacts import atomic_write_json

PLAN_SCHEMA_VERSION = 1
PLAN_NAME = "puma_trial_reuse_plan.json"
MISSING_QUESTIONS = "missing_steps_shard{shard}.json"
MISSING_TRIALS = "trial_answers_missing_shard{shard}.json"
LEGACY_QUESTIONS = "filtered_steps_shard{shard}.json"
LEGACY_TRIALS = "trial_answers_shard{shard}.json"

TrialKey = tuple[int, int]


def _load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(row: dict[str, Any]) -> TrialKey:
    return int(row["question_idx"]), int(row["stopped_len"])


def _expected_keys(questions: list[dict[str, Any]]) -> set[TrialKey]:
    return {
        (question_idx, stopped_len)
        for question_idx, question in enumerate(questions, start=1)
        for stopped_len in range(1, len(question.get("reasoning_steps") or []) + 1)
    }


def _usable_rows(
    rows: list[dict[str, Any]], expected: set[TrialKey]
) -> dict[TrialKey, dict[str, Any]]:
    reusable: dict[TrialKey, dict[str, Any]] = {}
    for row in rows:
        if row.get("skipped"):
            continue
        try:
            key = _key(row)
        except (KeyError, TypeError, ValueError):
            continue
        if key in expected:
            reusable[key] = dict(row)
    return reusable


def _remap_shard_rows(
    question_path: Path,
    trial_path: Path,
    expected: set[TrialKey],
) -> dict[TrialKey, dict[str, Any]]:
    questions = _load_list(question_path)
    local_to_abs = {
        local_idx: int(question["_abs_question_idx"])
        for local_idx, question in enumerate(questions, start=1)
    }
    remapped: list[dict[str, Any]] = []
    for source in _load_list(trial_path):
        if source.get("skipped"):
            continue
        row = dict(source)
        try:
            row["question_idx"] = local_to_abs[int(row["question_idx"])]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid local question_idx in {trial_path}: {source.get('question_idx')}"
            ) from exc
        remapped.append(row)
    return _usable_rows(remapped, expected)


def _verify_plan(
    plan: dict[str, Any],
    *,
    filtered_steps: Path,
    puma_trials: Path,
    num_shards: int,
) -> None:
    expected = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "filtered_steps_sha256": _sha256(filtered_steps),
        "puma_trials_sha256": _sha256(puma_trials),
        "num_shards": num_shards,
    }
    mismatches = {
        key: (plan.get(key), value)
        for key, value in expected.items()
        if plan.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "existing dense reuse plan does not match current inputs: "
            + ", ".join(
                f"{key}={actual!r} expected {wanted!r}"
                for key, (actual, wanted) in mismatches.items()
            )
        )


def prepare_reuse_plan(
    *,
    filtered_steps: Path,
    puma_trials: Path,
    shards_dir: Path,
    num_shards: int,
) -> dict[str, Any]:
    """Freeze missing-step shard inputs without invalidating resumable outputs."""

    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    plan_path = shards_dir / PLAN_NAME
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        _verify_plan(
            plan,
            filtered_steps=filtered_steps,
            puma_trials=puma_trials,
            num_shards=num_shards,
        )
        for shard in plan["active_shards"]:
            if not (shards_dir / MISSING_QUESTIONS.format(shard=shard)).is_file():
                raise RuntimeError(f"dense reuse plan is missing shard input {shard}")
        return plan

    questions = _load_list(filtered_steps)
    expected = _expected_keys(questions)
    puma_rows = _usable_rows(_load_list(puma_trials), expected)

    legacy_rows: dict[TrialKey, dict[str, Any]] = {}
    legacy_sources: list[dict[str, str]] = []
    for shard in range(num_shards):
        question_path = shards_dir / LEGACY_QUESTIONS.format(shard=shard)
        trial_path = shards_dir / LEGACY_TRIALS.format(shard=shard)
        if question_path.is_file() != trial_path.is_file():
            if trial_path.is_file():
                raise RuntimeError(
                    f"cannot remap existing dense shard without {question_path}"
                )
            continue
        if not trial_path.is_file():
            continue
        legacy_rows.update(_remap_shard_rows(question_path, trial_path, expected))
        legacy_sources.append(
            {"questions": question_path.name, "trials": trial_path.name}
        )

    completed = set(puma_rows) | set(legacy_rows)
    missing = expected - completed
    per_shard: list[int] = []
    active_shards: list[int] = []
    shards_dir.mkdir(parents=True, exist_ok=True)
    for shard in range(num_shards):
        payload: list[dict[str, Any]] = []
        missing_in_shard = 0
        for question_idx, question in enumerate(questions, start=1):
            if (question_idx - 1) % num_shards != shard:
                continue
            step_count = len(question.get("reasoning_steps") or [])
            mask = [
                (question_idx, stopped_len) in missing
                for stopped_len in range(1, step_count + 1)
            ]
            if not any(mask):
                continue
            item = dict(question)
            item["_abs_question_idx"] = question_idx
            item["should_generate_trial"] = mask
            payload.append(item)
            missing_in_shard += sum(mask)
        per_shard.append(missing_in_shard)
        if payload:
            active_shards.append(shard)
            atomic_write_json(
                shards_dir / MISSING_QUESTIONS.format(shard=shard), payload
            )

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "mode": "reuse-puma-generated-trials-and-fill-missing-v1",
        "filtered_steps": str(filtered_steps),
        "filtered_steps_sha256": _sha256(filtered_steps),
        "puma_trials": str(puma_trials),
        "puma_trials_sha256": _sha256(puma_trials),
        "num_shards": num_shards,
        "expected_steps": len(expected),
        "puma_reused_steps": len(set(puma_rows) - set(legacy_rows)),
        "legacy_dense_steps": len(legacy_rows),
        "missing_steps": len(missing),
        "missing_steps_per_shard": per_shard,
        "active_shards": active_shards,
        "legacy_sources": legacy_sources,
    }
    atomic_write_json(plan_path, plan)
    return plan


def merge_dense_trials(
    *,
    filtered_steps: Path,
    puma_trials: Path,
    shards_dir: Path,
    num_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Merge reused PUMA rows and generated missing rows with exact coverage."""

    plan_path = shards_dir / PLAN_NAME
    if not plan_path.is_file():
        raise RuntimeError(f"missing dense reuse plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _verify_plan(
        plan,
        filtered_steps=filtered_steps,
        puma_trials=puma_trials,
        num_shards=num_shards,
    )

    questions = _load_list(filtered_steps)
    expected = _expected_keys(questions)
    merged = _usable_rows(_load_list(puma_trials), expected)

    # Preserve already-generated dense rows when upgrading an interrupted cell.
    for source in plan.get("legacy_sources", []):
        merged.update(
            _remap_shard_rows(
                shards_dir / source["questions"],
                shards_dir / source["trials"],
                expected,
            )
        )

    generated_steps = 0
    for shard in plan["active_shards"]:
        question_path = shards_dir / MISSING_QUESTIONS.format(shard=shard)
        trial_path = shards_dir / MISSING_TRIALS.format(shard=shard)
        if not trial_path.is_file():
            raise RuntimeError(f"missing generated dense shard: {trial_path}")
        rows = _remap_shard_rows(question_path, trial_path, expected)
        generated_steps += len(rows)
        merged.update(rows)

    missing = sorted(expected - set(merged))
    extra = sorted(set(merged) - expected)
    if missing or extra:
        raise RuntimeError(
            f"dense trial coverage mismatch: missing={len(missing)} extra={len(extra)}"
            + (f" first_missing={missing[0]}" if missing else "")
            + (f" first_extra={extra[0]}" if extra else "")
        )

    ordered = [merged[key] for key in sorted(expected)]
    atomic_write_json(output, ordered)
    summary = {
        "output": str(output),
        "steps": len(ordered),
        "puma_reused_steps": int(plan["puma_reused_steps"]),
        "legacy_dense_steps": int(plan["legacy_dense_steps"]),
        "generated_missing_steps": generated_steps,
    }
    atomic_write_json(shards_dir / "puma_trial_reuse_result.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "merge"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--filtered-steps", type=Path, required=True)
        subparser.add_argument("--puma-trials", type=Path, required=True)
        subparser.add_argument("--shards-dir", type=Path, required=True)
        subparser.add_argument("--num-shards", type=int, required=True)
        if command == "merge":
            subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kwargs = {
        "filtered_steps": args.filtered_steps,
        "puma_trials": args.puma_trials,
        "shards_dir": args.shards_dir,
        "num_shards": args.num_shards,
    }
    if args.command == "prepare":
        result = prepare_reuse_plan(**kwargs)
    else:
        result = merge_dense_trials(output=args.output, **kwargs)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
