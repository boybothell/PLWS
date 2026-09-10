#!/usr/bin/env python3
"""Build larger five-dataset confirm jobs and seed reusable 7B scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WINDOW_ROOT = ROOT / "results" / "runs" / "plws" / "window_first" / "k_4" / "lexicon_core"
JOBS_DIR = ROOT / "results" / "experiments" / "lexicon_ablation" / "jobs"
CONFIRM_ROOT = ROOT / "results" / "experiments" / "lexicon_ablation" / "confirm"
SCREEN_ROOT = ROOT / "results" / "experiments" / "lexicon_ablation"
MODELS = ("r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b", "qwen3_8b")
DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
KINDS = ("low", "mix", "high")
RARE_STRATUM_MAX = 10
QUOTAS = {
    "confirm": {
        "math-500": None,
        "olympiadbench": 350,
        "gpqa-diamond": None,
        "aime24": None,
        "aime25": None,
    },
    "pre": {
        "math-500": 100,
        "olympiadbench": 100,
        "gpqa-diamond": 80,
        "aime24": None,
        "aime25": None,
    },
}
REUSE_CONFIGS = ("wait", "core_plus_let_me")
REUSE_SOURCES = (
    SCREEN_ROOT / "screen",
    SCREEN_ROOT / "screen_hard",
)


def stable_order(row: dict[str, Any]) -> bytes:
    return hashlib.sha256(str(row["uid"]).encode()).digest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def load_pool(model: str, dataset: str) -> list[dict[str, Any]]:
    output = []
    for kind in KINDS:
        path = WINDOW_ROOT / model / dataset / "seed_42" / "jobs" / f"{kind}.jsonl"
        for row in load_jsonl(path):
            packed = dict(row)
            packed["source_kind"] = str(row.get("kind") or kind)
            packed["kind"] = "pilot"
            packed.pop("k", None)
            output.append(packed)
    return sorted(output, key=stable_order)


def stratified_pick(
    rows: list[dict[str, Any]],
    size: int | None,
    forced_uids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not rows:
        return [], {}
    if size is None or size >= len(rows):
        labels = Counter(
            f"{row['dataset']}:{row['source_kind']}:host_"
            f"{'correct' if row.get('host_ok') else 'wrong'}"
            for row in rows
        )
        return list(rows), dict(sorted(labels.items()))

    strata: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["source_kind"]), bool(row.get("host_ok")))].append(row)
    for values in strata.values():
        values.sort(key=stable_order)

    quotas = {
        key: len(values) if len(values) <= RARE_STRATUM_MAX else 0
        for key, values in strata.items()
    }
    remaining = size - sum(quotas.values())
    common_total = sum(
        len(values) for values in strata.values() if len(values) > RARE_STRATUM_MAX
    )
    remainders = []
    if remaining > 0 and common_total:
        for key, values in strata.items():
            if len(values) <= RARE_STRATUM_MAX:
                continue
            exact = remaining * len(values) / common_total
            quotas[key] = math.floor(exact)
            remainders.append((exact - quotas[key], key))
        leftover = remaining - sum(
            quotas[key]
            for key, values in strata.items()
            if len(values) > RARE_STRATUM_MAX
        )
        for _, key in sorted(remainders, reverse=True)[: max(0, leftover)]:
            quotas[key] += 1

    selected: dict[str, dict[str, Any]] = {}
    for key, values in strata.items():
        take = len(values) if len(values) <= RARE_STRATUM_MAX else quotas[key]
        for row in values[:take]:
            selected[row["uid"]] = row
    for row in rows:
        if row["uid"] in (forced_uids or ()):
            selected[row["uid"]] = row

    picked = sorted(selected.values(), key=lambda row: (row["dataset"], stable_order(row)))
    labels = Counter(
        f"{row['dataset']}:{row['source_kind']}:host_"
        f"{'correct' if row.get('host_ok') else 'wrong'}"
        for row in picked
    )
    return picked, dict(sorted(labels.items()))


def seed_reuse(
    model: str,
    config: str,
    jobs: list[dict[str, Any]],
    out_root: Path,
) -> int:
    if model != "r1_7b" or config not in REUSE_CONFIGS:
        return 0
    wanted = {row["uid"] for row in jobs}
    seen: set[str] = set()
    copied: list[dict[str, Any]] = []
    sources = [
        *REUSE_SOURCES,
        SCREEN_ROOT / "confirm",
    ]
    for root in sources:
        for row in load_jsonl(root / config / "scores.jsonl"):
            uid = str(row.get("uid") or "")
            if uid not in wanted or uid in seen:
                continue
            if row.get("status") not in {"ok", "too_long"}:
                continue
            if row.get("protocol_id") != "puma-fullcot-32k-v2":
                continue
            seen.add(uid)
            copied.append(row)
        extra = root / "r1_7b" / config / "scores.jsonl"
        for row in load_jsonl(extra):
            uid = str(row.get("uid") or "")
            if uid not in wanted or uid in seen:
                continue
            if row.get("status") not in {"ok", "too_long"}:
                continue
            if row.get("protocol_id") != "puma-fullcot-32k-v2":
                continue
            seen.add(uid)
            copied.append(row)
    out = out_root / model / config / "scores.jsonl"
    if copied:
        copied.sort(key=lambda row: (str(row.get("dataset") or ""), str(row["uid"])))
        write_jsonl(out, copied)
    return len(copied)


def build_model(model: str, stage: str) -> dict[str, Any]:
    quotas = QUOTAS[stage]
    jobs: list[dict[str, Any]] = []
    pool_counts: dict[str, int] = {}
    picked_counts: dict[str, int] = {}
    strata: dict[str, int] = {}
    forced: set[str] = set()
    if model == "r1_7b" and stage == "confirm":
        for name in ("screen.jsonl", "screen_hard.jsonl"):
            forced.update(row["uid"] for row in load_jsonl(JOBS_DIR / name))
    missing = []
    for dataset in DATASETS:
        pool = load_pool(model, dataset)
        pool_counts[dataset] = len(pool)
        if not pool:
            missing.append(dataset)
            continue
        dataset_forced = {uid for uid in forced if uid.startswith(f"{model}:{dataset}:")}
        picked, labels = stratified_pick(pool, quotas[dataset], dataset_forced)
        picked_counts[dataset] = len(picked)
        strata.update(labels)
        jobs.extend(picked)
    jobs.sort(key=lambda row: (row["dataset"], stable_order(row)))
    write_jsonl(JOBS_DIR / f"{stage}_{model}.jsonl", jobs)
    reused = {
        config: seed_reuse(model, config, jobs, SCREEN_ROOT / stage)
        for config in REUSE_CONFIGS
    }
    return {
        "model": model,
        "jobs": len(jobs),
        "pool": pool_counts,
        "picked": picked_counts,
        "missing_datasets": missing,
        "strata": strata,
        "reused_scores": reused,
        "forced_existing_uids": len(forced) if model == "r1_7b" else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--stage", choices=tuple(QUOTAS), default="confirm")
    args = parser.parse_args()
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    reports = [build_model(model, args.stage) for model in models]
    quotas = QUOTAS[args.stage]
    manifest = {
        "stage": args.stage,
        "protocol_id": "puma-fullcot-32k-v2",
        "seed": 42,
        "k": 4,
        "sampling_seed": 20260904,
        "quotas": {
            dataset: "all_lockable" if quota is None else quota
            for dataset, quota in quotas.items()
        },
        "models": reports,
    }
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    (JOBS_DIR / f"manifest_{args.stage}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
