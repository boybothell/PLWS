#!/usr/bin/env python3
"""Build deterministic mixed-tier jobs for the R1-7B lexicon ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "results"
    / "runs"
    / "plws"
    / "window_first"
    / "k_4"
    / "lexicon_core"
    / "r1_7b"
)
DEFAULT_OUT = (
    ROOT
    / "results"
    / "experiments"
    / "lexicon_ablation"
    / "jobs"
)
SOURCES = (
    ("math-500", "low"),
    ("math-500", "mix"),
    ("math-500", "high"),
    ("gpqa-diamond", "low"),
    ("gpqa-diamond", "mix"),
)
SMOKE_QUOTAS = {
    ("math-500", "low"): 4,
    ("math-500", "mix"): 4,
    ("math-500", "high"): 4,
    ("gpqa-diamond", "low"): 6,
    ("gpqa-diamond", "mix"): 2,
}
SCREEN_SIZE = 300
RARE_STRATUM_MAX = 10


def stable_order(row: dict[str, Any]) -> bytes:
    return hashlib.sha256(str(row["uid"]).encode()).digest()


def load_source(dataset: str, kind: str) -> list[dict[str, Any]]:
    path = SOURCE / dataset / "seed_42" / "jobs" / f"{kind}.jsonl"
    if not path.is_file():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    output = []
    for row in rows:
        packed = dict(row)
        packed["source_kind"] = str(row.get("kind") or kind)
        packed["kind"] = "pilot"
        output.append(packed)
    return sorted(output, key=stable_order)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def stratified_screen(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Draw a stable sample while retaining every member of rare strata."""

    strata: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["source_kind"]),
            bool(row.get("host_ok")),
        )
        strata[key].append(row)
    for values in strata.values():
        values.sort(key=stable_order)

    quotas = {
        key: len(values) if len(values) <= RARE_STRATUM_MAX else 0
        for key, values in strata.items()
    }
    remaining = SCREEN_SIZE - sum(quotas.values())
    common_total = sum(
        len(values)
        for values in strata.values()
        if len(values) > RARE_STRATUM_MAX
    )
    remainders = []
    for key, values in strata.items():
        if len(values) <= RARE_STRATUM_MAX:
            continue
        exact = remaining * len(values) / common_total
        quotas[key] = math.floor(exact)
        remainders.append((exact - quotas[key], key))
    for _, key in sorted(remainders, reverse=True)[
        : SCREEN_SIZE - sum(quotas.values())
    ]:
        quotas[key] += 1

    selected = [
        row
        for key, values in strata.items()
        for row in values[: quotas[key]]
    ]
    selected.sort(key=lambda row: (row["dataset"], stable_order(row)))
    labels = {
        f"{dataset}:{kind}:host_{'correct' if host_ok else 'wrong'}": quotas[key]
        for key in sorted(quotas)
        for dataset, kind, host_ok in (key,)
    }
    return selected, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    full = []
    smoke = []
    counts = {}
    for dataset, kind in SOURCES:
        rows = load_source(dataset, kind)
        counts[f"{dataset}:{kind}"] = len(rows)
        full.extend(rows)
        smoke.extend(rows[: SMOKE_QUOTAS[(dataset, kind)]])

    full = sorted(full, key=lambda row: (row["dataset"], stable_order(row)))
    smoke = sorted(smoke, key=lambda row: (row["dataset"], stable_order(row)))
    screen, screen_quotas = stratified_screen(full)
    write_jsonl(args.out_dir / "full.jsonl", full)
    write_jsonl(args.out_dir / "smoke.jsonl", smoke)
    write_jsonl(args.out_dir / "screen.jsonl", screen)
    manifest = {
        "model": "r1_7b",
        "seed": 42,
        "k": 4,
        "source_counts": counts,
        "full": len(full),
        "smoke": len(smoke),
        "screen": len(screen),
        "screen_strata": [
            "dataset",
            "source_kind",
            "host_ok",
        ],
        "screen_rare_stratum_max": RARE_STRATUM_MAX,
        "screen_quotas": screen_quotas,
        "smoke_quotas": {
            f"{dataset}:{kind}": count
            for (dataset, kind), count in SMOKE_QUOTAS.items()
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
