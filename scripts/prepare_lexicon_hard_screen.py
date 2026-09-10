#!/usr/bin/env python3
"""Pick OlympiadBench + AIME25 jobs for the reduced lexicon pre-screen."""

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
DEFAULT_OUT = ROOT / "results" / "experiments" / "lexicon_ablation" / "jobs"
KINDS = ("low", "mix", "high")
OLY_SCREEN_SIZE = 150
RARE_STRATUM_MAX = 10
KEEP_CONFIGS = (
    "core",
    "core_plus_let_me",
    "wait",
    "let_me",
    "but",
)
DROP_CONFIGS = (
    "core_no_wait",
    "core_no_alternatively",
    "core_no_hmm",
    "alternatively",
    "hmm",
    "safe",
    "free",
)


def stable_order(row: dict[str, Any]) -> bytes:
    return hashlib.sha256(str(row["uid"]).encode()).digest()


def load_source(dataset: str, kind: str) -> list[dict[str, Any]]:
    path = SOURCE / dataset / "seed_42" / "jobs" / f"{kind}.jsonl"
    if not path.is_file():
        return []
    output = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        packed = json.loads(line)
        packed["source_kind"] = str(packed.get("kind") or kind)
        packed["kind"] = "pilot"
        output.append(packed)
    return sorted(output, key=stable_order)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def stratified_screen(
    rows: list[dict[str, Any]], size: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
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
    remaining = size - sum(quotas.values())
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
    leftover = size - sum(quotas.values())
    for _, key in sorted(remainders, reverse=True)[:leftover]:
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
    parser.add_argument("--oly-size", type=int, default=OLY_SCREEN_SIZE)
    args = parser.parse_args()

    oly: list[dict[str, Any]] = []
    aime: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for kind in KINDS:
        oly_rows = load_source("olympiadbench", kind)
        aime_rows = load_source("aime25", kind)
        source_counts[f"olympiadbench:{kind}"] = len(oly_rows)
        source_counts[f"aime25:{kind}"] = len(aime_rows)
        oly.extend(oly_rows)
        aime.extend(aime_rows)

    oly_screen, oly_quotas = stratified_screen(oly, args.oly_size)
    aime_all = sorted(aime, key=lambda row: (row["dataset"], stable_order(row)))
    selected = sorted(
        oly_screen + aime_all,
        key=lambda row: (row["dataset"], stable_order(row)),
    )
    write_jsonl(args.out_dir / "screen_hard.jsonl", selected)
    manifest = {
        "model": "r1_7b",
        "seed": 42,
        "k": 4,
        "stage": "screen_hard",
        "protocol_id": "puma-fullcot-32k-v2",
        "source_counts": source_counts,
        "olympiad_pool": len(oly),
        "olympiad_screen": len(oly_screen),
        "aime25_all": len(aime_all),
        "screen_hard": len(selected),
        "screen_strata": ["dataset", "source_kind", "host_ok"],
        "screen_rare_stratum_max": RARE_STRATUM_MAX,
        "olympiad_quotas": oly_quotas,
        "keep_configs": list(KEEP_CONFIGS),
        "drop_configs": list(DROP_CONFIGS),
        "drop_reason": (
            "MATH+GPQA 300 题上 leave-one-out 已回答；"
            "Alternatively/Hmm/SAFE 无明确候选信号；"
            "free 不用重采样。"
        ),
        "sampling_seed": 20260904,
    }
    (args.out_dir / "manifest_screen_hard.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
