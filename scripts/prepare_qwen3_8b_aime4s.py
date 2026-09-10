#!/usr/bin/env python3
"""Qwen3-8B AIME24/25 four-seed jobs for core / wait / core+."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOW_ROOT = ROOT / "results" / "runs" / "plws" / "window_first" / "k_4" / "lexicon_core" / "qwen3_8b"
JOBS_DIR = ROOT / "results" / "experiments" / "lexicon_ablation" / "jobs"
OUT_ROOT = ROOT / "results" / "experiments" / "lexicon_ablation" / "aime4s" / "qwen3_8b"
PRE_ROOT = ROOT / "results" / "experiments" / "lexicon_ablation" / "pre" / "qwen3_8b"
SEEDS = (42, 0, 1, 123)
DATASETS = ("aime24", "aime25")
KINDS = ("low", "mix", "high")
CONFIGS = ("core", "wait", "core_plus_let_me")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def pool_for_seed(seed: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for dataset in DATASETS:
        for kind in KINDS:
            path = WINDOW_ROOT / dataset / f"seed_{seed}" / "jobs" / f"{kind}.jsonl"
            for row in load_jsonl(path):
                uid = str(row.get("uid") or "")
                if not uid or uid in seen:
                    continue
                packed = dict(row)
                packed["source_kind"] = str(row.get("kind") or kind)
                packed["kind"] = "pilot"
                packed.pop("k", None)
                rows.append(packed)
                seen.add(uid)
    rows.sort(key=lambda row: (str(row.get("dataset") or ""), int(row.get("question_idx") or 0)))
    return rows


def reuse_seed42(config: str) -> int:
    wanted = {
        row["uid"]
        for row in load_jsonl(JOBS_DIR / "aime4s_qwen3_8b_s42.jsonl")
    }
    copied: list[dict] = []
    seen: set[str] = set()
    folder = PRE_ROOT / config
    for path in sorted(folder.glob("scores*.jsonl")):
        for row in load_jsonl(path):
            uid = row.get("uid")
            if uid not in wanted or uid in seen:
                continue
            if row.get("status") != "ok":
                continue
            if row.get("protocol_id") != "puma-fullcot-32k-v2":
                continue
            copied.append(row)
            seen.add(str(uid))
    write_jsonl(OUT_ROOT / "s42" / config / "scores.jsonl", copied)
    return len(copied)


def main() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for seed in SEEDS:
        rows = pool_for_seed(seed)
        path = JOBS_DIR / f"aime4s_qwen3_8b_s{seed}.jsonl"
        write_jsonl(path, rows)
        summary[f"s{seed}"] = {
            "n": len(rows),
            "datasets": dict(Counter(row["dataset"] for row in rows)),
            "kinds": dict(Counter(row.get("source_kind") for row in rows)),
            "jobs": str(path),
        }
    reused = {config: reuse_seed42(config) for config in CONFIGS}
    print(json.dumps({"jobs": summary, "reused_s42": reused}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
