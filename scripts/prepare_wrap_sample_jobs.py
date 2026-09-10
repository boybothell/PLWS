#!/usr/bin/env python3
"""Build the CORE vs CORE+But/So/Therefore sample jobs.

Select questions, then take every firstwin seed of those questions.
A question is selected if any seed flips versus Full-CoT, or if it is
in the seed-42 control panel (long keep-right, keep-wrong, short keep-right).
Seeds without a firstwin prefix are skipped.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.matrix import FIRSTWIN  # noqa: E402
from plws.paths import PLWSPaths  # noqa: E402
from report_fullcot_puma_plws import (  # noqa: E402
    DATASETS,
    KINDS,
    load_json,
    load_scores,
)

MODELS = ("qwen3_4b", "nemotron_8b")
SEEDS = (42, 0, 1, 123)
CONTROL_SEED = 42
KEEP_RIGHT_LONG = 80
KEEP_WRONG = 40
KEEP_RIGHT_SHORT = 20
JOBS_DIR = ROOT / "results" / "experiments" / "lexicon_ablation" / "jobs"
OUT_ROOT = ROOT / "results" / "experiments" / "lexicon_ablation" / "wrap_sample"
SUMMARY = OUT_ROOT / "sample_summary.json"


def stable_key(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def leftover_tok(rec: dict[str, Any]) -> float:
    if rec.get("n_cont_tok") is not None:
        return float(rec["n_cont_tok"])
    think = float(rec.get("n_think_tok") or 0)
    left = float(rec.get("n_left_tok") or 0)
    return max(0.0, think - left)


def load_jobs(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> dict[int, dict[str, Any]]:
    jobs: dict[int, dict[str, Any]] = {}
    jobs_path = paths.jobs_path(
        model, dataset, seed, FIRSTWIN, k=4, lexicon="core"
    )
    if not jobs_path.is_file():
        return jobs
    for line in jobs_path.read_text().splitlines():
        if not line.strip():
            continue
        job = json.loads(line)
        if job.get("dataset") not in (None, dataset):
            continue
        packed = dict(job)
        packed["dataset"] = dataset
        packed["model"] = model
        packed["seed"] = seed
        packed["k"] = int(job.get("k") or 4)
        jobs[int(job["question_idx"])] = packed
    return jobs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def pick_controls(
    candidates: list[dict[str, Any]],
    *,
    size: int,
    reverse: bool | None,
) -> list[dict[str, Any]]:
    if reverse is None:
        ordered = sorted(
            candidates,
            key=lambda row: stable_key(f"{row['dataset']}:{row['qid']}"),
        )
    else:
        ordered = sorted(
            candidates,
            key=lambda row: (
                leftover_tok(row["core_score"]) * (-1 if reverse else 1),
                stable_key(f"{row['dataset']}:{row['qid']}"),
            ),
        )
    return ordered[:size]


def load_pool(paths: PLWSPaths, model: str) -> dict[tuple[str, int], dict[int, dict[str, Any]]]:
    pool: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for dataset, _dszh, _n in DATASETS:
        for seed in SEEDS:
            official = {
                int(row["question_idx"]): row
                for row in load_json(paths.puma_statistics_path(model, dataset, seed))
            }
            jobs = load_jobs(paths, model, dataset, seed)
            scores: dict[tuple[str, int], dict] = {}
            for kind in KINDS:
                scores.update(
                    load_scores(
                        paths.score_dir(
                            model, dataset, seed, kind, k=4, lexicon="core"
                        )
                    )
                )
            for qid, info in official.items():
                rec = scores.get((dataset, int(qid)))
                job = jobs.get(int(qid))
                if rec is None or job is None:
                    continue
                full_ok = bool(info.get("original_correct"))
                core_ok = bool(rec.get("new_gold_ok"))
                pool[(dataset, int(qid))][int(seed)] = {
                    "job": job,
                    "core_score": rec,
                    "full_ok": full_ok,
                    "core_ok": core_ok,
                    "flipped": core_ok != full_ok,
                    "dataset": dataset,
                    "qid": int(qid),
                    "seed": int(seed),
                }
    return pool


def build_model(paths: PLWSPaths, model: str) -> dict[str, Any]:
    pool = load_pool(paths, model)
    flip_qids = {
        key for key, seeds in pool.items() if any(row["flipped"] for row in seeds.values())
    }
    control_rows = []
    for (dataset, qid), seeds in pool.items():
        if (dataset, qid) in flip_qids:
            continue
        row = seeds.get(CONTROL_SEED)
        if row is None:
            continue
        control_rows.append(row)

    keep_right = [row for row in control_rows if row["core_ok"]]
    keep_wrong = [row for row in control_rows if not row["core_ok"]]
    selected: dict[tuple[str, int], str] = {key: "flip" for key in flip_qids}
    for row in pick_controls(keep_right, size=KEEP_RIGHT_LONG, reverse=True):
        selected[(row["dataset"], row["qid"])] = "keep_right_long"
    used = set(selected)
    short_pool = [
        row for row in keep_right if (row["dataset"], row["qid"]) not in used
    ]
    for row in pick_controls(short_pool, size=KEEP_RIGHT_SHORT, reverse=False):
        selected[(row["dataset"], row["qid"])] = "keep_right_short"
    for row in pick_controls(keep_wrong, size=KEEP_WRONG, reverse=None):
        selected[(row["dataset"], row["qid"])] = "keep_wrong"

    jobs_out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    by_seed: dict[int, Counter[str]] = defaultdict(Counter)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    qid_seed_counts: list[int] = []
    missing_seeds = 0
    for key, reason in selected.items():
        seeds = pool[key]
        qid_seed_counts.append(len(seeds))
        missing_seeds += len(SEEDS) - len(seeds)
        for seed in SEEDS:
            row = seeds.get(seed)
            if row is None:
                continue
            packed = dict(row["job"])
            packed["sample_reason"] = reason
            packed["sample_anchor"] = seed == CONTROL_SEED or (
                reason == "flip" and row["flipped"]
            )
            packed["sample_stratum"] = (
                "flip"
                if row["flipped"]
                else "keep_right"
                if row["core_ok"]
                else "keep_wrong"
            )
            packed["core_gold_ok"] = row["core_ok"]
            packed["core_n_think_tok"] = row["core_score"].get("n_think_tok")
            packed["core_n_cont_tok"] = row["core_score"].get("n_cont_tok")
            packed["core_n_ans_tok"] = row["core_score"].get("n_ans_tok")
            jobs_out.append(packed)
            counts[reason] += 1
            by_seed[seed][reason] += 1
            by_dataset[key[0]][reason] += 1

    jobs_out.sort(key=lambda row: row["uid"])
    jobs_path = JOBS_DIR / f"wrap_sample_{model}.jsonl"
    write_jsonl(jobs_path, jobs_out)
    return {
        "model": model,
        "n": len(jobs_out),
        "n_questions": len(selected),
        "questions_by_reason": dict(Counter(selected.values())),
        "jobs": str(jobs_path.relative_to(ROOT)),
        "jobs_by_reason": dict(counts),
        "by_seed": {str(seed): dict(items) for seed, items in sorted(by_seed.items())},
        "by_dataset": {name: dict(items) for name, items in sorted(by_dataset.items())},
        "seeds_per_question": {
            "mean": (sum(qid_seed_counts) / len(qid_seed_counts)) if qid_seed_counts else 0,
            "missing_unwindowed_seeds": missing_seeds,
        },
    }


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = [build_model(paths, model) for model in MODELS]
    payload = {
        "script": "scripts/prepare_wrap_sample_jobs.py",
        "lexicon": "core_plus_but_so_therefore",
        "unit": "question then all firstwin seeds",
        "control_seed": CONTROL_SEED,
        "quotas": {
            "flip": "any-seed flip, then all windowed seeds",
            "keep_right_long": KEEP_RIGHT_LONG,
            "keep_wrong": KEEP_WRONG,
            "keep_right_short": KEEP_RIGHT_SHORT,
        },
        "models": models,
    }
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {SUMMARY}")
    for model in models:
        print(
            f"{model['model']}: questions={model['n_questions']} jobs={model['n']} "
            + " ".join(
                f"{name}={count}"
                for name, count in model["questions_by_reason"].items()
            )
        )


if __name__ == "__main__":
    main()
