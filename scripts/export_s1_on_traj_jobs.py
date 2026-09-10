#!/usr/bin/env python3
"""导出「已有轨迹 + 2× Wait」jobs：官方 Full-CoT / 窗后压思考。不改官方 generated_text。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(AE / "scripts"))
import replay_default_dense_gate as dd  # noqa: E402

DATASETS = ("math-500", "olympiadbench", "gpqa-diamond", "aime24", "aime25")
SEEDS = (42, 0, 1, 123)


def thinking_of(text: str) -> str:
    text = str(text or "")
    if "</think>" in text:
        return text.split("</think>", 1)[0]
    return text


def sample_path(tag: str, ds: str, seed: int) -> Path:
    return AE / f"samples/{tag}/{ds}/seed_{seed}/answers.json"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def official_jobs(tag: str, seed: int, datasets: tuple[str, ...]) -> list[dict]:
    jobs: list[dict] = []
    for ds in datasets:
        samples = json.loads(sample_path(tag, ds, seed).read_text())
        official = {int(r["question_idx"]): r for r in dd.load_json(dd.puma_stat_path(tag, ds, seed))}
        for i, sample in enumerate(samples):
            qi = i + 1
            info = official.get(qi) or {}
            sq = " ".join(str(sample.get("question") or "").split())
            oq = " ".join(str(info.get("question") or "").split())
            if oq and sq and oq != sq:
                continue
            thought = thinking_of(sample.get("reasoning") or "") or thinking_of(sample.get("generated_text") or "")
            question = str(sample.get("question") or info.get("question") or "")
            if not thought or not question:
                continue
            jobs.append(
                {
                    "uid": f"{tag}:{ds}:{seed}:{qi}:offwait",
                    "src": "official",
                    "model": tag,
                    "dataset": ds,
                    "seed": seed,
                    "question_idx": qi,
                    "question": question,
                    "thought": thought,
                    "gt": info.get("ground_truth") or sample.get("ground_truth_answer"),
                    "host_ok": bool(info.get("original_correct")),
                    "original_tokens": info.get("original_tokens"),
                    "old_answer": info.get("original_answer") or sample.get("model_answer"),
                }
            )
    return jobs


def window_jobs(tag: str, seed: int, datasets: tuple[str, ...]) -> list[dict]:
    official_by: dict[tuple[str, int], dict] = {}
    for ds in datasets:
        path = dd.puma_stat_path(tag, ds, seed)
        if not path.is_file():
            continue
        for row in dd.load_json(path):
            official_by[(ds, int(row["question_idx"]))] = row
    seen: set[tuple[str, int]] = set()
    jobs: list[dict] = []
    root = AE / "results/leftover_suppress_toend"
    for suf in ("", "_high", "_mix"):
        folder = root / f"{tag}_s{seed}_suppress{suf}"
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("scores_shard*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ds = rec.get("dataset")
                if ds not in datasets:
                    continue
                qi = int(rec.get("question_idx"))
                key = (ds, qi)
                if key in seen:
                    continue
                thought = thinking_of(rec.get("generated_text") or "")
                info = official_by.get(key) or {}
                question = str(info.get("question") or rec.get("question") or "")
                if not thought or not question:
                    continue
                seen.add(key)
                jobs.append(
                    {
                        "uid": f"{tag}:{ds}:{seed}:{qi}:winwait",
                        "src": "window",
                        "model": tag,
                        "dataset": ds,
                        "seed": seed,
                        "question_idx": qi,
                        "question": question,
                        "thought": thought,
                        "gt": info.get("ground_truth") or rec.get("gt"),
                        "host_ok": bool(info.get("original_correct", rec.get("host_ok"))),
                        "original_tokens": info.get("original_tokens", rec.get("original_tokens")),
                        "old_answer": info.get("original_answer") or rec.get("new_answer"),
                    }
                )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--seeds", default="42,0,1,123")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    out_root = AE / "results/s1_on_traj/jobs"
    for seed in seeds:
        off = official_jobs(args.model_tag, seed, datasets)
        win = window_jobs(args.model_tag, seed, datasets)
        off_p = out_root / f"{args.model_tag}_s{seed}_official.jsonl"
        win_p = out_root / f"{args.model_tag}_s{seed}_window.jsonl"
        write_jsonl(off_p, off)
        write_jsonl(win_p, win)
        print(
            f"s{seed} official {len(off)} {dict(Counter(j['dataset'] for j in off))} -> {off_p}",
            flush=True,
        )
        print(
            f"s{seed} window {len(win)} {dict(Counter(j['dataset'] for j in win))} -> {win_p}",
            flush=True,
        )


if __name__ == "__main__":
    main()
