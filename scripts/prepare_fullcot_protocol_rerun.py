#!/usr/bin/env python3
"""Replace all 7B/8B/14B PLWS continuations after protocol drift."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = (
    ROOT / "results" / "runs" / "plws" / "window_first" / "k_4" / "lexicon_core"
)
REPAIR_ROOT = ROOT / "results" / "runs" / "plws_protocol_repair_v1"
REPORT_JSON = ROOT / "results" / "reports" / "plws_protocol_v1_affected.json"
REPORT_MD = ROOT / "tables" / "firstwin_wait" / "protocol_repair.md"
MODELS = ("r1_7b", "nemotron_8b", "r1_14b")
LANES = (
    ("r1_7b", 0, 2),
    ("r1_7b", 1, 2),
    ("nemotron_8b", 0, 1),
    ("r1_14b", 0, 2),
    ("r1_14b", 1, 2),
)


def records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def collect_jobs() -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for path in SCORE_ROOT.glob(f"{model}/*/seed_*/jobs/*.jsonl"):
            parts = path.relative_to(SCORE_ROOT).parts
            dataset = parts[1]
            seed = int(parts[2].removeprefix("seed_"))
            kind = path.stem
            target = (
                SCORE_ROOT
                / model
                / dataset
                / f"seed_{seed}"
                / "scores"
                / kind
                / "shard_0.jsonl"
            )
            for source in records(path):
                uid = str(source["uid"])
                job = dict(source)
                job["_repair_target"] = str(target.relative_to(ROOT))
                job["_repair_kind"] = kind
                job["_repair_seed"] = seed
                job["kind"] = "repair"
                job["seed"] = 42
                selected[uid] = {
                    "uid": uid,
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "kind": kind,
                    "question_idx": source.get("question_idx"),
                    "target": str(target.relative_to(ROOT)),
                    "job": job,
                }
    return sorted(selected.values(), key=lambda row: row["uid"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("pass --apply to delete all old 7B/8B/14B PLWS scores")

    previous = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    token_affected = previous["affected"]
    all_jobs = collect_jobs()

    removed = {}
    for model in MODELS:
        count = 0
        for directory in SCORE_ROOT.glob(f"{model}/*/seed_*/scores"):
            count += sum(1 for path in directory.rglob("*") if path.is_file())
            shutil.rmtree(directory)
        removed[model] = count
    if REPAIR_ROOT.exists():
        shutil.rmtree(REPAIR_ROOT)

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_jobs:
        by_model[item["model"]].append(item)
    plan = []
    public_jobs = []
    for lane, (model, shard, num_shards) in enumerate(LANES):
        selected = [
            item
            for index, item in enumerate(by_model[model])
            if index % num_shards == shard
        ]
        jobs_path = REPAIR_ROOT / "jobs" / f"lane_{lane}.jsonl"
        output = REPAIR_ROOT / "outputs" / f"lane_{lane}.jsonl"
        write_jsonl(jobs_path, [item["job"] for item in selected])
        public_jobs.extend(
            [{key: value for key, value in item.items() if key != "job"} for item in selected]
        )
        plan.append(
            {
                "lane": lane,
                "gpu": lane,
                "model": model,
                "jobs": len(selected),
                "jobs_path": str(jobs_path),
                "output": str(output),
            }
        )
    REPAIR_ROOT.mkdir(parents=True, exist_ok=True)
    (REPAIR_ROOT / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    payload = {
        "protocol_id": "puma-fullcot-32k-v2",
        "direct_token_limit_affected": token_affected,
        "full_rerun": public_jobs,
        "full_rerun_reason": [
            "old scorer used a generic prompt instead of the PUMA default prompt",
            "old scorer forced top_k=30 instead of model generation_config",
        ],
        "removed_score_files": removed,
        "repair_plan": plan,
        "removed_qwen_and_ablation_files": previous["removed_files"],
    }
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    direct_counts = Counter(row["model"] for row in token_affected)
    full_counts = Counter(row["model"] for row in public_jobs)
    detail = Counter(
        (row["model"], row["dataset"], row["seed"]) for row in token_affected
    )
    lines = [
        "# Full-CoT 32K 协议修复",
        "",
        "唯一规范见 [`../../docs/FULLCOT_PROTOCOL.md`](../../docs/FULLCOT_PROTOCOL.md)。",
        "",
        "旧 scorer 除 1024-token 上限外还使用了错误的通用 prompt 和固定",
        "`top_k=30`。因此 7B/8B/14B 旧 PLWS 分数已全部删除并完整重跑；",
        "“直接触碰旧 token 上限”的逐题审计仍单独保留。",
        "",
        "| 模型 | 直接受 token 上限影响 | 完整重跑 |",
        "|---|---:|---:|",
        *[
            f"| {model} | {direct_counts[model]} | {full_counts[model]} |"
            for model in MODELS
        ],
        "",
        "## 直接触碰旧上限的分布",
        "",
        "| 模型 | 数据集 | seed | 题数 |",
        "|---|---|---:|---:|",
    ]
    for (model, dataset, seed), count in sorted(detail.items()):
        lines.append(f"| {model} | {dataset} | {seed} | {count} |")
    lines.extend(
        [
            "",
            "逐题 UID、原因、原分片及完整重跑清单见",
            f"`{REPORT_JSON.relative_to(ROOT)}`。",
            "Qwen3 旧 PLWS 分数和旧词表消融输出已删除；本轮暂不重跑 Qwen3。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "token_affected": direct_counts,
                "full_rerun": full_counts,
                "removed_score_files": removed,
                "lanes": plan,
            },
            ensure_ascii=False,
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
