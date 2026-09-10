#!/usr/bin/env python3
"""Purge invalid PLWS outputs and prepare bounded-row protocol repairs."""

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
QWEN_MODELS = ("qwen3_4b", "qwen3_8b", "qwen3_30b_a3b")
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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temp.replace(path)


def affected_reason(row: dict[str, Any]) -> str | None:
    if row.get("status") == "too_long":
        return "too_long"
    reasons = []
    if bool(row.get("hit_cap")):
        reasons.append("reasoning_cap")
    if int(row.get("n_ans_tok") or 0) >= 1024:
        reasons.append("answer_cap")
    return "+".join(reasons) or None


def job_index(model: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in SCORE_ROOT.glob(f"{model}/*/seed_*/jobs/*.jsonl"):
        for row in records(path):
            indexed[str(row["uid"])] = row
    return indexed


def remove_invalid_outputs() -> dict[str, int]:
    removed = {}
    for model in QWEN_MODELS:
        count = 0
        for directory in SCORE_ROOT.glob(f"{model}/*/seed_*/scores"):
            count += sum(1 for path in directory.rglob("*") if path.is_file())
            shutil.rmtree(directory)
        removed[model] = count
    ablation = ROOT / "results" / "experiments" / "lexicon_ablation"
    count = 0
    for stage in ("smoke", "screen", "full"):
        directory = ablation / stage
        if directory.is_dir():
            count += sum(1 for path in directory.rglob("*") if path.is_file())
            shutil.rmtree(directory)
    removed["lexicon_ablation"] = count
    return removed


def audit_and_strip() -> list[dict[str, Any]]:
    affected: dict[str, dict[str, Any]] = {}
    indices = {model: job_index(model) for model in MODELS}
    for model in MODELS:
        for path in SCORE_ROOT.glob(f"{model}/*/seed_*/scores/*/shard_*.jsonl"):
            if path.name.startswith("shard_free_"):
                continue
            kept = []
            changed = False
            for row in records(path):
                reason = affected_reason(row)
                if reason:
                    changed = True
                    uid = str(row["uid"])
                    if uid not in indices[model]:
                        raise RuntimeError(f"missing source job for {uid}")
                    parts = path.relative_to(SCORE_ROOT).parts
                    dataset = parts[1]
                    seed = int(parts[2].removeprefix("seed_"))
                    kind = parts[4]
                    affected.setdefault(
                        uid,
                        {
                            "uid": uid,
                            "model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "kind": kind,
                            "question_idx": row.get("question_idx"),
                            "reason": reason,
                            "target": str(path.relative_to(ROOT)),
                            "job": indices[model][uid],
                        },
                    )
                    continue
                if row.get("status") == "ok":
                    row["protocol_id"] = "puma-fullcot-32k-v2"
                    row["protocol_compatibility"] = (
                        "legacy_natural_termination_below_caps"
                    )
                kept.append(row)
            if changed:
                write_jsonl(path, kept)
                for meta in path.parent.glob(f"*_{path.stem}.json"):
                    meta.unlink()
            elif kept:
                write_jsonl(path, kept)
    return sorted(affected.values(), key=lambda row: row["uid"])


def prepare_repair_jobs(affected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if REPAIR_ROOT.exists():
        shutil.rmtree(REPAIR_ROOT)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in affected:
        by_model[item["model"]].append(item)

    plan = []
    for lane, (model, shard, num_shards) in enumerate(LANES):
        selected = [
            item
            for index, item in enumerate(by_model[model])
            if index % num_shards == shard
        ]
        jobs = []
        for item in selected:
            job = dict(item["job"])
            job["_repair_target"] = item["target"]
            job["_repair_kind"] = item["kind"]
            job["_repair_seed"] = item["seed"]
            job["kind"] = "repair"
            job["seed"] = 42
            jobs.append(job)
        jobs_path = REPAIR_ROOT / "jobs" / f"lane_{lane}.jsonl"
        output = REPAIR_ROOT / "outputs" / f"lane_{lane}.jsonl"
        write_jsonl(jobs_path, jobs)
        plan.append(
            {
                "lane": lane,
                "gpu": lane,
                "model": model,
                "jobs": len(jobs),
                "jobs_path": str(jobs_path),
                "output": str(output),
            }
        )
    (REPAIR_ROOT / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return plan


def write_report(
    affected: list[dict[str, Any]],
    removed: dict[str, int],
    plan: list[dict[str, Any]],
) -> None:
    public = [{key: value for key, value in row.items() if key != "job"} for row in affected]
    payload = {
        "protocol_id": "puma-fullcot-32k-v2",
        "affected": public,
        "removed_files": removed,
        "repair_plan": plan,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = Counter(
        (row["model"], row["dataset"], row["seed"]) for row in affected
    )
    model_counts = Counter(row["model"] for row in affected)
    lines = [
        "# Full-CoT 32K 协议修复",
        "",
        "唯一规范见 [`../../docs/FULLCOT_PROTOCOL.md`](../../docs/FULLCOT_PROTOCOL.md)。",
        "下表是触碰旧 1024-token 思考/终答上限或写成 `too_long`、必须重跑的题。",
        "",
        "| 模型 | 受影响 UID |",
        "|---|---:|",
        *[f"| {model} | {model_counts[model]} |" for model in MODELS],
        "",
        "| 模型 | 数据集 | seed | 题数 |",
        "|---|---|---:|---:|",
    ]
    for (model, dataset, seed), count in sorted(counts.items()):
        lines.append(f"| {model} | {dataset} | {seed} | {count} |")
    lines.extend(
        [
            "",
            f"逐题 UID、原因和目标分片见 `{REPORT_JSON.relative_to(ROOT)}`。",
            "Qwen3 旧 PLWS 分数和旧词表消融输出已删除，不在本表逐题列出。",
        ]
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Required because this migration deletes invalid generated outputs.",
    )
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("pass --apply to purge invalid protocol outputs")
    affected = audit_and_strip()
    removed = remove_invalid_outputs()
    plan = prepare_repair_jobs(affected)
    write_report(affected, removed, plan)
    print(
        json.dumps(
            {
                "affected": Counter(row["model"] for row in affected),
                "removed_files": removed,
                "repair_lanes": plan,
            },
            ensure_ascii=False,
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
