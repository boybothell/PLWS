#!/usr/bin/env python3
"""Report rental-platform gaps for the four large-model comparison lanes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plws.artifacts import load_jsonl  # noqa: E402
from plws.contest import (  # noqa: E402
    NO_NEW_WORK_DATASETS,
    OFFICIAL_FIRST_SEEDS,
    OFFICIAL_LATER_SEEDS,
    OFFICIAL_NEW_DATASETS,
    fill_plws_complete,
    sample_answers_path,
    sample_matches_protocol,
)
from plws.matrix import (  # noqa: E402
    dense_complete,
    deer_complete,
    job_rows,
    jobs_present,
    plws_score_read_dirs,
    puma_complete,
)
from plws.paths import PLWSPaths  # noqa: E402
from plws.protocol import (  # noqa: E402
    FULLCOT_GENERATION_TOKENS,
    MAX_MODEL_LEN,
    PROMPT_RESERVE_TOKENS,
    PROTOCOL_ID,
    TRUNCATED_ANSWER_FIX_TOKENS,
)

MODELS = (
    ("qwen3_30b_a3b", "Qwen3-30B-A3B"),
    ("r1_32b", "R1-Distill-Qwen-32B"),
    ("qwen3_32b", "Qwen3-32B"),
    ("qwq_32b", "QwQ-32B"),
)
DATASETS = (
    ("math-500", "MATH-500", 500),
    ("olympiadbench", "OlympiadBench", 675),
    ("gpqa-diamond", "GPQA-Diamond", 198),
    ("aime24", "AIME24", 30),
    ("aime25", "AIME25", 30),
    ("aime26", "AIME26", 30),
    ("brumo25", "BRUMO25", 30),
    ("hmmt25", "HMMT25", 30),
    ("amc23", "AMC23", 40),
)
SEEDS = (42, 0, 1, 123, 7)
CURRENT_DATASETS = tuple(
    item for item in DATASETS if item[0] in OFFICIAL_NEW_DATASETS
)
CURRENT_SEEDS = OFFICIAL_FIRST_SEEDS
CURRENT_CELL_COUNT = len(MODELS) * len(CURRENT_DATASETS) * len(CURRENT_SEEDS)
HISTORICAL_CELL_COUNT = len(MODELS) * len(DATASETS) * len(SEEDS)


def host_state(
    paths: PLWSPaths, model: str, dataset: str, seed: int, expected: int
) -> tuple[str, int]:
    sample = sample_answers_path(paths, model, dataset, seed)
    if not sample.is_file():
        return "missing", 0
    try:
        count = len(json.loads(sample.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError):
        return "invalid", 0
    if count != expected:
        return f"partial:{count}/{expected}", count
    if not sample_matches_protocol(paths, model, dataset, seed):
        return "legacy-meta", count
    return "ready", count


def plws_progress(
    paths: PLWSPaths, model: str, dataset: str, seed: int
) -> tuple[int, int]:
    if not jobs_present(paths, model, dataset, seed):
        return 0, 0
    jobs = job_rows(paths, model, dataset, seed)
    wanted = {str(row["uid"]) for row in jobs if row.get("uid") is not None}
    done: set[str] = set()
    for directory in plws_score_read_dirs(paths, model, dataset, seed):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.jsonl"):
            try:
                rows = load_jsonl(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            for row in rows:
                if row.get("uid") is not None and row.get("status") in {
                    "ok",
                    "too_long",
                    None,
                }:
                    done.add(str(row["uid"]))
    return len(wanted & done), len(wanted)


def collect(paths: PLWSPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_name in MODELS:
        for dataset, dataset_name, expected in DATASETS:
            for seed in SEEDS:
                host, host_count = host_state(
                    paths, model, dataset, seed, expected
                )
                plws, plws_reason = fill_plws_complete(
                    paths, model, dataset, seed
                )
                scored, jobs = plws_progress(paths, model, dataset, seed)
                deer, deer_reason = deer_complete(
                    paths, model, dataset, seed, expected=expected
                )
                rows.append(
                    {
                        "model": model,
                        "model_name": model_name,
                        "dataset": dataset,
                        "dataset_name": dataset_name,
                        "seed": seed,
                        "expected": expected,
                        "host": host,
                        "host_count": host_count,
                        "puma": puma_complete(paths, model, dataset, seed),
                        "dense": dense_complete(paths, model, dataset, seed),
                        "plws": plws,
                        "plws_reason": plws_reason,
                        "plws_scored": scored,
                        "plws_jobs": jobs,
                        "deer": deer,
                        "deer_reason": deer_reason,
                    }
                )
    return rows


def seed_text(rows: list[dict[str, Any]], method: str) -> str:
    values: list[str] = []
    for row in rows:
        if row[method]:
            continue
        value = f"s{row['seed']}"
        if method == "puma" and row["host"] != "ready":
            value += "（还缺 Full-CoT）"
        elif method == "plws":
            if row["plws_jobs"]:
                value += (
                    f"（已写 {row['plws_scored']}/{row['plws_jobs']}）"
                )
            else:
                value += "（还没有 jobs）"
        values.append(value)
    return "、".join(values)


def in_current_scope(row: dict[str, Any]) -> bool:
    return (
        row["dataset"] in OFFICIAL_NEW_DATASETS
        and int(row["seed"]) in CURRENT_SEEDS
    )


def method_lines(model_rows: list[dict[str, Any]], method: str) -> list[str]:
    scoped = [row for row in model_rows if in_current_scope(row)]
    per_model = len(CURRENT_DATASETS) * len(CURRENT_SEEDS)
    if method == "deer" and scoped and all(not row["deer"] for row in scoped):
        return [
            f"- 现行范围全部缺：5 个数据集 × 3 seeds，共 {per_model} 格；"
            "seeds 为 42、0、1。"
        ]
    lines: list[str] = []
    for dataset, dataset_name, _expected in CURRENT_DATASETS:
        rows = [row for row in scoped if row["dataset"] == dataset]
        missing = seed_text(rows, method)
        if missing:
            lines.append(f"- {dataset_name}：{missing}")
    return lines or ["- 无缺格。"]


def render(rows: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# 四个大模型租卡缺格清单",
        "",
        f"机器产物快照：{generated_at}",
        "",
        "## 范围与口径",
        "",
        "- 模型：Qwen3-30B-A3B、R1-Distill-Qwen-32B、"
        "Qwen3-32B、QwQ-32B。",
        "- 现行新跑：MATH-500、OlympiadBench、GPQA-Diamond、"
        "AIME25、HMMT25；第一波 seed 42、0、1。"
        f"每种方法总计 4 × 5 × 3 = {CURRENT_CELL_COUNT} 格。",
        "- 第一波齐了再排 seed 123、7。不再新开 AIME24、AIME26、"
        "BRUMO25、AMC23；已有产物保留，不删、不重跑、不入新队列。",
        "- 传输包仍按历史 4 × 9 × 5 = "
        f"{HISTORICAL_CELL_COUNT} 格核验，避免漏解已有前缀。",
        f"- 公平宿主：`{PROTOCOL_ID}`；Full-CoT 主预算 "
        f"{FULLCOT_GENERATION_TOKENS}，answer-fix "
        f"{TRUNCATED_ANSWER_FIX_TOKENS}，prompt 预留 "
        f"{PROMPT_RESERVE_TOKENS}，`max_model_len={MAX_MODEL_LEN}`。",
        "- PUMA、PLWS、DEER 分别检查规范产物；归档、旧协议和半截 "
        "`answers.json` 不算完成。主表分数线先不动。",
        "",
        "## 缺格总览（现行 5 集 × 3 seed）",
        "",
    ]
    per_model = len(CURRENT_DATASETS) * len(CURRENT_SEEDS)
    for model, model_name in MODELS:
        model_rows = [
            row for row in rows if row["model"] == model and in_current_scope(row)
        ]
        host_missing = sum(row["host"] != "ready" for row in model_rows)
        puma_missing = sum(not row["puma"] for row in model_rows)
        dense_missing = sum(not row["dense"] for row in model_rows)
        plws_missing = sum(not row["plws"] for row in model_rows)
        deer_missing = sum(not row["deer"] for row in model_rows)
        lines.append(
            f"- {model_name}：Full-CoT 宿主未齐 {host_missing}/{per_model}；"
            f"PUMA 缺 {puma_missing}/{per_model}；dense 缺 {dense_missing}/{per_model}；"
            f"PLWS 缺 {plws_missing}/{per_model}；"
            f"DEER 缺 {deer_missing}/{per_model}。"
        )

    for model, model_name in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        lines.extend(["", f"## {model_name}", "", "### PUMA"])
        lines.extend(method_lines(model_rows, "puma"))
        lines.extend(["", "### PLWS"])
        lines.extend(method_lines(model_rows, "plws"))
        lines.extend(["", "### DEER"])
        lines.extend(method_lines(model_rows, "deer"))

    lines.extend(
        [
            "",
            "## 租卡执行方案",
            "",
            "1. 先把本机 `samples/`、PUMA、dense、jobs、PLWS shard 和 "
            "DEER 规范目录同步到租卡机；以本清单对应 JSON 重扫，禁止按旧 "
            "`status.json` 直接重跑。",
            "2. A800-80GB 先对每个 checkpoint 做单题 TP=1 预检：真实导入、"
            "`max_model_len=37888`、32K 生成和写盘都通过后才放全量。若某模型"
            "单卡因 KV 余量不足，只把该模型回退 TP=2，不整队统一 TP=2。",
            "3. 第一波只领现行五集的 seed 42、0、1。已齐格跳过；"
            "可续 PLWS 优先 R1-32B Olympiad s0 496/560、MATH s1 40/272，"
            "再收 Qwen3-30B 已有 jobs 且落在现行范围内的格子。"
            "这些格子不需要重做 Full-CoT。",
            "4. 第二批补缺 Full-CoT/PUMA，并立即产出 dense、第一扇 k=4 "
            "窗口 jobs；同一格 jobs 一齐就进入 PLWS 动态池。",
            "5. DEER 是独立方法池，只排现行 60 格。保留各模型族"
            "自己的 think_ratio、置信聚合和退出机制；probe 成本单独记录。",
            "6. 空闲 GPU 动态领下一格；所有模型冷加载整机串行。完整性看"
            "规范产物与 manifest，不看 wrapper 退出码。",
            "7. seed 123/7 等第一波齐了再开。AIME24 / AIME26 / BRUMO25 / "
            "AMC23 不入队。",
            "",
            "## 本机分工",
            "",
            "- 本机不再启动上述四个模型的 TP=2 任务；已有 R1-32B "
            "Olympiad s0 保留 496/560，MATH s1 保留 40/272，转租卡续。",
            "- 已在飞的 14B Olympiad Full-CoT s7 等落盘，不再新开 seed 7。",
            "- 本机新开只按现行五集、先三个 seed；AIME24 / AIME26 / "
            "BRUMO25 / AMC23 不再补。",
            "- 不把租卡上的四个大模型 DEER 混回本机。",
            "",
            "机器可读清单："
            "`manifests/large_model_rental_inventory.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tables" / "firstwin_wait" / "large_model_rental_missing.md",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "manifests" / "large_model_rental_inventory.json",
    )
    parser.add_argument(
        "--expect-transfer",
        type=Path,
        help="Verify that an extracted transfer contains every recorded reusable artifact.",
    )
    args = parser.parse_args()
    paths = PLWSPaths(ROOT)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = collect(paths)
    payload = {
        "generated_at": generated_at,
        "protocol_id": PROTOCOL_ID,
        "models": [model for model, _name in MODELS],
        "datasets": [dataset for dataset, _name, _expected in DATASETS],
        "seeds": list(SEEDS),
        "current_datasets": list(OFFICIAL_NEW_DATASETS),
        "current_seeds": list(CURRENT_SEEDS),
        "later_seeds": list(OFFICIAL_LATER_SEEDS),
        "no_new_work_datasets": list(NO_NEW_WORK_DATASETS),
        "rows": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows, generated_at), encoding="utf-8")
    print(args.output)
    print(args.json_output)
    if args.expect_transfer is not None:
        expected = json.loads(args.expect_transfer.read_text(encoding="utf-8"))
        indexed = {
            (row["model"], row["dataset"], int(row["seed"])): row for row in rows
        }
        errors: list[str] = []
        checks = {
            "host_ready": lambda row: row["host"] == "ready",
            "puma_complete": lambda row: bool(row["puma"]),
            "dense_complete": lambda row: bool(row["dense"]),
            "plws_complete": lambda row: bool(row["plws"]),
        }
        for name, check in checks.items():
            for cell in expected["required"][name]:
                key = (cell["model"], cell["dataset"], int(cell["seed"]))
                row = indexed.get(key)
                if row is None or not check(row):
                    errors.append(f"{name} missing: {key}")
        for cell in expected["required"]["plws_partial"]:
            key = (cell["model"], cell["dataset"], int(cell["seed"]))
            row = indexed.get(key)
            if row is None:
                errors.append(f"plws_partial missing cell: {key}")
                continue
            if row["plws_jobs"] != int(cell["jobs"]):
                errors.append(
                    f"plws_partial jobs changed: {key} "
                    f"{row['plws_jobs']} != {cell['jobs']}"
                )
            if row["plws_scored"] < int(cell["min_scored"]):
                errors.append(
                    f"plws_partial progress lost: {key} "
                    f"{row['plws_scored']} < {cell['min_scored']}"
                )
        if errors:
            print("\n".join(f"[transfer-error] {error}" for error in errors), file=sys.stderr)
            return 2
        print(f"[ok] transfer baseline verified: {args.expect_transfer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
