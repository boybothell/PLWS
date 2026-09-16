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
    fill_plws_complete,
    sample_answers_path,
    sample_matches_protocol,
)
from plws.matrix import (  # noqa: E402
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


def method_lines(model_rows: list[dict[str, Any]], method: str) -> list[str]:
    if method == "deer" and all(not row["deer"] for row in model_rows):
        return [
            "- 全部缺：9 个数据集 × 5 seeds，共 45 格；"
            "seeds 为 42、0、1、123、7。"
        ]
    lines: list[str] = []
    for dataset, dataset_name, _expected in DATASETS:
        rows = [row for row in model_rows if row["dataset"] == dataset]
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
        "- 数据集：MATH-500、OlympiadBench、GPQA-Diamond、"
        "AIME24、AIME25、AIME26、BRUMO25、HMMT25、AMC23。",
        "- seeds：42、0、1、123、7。每种方法总计 4 × 9 × 5 = 180 格。",
        f"- 公平宿主：`{PROTOCOL_ID}`；Full-CoT 主预算 "
        f"{FULLCOT_GENERATION_TOKENS}，answer-fix "
        f"{TRUNCATED_ANSWER_FIX_TOKENS}，prompt 预留 "
        f"{PROMPT_RESERVE_TOKENS}，`max_model_len={MAX_MODEL_LEN}`。",
        "- PUMA、PLWS、DEER 分别检查规范产物；归档、旧协议和半截 "
        "`answers.json` 不算完成。",
        "",
        "## 缺格总览",
        "",
    ]
    for model, model_name in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        host_missing = sum(row["host"] != "ready" for row in model_rows)
        puma_missing = sum(not row["puma"] for row in model_rows)
        plws_missing = sum(not row["plws"] for row in model_rows)
        deer_missing = sum(not row["deer"] for row in model_rows)
        lines.append(
            f"- {model_name}：Full-CoT 宿主未齐 {host_missing}/45；"
            f"PUMA 缺 {puma_missing}/45；PLWS 缺 {plws_missing}/45；"
            f"DEER 缺 {deer_missing}/45。"
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
            "3. 第一批先收可续的 PLWS：R1-32B Olympiad s0 "
            "496/560、MATH s1 40/272；再收 Qwen3-30B 已有 jobs 的 7 格。"
            "这些格子不需要重做 Full-CoT。",
            "4. 第二批补缺 Full-CoT/PUMA，并立即产出 dense、第一扇 k=4 "
            "窗口 jobs；同一格 jobs 一齐就进入 PLWS 动态池。",
            "5. DEER 是独立方法池，四个模型当前 180 格全缺。保留各模型族"
            "自己的 think_ratio、置信聚合和退出机制；probe 成本单独记录。",
            "6. 空闲 GPU 动态领下一格；所有模型冷加载整机串行。完整性看"
            "规范产物与 manifest，不看 wrapper 退出码。",
            "",
            "## 本机分工",
            "",
            "- 本机不再启动上述四个模型的 TP=2 任务；已有 R1-32B "
            "Olympiad s0 保留 496/560，MATH s1 保留 40/272。",
            "- 当前只用空闲的 GPU 3、4 跑单卡模型；GPU 5、6 已被其他用户"
            "占用，不抢占。",
            "- 第一优先：14B 主三集 seed 7。先并行续 GPQA 120/178 和 "
            "MATH 0/414，再补 Olympiad 的 Full-CoT/PUMA/PLWS。",
            "- 第二优先：1.5B 主三集五 seeds；第三优先：Nemotron、4B、8B "
            "主三集 seed 7；第四优先：Llama-8B 主三集与 AIME26/AMC23 缺格。",
            "- 上述 PUMA/PLWS 收齐后，本机单卡继续补小模型 DEER；不把"
            "租卡上的四个大模型 DEER 混回本机。",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
