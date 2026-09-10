#!/usr/bin/env python3
"""剩窗 wait-help：标签分布，以及打完分之后的通用标量 AUROC。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg

OUT = AE / "tables/leftover_waithelp.md"
SCORE = AE / "results/leftover_waithelp"
SIGS = (
    ("confidence", "把握"),
    ("stop_margin", "停边距"),
    ("next_entropy", "下一步熵"),
    ("last_mean_logp", "当前答 logp"),
    ("alt_margin", "对其他试答差"),
    ("run_len", "同答已连多长"),
    ("n_distinct", "换过几答"),
    ("decision_step", "绝对步数"),
)
HID = (
    ("last_norm", "boxed 末范数"),
    ("pre_norm", "boxed 前范数"),
    ("cos_pre_last", "前后夹角"),
    ("delta_norm", "前后位移"),
)
# AUROC 越大越像「再等会更好」（wait_helps=1），必须等。


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell_name(row: dict[str, Any]) -> str:
    ds = {
        "math-500": "MATH",
        "olympiadbench": "奥赛",
        "gpqa-diamond": "GPQA",
        "aime24": "AIME24",
        "aime25": "AIME25",
    }[row["dataset"]]
    model = {
        "r1_7b": "7B",
        "nemotron_8b": "8B",
        "r1_14b": "14B",
        "qwen3_4b": "Qwen3-4B",
        "qwen3_8b": "Qwen3-8B",
    }[row["model"]]
    return f"{model} {ds}"


def fmt_auroc(v: float) -> str:
    return "—" if v != v else f"{v:.2f}"


def attach_hidden(scores: list[dict[str, Any]]) -> None:
    by_shard: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        src = row.get("_src")
        if src:
            by_shard[Path(src)].append(row)
    for path, rows in by_shard.items():
        last_p = path.with_name(f"hidden_last_{path.stem}.npy")
        pre_p = path.with_name(f"hidden_pre_{path.stem}.npy")
        if not last_p.is_file() or not pre_p.is_file():
            continue
        last = np.load(last_p)
        pre = np.load(pre_p)
        for row in rows:
            i = int(row.get("hidden_idx", -1))
            if i < 0 or i >= len(last) or i >= len(pre):
                continue
            a = last[i].astype(np.float32)
            b = pre[i].astype(np.float32)
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            row["last_norm"] = na
            row["pre_norm"] = nb
            row["cos_pre_last"] = float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")
            row["delta_norm"] = float(np.linalg.norm(a - b))


def main() -> None:
    cands: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for path in sorted(SCORE.glob("*.jsonl")):
        if "_shard" in path.name:
            for row in load_jsonl(path):
                row["_src"] = str(path)
                scores.append(row)
        elif path.stem in {"r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b", "qwen3_8b"}:
            cands.extend(load_jsonl(path))
    attach_hidden(scores)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cands:
        by[cell_name(row)].append(row)
    scored_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        if row.get("status") != "ok":
            continue
        scored_by[cell_name(row)].append(row)
    lines = [
        "# 剩窗：再等会不会更好",
        "",
        "只看第一扇非高把握同答窗。正类 `wait_helps` = 交这扇会错，而 k4 无后路再等是对的。",
        "这是误杀类。AUROC 越大越像必须等。全部免训：没有线性探针。",
        "hidden 只用范数 / 夹角 / 位移，当场能算。",
        "",
        "## 1. 标签",
        "",
        "| 集 | 剩窗 | 再等会更好 | 交了不伤 Acc | 后面有高把握 | 其中同答 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(by):
        xs = by[name]
        n = len(xs)
        helps = sum(1 for x in xs if x["wait_helps"])
        high = sum(1 for x in xs if x["high_later"])
        same = sum(1 for x in xs if x["same_as_high"])
        lines.append(
            f"| {name} | {n} | {helps}（{100.0 * helps / n:.0f}%） "
            f"| {n - helps} | {high} | {same} |"
        )
    lines += [
        "",
        "## 2. 通用标量 AUROC（正类 = 再等会更好）",
        "",
        "| 集 | 已打分 必须等/可停 | 把握 | 停边距 | 下一步熵 | 当前答 logp | 对其他试答差 | 同答已连 | 换过几答 | 绝对步数 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(by):
        xs = scored_by.get(name) or []
        if not xs:
            lines.append(f"| {name} | — | — | — | — | — | — | — | — | — |")
            continue
        pos_n = sum(1 for x in xs if x["wait_helps"])
        neg_n = len(xs) - pos_n
        cells = [fmt_auroc(rg.auroc(
            [float(x[f]) for x in xs if x["wait_helps"]],
            [float(x[f]) for x in xs if not x["wait_helps"]],
        )) for f, _ in SIGS]
        lines.append(f"| {name} | {pos_n}/{neg_n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 3. hidden 免训几何（正类 = 再等会更好）",
        "",
        "boxed 前 = 思路最后一 token；boxed 末 = 试答最后一个 token。不拟合任何权重。",
        "",
        "| 集 | 已打分 必须等/可停 | boxed 末范数 | boxed 前范数 | 前后夹角 | 前后位移 |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(by):
        xs = [x for x in scored_by.get(name, []) if x.get("last_norm") == x.get("last_norm")]
        if not xs:
            lines.append(f"| {name} | — | — | — | — | — |")
            continue
        pos_n = sum(1 for x in xs if x["wait_helps"])
        neg_n = len(xs) - pos_n
        cells = [fmt_auroc(rg.auroc(
            [float(x[f]) for x in xs if x["wait_helps"]],
            [float(x[f]) for x in xs if not x["wait_helps"]],
        )) for f, _ in HID]
        lines.append(f"| {name} | {pos_n}/{neg_n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "读法：免训第三条信号要在各集都明显高于 0.50，才能谈同一条门槛。贴 0.5 就停。",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
