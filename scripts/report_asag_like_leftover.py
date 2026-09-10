#!/usr/bin/env python3
"""ASAG 类似：注意力熵改标 wait-help；试答集合熵（输出侧）。免训。"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_first_lock_room as room
import report_leftover_attn as la

OUT = AE / "tables/asag_like_leftover.md"
WAIT = AE / "results/leftover_waithelp"
ATTN_SIGS = (
    ("confidence", "把握"),
    ("attn_H", "全序列注意力熵"),
    ("q_entropy", "题面 key 熵"),
    ("lookback_ratio", "boxed 题面占比"),
    ("lens_rise", "boxed 末层−一半深"),
    ("wait_q_entropy", "Wait 题面熵"),
    ("wait_stop_margin", "Wait 收口−Wait"),
    ("wait_lookback_ratio", "Wait 题面占比"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wait_map(model: str, dataset: str) -> dict[tuple[int, int], dict[str, Any]]:
    path = WAIT / f"{model}.jsonl"
    out = {}
    if not path.is_file():
        return out
    for row in load_jsonl(path):
        if row["dataset"] != dataset:
            continue
        out[(int(row["seed"]), int(row["question_idx"]))] = row
    return out


def entropy(counts: list[int]) -> float:
    n = sum(counts)
    if n <= 0:
        return float("nan")
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return h


def feats_from_rows(rows: list[dict[str, Any]], step: int, ans: Any) -> dict[str, float]:
    used = [
        r
        for r in rows
        if int(r["stopped_len"]) <= step
        and str(r.get("final_answer") or "")
    ]
    if not used:
        return {}
    recent = used[-8:]
    keys = [str(r.get("final_answer") or "") for r in used]
    rec_keys = [str(r.get("final_answer") or "") for r in recent]
    ctr = Counter(keys)
    rctr = Counter(rec_keys)
    same = sum(1 for k in keys if rg.same(k, ans))
    r_same = sum(1 for k in rec_keys if rg.same(k, ans))
    confs = [float(r["confidence"]) for r in used if r.get("confidence") == r.get("confidence")]
    cur_c = next((float(r["confidence"]) for r in reversed(used) if rg.same(r.get("final_answer"), ans)), float("nan"))
    others = [float(r["confidence"]) for r in used if not rg.same(r.get("final_answer"), ans) and r.get("confidence") == r.get("confidence")]
    return {
        "ans_entropy": entropy(list(ctr.values())),
        "ans_entropy8": entropy(list(rctr.values())),
        "ans_purity": max(ctr.values()) / len(keys),
        "ans_share": same / len(keys),
        "ans_share8": r_same / len(rec_keys),
        "conf_gap": cur_c - (max(others) if others else float("nan")),
    }


def fmt(v: float) -> str:
    return "—" if v != v else f"{v:.2f}"


def auroc_field(rows: list[dict[str, Any]], field: str) -> float:
    return rg.auroc(
        [float(r[field]) for r in rows if r.get("wait_helps") and r.get(field) == r.get(field)],
        [float(r[field]) for r in rows if (not r.get("wait_helps")) and r.get(field) == r.get(field)],
    )


def main() -> None:
    lines = [
        "# ASAG 类似：注意力熵改标 + 试答集合熵",
        "",
        "免训。正类 = wait_helps（交剩窗会错，k4 无后路再等是对的）。",
        "注意力熵来自已打的 7B MATH/GPQA 剩窗前向。试答集合熵只读密探轨迹，不新开卡。",
        "",
        "## 1. 7B 注意力 / 层对比，标签改成 wait-help",
        "",
        "| 集 | 必须等/可停 | 把握 | 全序列注意力熵 | 题面 key 熵 | boxed 题面占比 | 末层−一半深 | Wait 题面熵 | Wait 收口−Wait | Wait 题面占比 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, stem in la.CELLS:
        dataset = "math-500" if "math" in stem else "gpqa-diamond"
        wmap = wait_map("r1_7b", dataset)
        rows = []
        for row in la.load_scores(stem):
            w = wmap.get((42, int(row["question_idx"])))
            if not w:
                continue
            row = dict(row)
            row["wait_helps"] = bool(w["wait_helps"])
            rows.append(row)
        pos = sum(1 for r in rows if r["wait_helps"])
        cells = [fmt(auroc_field(rows, f)) for f, _ in ATTN_SIGS]
        lines.append(f"| {name} | {pos}/{len(rows) - pos} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 2. 试答集合熵 / 占比 / 把握差（全模型已导出剩窗）",
        "",
        "到剩窗这一步为止，轨迹里各试答的频率熵、近 8 步熵、当前答占比、当前把握−其他答最高把握。",
        "",
        "| 集 | 必须等/可停 | 全程答熵 | 近8步答熵 | 最大答占比 | 当前答占比 | 近8步当前占比 | 把握差 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    from report_leftover_waithelp import cell_name

    cands = []
    for tag in ("r1_7b", "nemotron_8b", "r1_14b", "qwen3_4b", "qwen3_8b"):
        path = WAIT / f"{tag}.jsonl"
        if path.is_file():
            cands.extend(load_jsonl(path))
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cands:
        groups[(row["model"], row["dataset"], int(row["seed"]))].append(row)
    scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, dataset, seed), jobs in groups.items():
        trials_path = room.dense_trial_path(model, dataset, seed)
        if not trials_path.is_file():
            continue
        by: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in dd.load_json(trials_path):
            by[int(rec["question_idx"])].append(rec)
        for job in jobs:
            feats = feats_from_rows(by.get(int(job["question_idx"]), []), int(job["decision_step"]), job["answer"])
            if not feats:
                continue
            row = {**job, **feats}
            scored[cell_name(row)].append(row)
        print(f"done {model} {dataset} seed={seed}", flush=True)
    fields = (
        "ans_entropy",
        "ans_entropy8",
        "ans_purity",
        "ans_share",
        "ans_share8",
        "conf_gap",
    )
    for name in sorted(scored):
        xs = scored[name]
        pos = sum(1 for r in xs if r["wait_helps"])
        cells = [fmt(auroc_field(xs, f)) for f in fields]
        lines.append(f"| {name} | {pos}/{len(xs) - pos} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "读法：ASAG 是「把握 + 注意力熵掉下来」。这里看改标后的熵、以及输出侧答集合是否收束。",
        "要通用，各集都得明显高于 0.50。",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
