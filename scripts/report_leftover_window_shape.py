#!/usr/bin/env python3
"""第一扇剩窗：4 步把握形状。标签 leftover_ok / wait_helps / will_change。"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_full_no_fs as nofs
import report_k4_hyps as hy
import report_k4_second_lock as sl
from export_leftover_waithelp import MODELS

TABLE = AE / "tables/leftover_window_shape.md"
MODEL_ZH = {
    "r1_7b": "7B",
    "nemotron_8b": "8B",
    "r1_14b": "14B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}
DS_ZH = {
    "math-500": "MATH",
    "olympiadbench": "奥赛",
    "gpqa-diamond": "GPQA",
    "aime24": "AIME24",
    "aime25": "AIME25",
}
FEATS = (
    ("last", "最后一步"),
    ("first", "第一步"),
    ("mean", "4 步平均"),
    ("min", "4 步最差"),
    ("delta", "last−first"),
    ("range", "max−min"),
    ("ndip", "回落次数"),
    ("std", "4 步标准差"),
)


def cell_name(row: dict[str, Any]) -> str:
    return f"{MODEL_ZH[row['model']]} {DS_ZH[row['dataset']]}"


def shape(confs: list[float]) -> dict[str, float]:
    last_gt_first = 1.0 if confs[-1] > confs[0] else 0.0
    no_drop = 1.0 if all(confs[i + 1] >= confs[i] for i in range(3)) else 0.0
    dips = [1.0 for i in range(3) if confs[i + 1] < confs[i]]
    mean = sum(confs) / 4.0
    var = sum((c - mean) ** 2 for c in confs) / 4.0
    return {
        "last": confs[-1],
        "first": confs[0],
        "mean": mean,
        "min": min(confs),
        "delta": confs[-1] - confs[0],
        "range": max(confs) - min(confs),
        "ndip": float(len(dips)),
        "std": math.sqrt(var),
        "last_gt_first": last_gt_first,
        "no_drop": no_drop,
        "has_dip": 1.0 if dips else 0.0,
    }


def mean4(xs: list[list[float]]) -> str:
    if not xs:
        return "—"
    m = [sum(col) / len(xs) for col in zip(*xs)]
    return " / ".join(f"{v:.3f}" for v in m)


def pct(xs: list[float]) -> str:
    if not xs:
        return "—"
    return f"{100.0 * sum(xs) / len(xs):.0f}%"


def fmt_auroc(v: float) -> str:
    return "—" if v != v else f"{v:.2f}"


def collect_cell(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not trials_path.is_file():
        return []
    puma_path = dd.puma_stat_path(model, dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)} if regen_path.is_file() else {}
    )
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    if official:
        qis = sorted(official)
    elif gmap:
        qis = sorted(set(gmap) & set(by))
    else:
        qis = sorted(by)
    out = []
    for qi in qis:
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        info = official.get(qi) or {}
        g = gmap.get(qi) or {}
        last = max(trials, key=lambda x: int(x["stopped_len"]))
        gt = info.get("ground_truth") or g.get("ground_truth")
        original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
        orig_ok = bool(info.get("original_correct")) if info else bool(
            low.credit(original, gt, original, True)
        )
        orig_tok = int(info.get("original_tokens") or last.get("count_reasoning_tokens") or 0)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        host = regen.get(qi)
        if sim["branch"] == "consec":
            nofs_ok = bool(host.get("compressed_correct")) if host else bool(
                low.credit(sim["answer"], gt, original, orig_ok)
            )
        else:
            nofs_ok = orig_ok
        wins = sl.same_windows(rows)
        left = next((w for w in wins if hy.leftover(w)), None)
        if left is None:
            continue
        high = next((w for w in wins if w["kind"] == "high" and w["step"] > left["step"]), None)
        window = rows[left["end"] + 1 - rg.K : left["end"] + 1]
        confs = rg.confs_of(window)
        if len(confs) != 4 or any(c != c for c in confs):
            continue
        leftover_ok = bool(low.credit(left["ans"], gt, original, orig_ok))
        feat = shape(confs)
        feat.update(
            {
                "model": model,
                "dataset": dataset,
                "kind": left["kind"],
                "confs": confs,
                "leftover_ok": leftover_ok,
                "wait_helps": bool(nofs_ok and not leftover_ok),
                "will_change": bool(
                    high is not None and not rg.same(left["ans"], high["ans"])
                ),
            }
        )
        out.append(feat)
    return out


def auroc_of(xs: list[dict[str, Any]], y: str, feat: str) -> float:
    pos = [x[feat] for x in xs if x[y]]
    neg = [x[feat] for x in xs if not x[y]]
    return rg.auroc(pos, neg)


def shape_row(name: str, pos: list[dict[str, Any]], neg: list[dict[str, Any]]) -> str:
    return (
        f"| {name} | {len(pos)}/{len(neg)} | "
        f"{mean4([x['confs'] for x in pos])} | {mean4([x['confs'] for x in neg])} | "
        f"{pct([x['last_gt_first'] for x in pos])} | {pct([x['last_gt_first'] for x in neg])} | "
        f"{pct([x['no_drop'] for x in pos])} | {pct([x['no_drop'] for x in neg])} | "
        f"{pct([x['has_dip'] for x in pos])} | {pct([x['has_dip'] for x in neg])} |"
    )


def auroc_row(name: str, xs: list[dict[str, Any]], y: str) -> str:
    npos = sum(1 for x in xs if x[y])
    nneg = len(xs) - npos
    if npos == 0 or nneg == 0:
        return f"| {name} | {npos}/{nneg} | " + " | ".join("—" for _ in FEATS) + " |"
    cells = [fmt_auroc(auroc_of(xs, y, f)) for f, _ in FEATS]
    return f"| {name} | {npos}/{nneg} | " + " | ".join(cells) + " |"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for _, dataset in nofs.DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            n = 0
            for seed in seeds:
                part = collect_cell(model, dataset, seed)
                rows.extend(part)
                n += len(part)
            if n:
                helps = sum(
                    1
                    for x in rows
                    if x["model"] == model and x["dataset"] == dataset and x["wait_helps"]
                )
                print(f"{model} {dataset} leftover={n} wait_helps={helps}", flush=True)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[cell_name(row)].append(row)
    names = sorted(by)
    low_rows = [x for x in rows if x["kind"] == "low"]
    mix_rows = [x for x in rows if x["kind"] == "mix"]
    lines = [
        "# 剩窗 4 步把握形状",
        "",
        "第一扇非高把握同答窗（低 + 混合），现成 4 步把握。零 GPU。",
        "`leftover_ok` = 这扇试答对金标或写完终答。`wait_helps` = 交这扇会错、k4 无后路再等是对的。",
        "`will_change` = 后面还有高把握锁，且答案不同。",
        "",
        "## 1. 4 步均值 + 升降（正 / 负）",
        "",
        "### leftover_ok",
        "",
        "| 集 | 对/错 | 对：4 步均值 | 错：4 步均值 | 对 last>first | 错 last>first | 对不降 | 错不降 | 对有回落 | 错有回落 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        xs = by[name]
        lines.append(shape_row(name, [x for x in xs if x["leftover_ok"]], [x for x in xs if not x["leftover_ok"]]))
    lines.append(shape_row("全体", [x for x in rows if x["leftover_ok"]], [x for x in rows if not x["leftover_ok"]]))
    lines.append(shape_row("全体低把握", [x for x in low_rows if x["leftover_ok"]], [x for x in low_rows if not x["leftover_ok"]]))
    lines.append(shape_row("全体混合", [x for x in mix_rows if x["leftover_ok"]], [x for x in mix_rows if not x["leftover_ok"]]))
    lines += [
        "",
        "### wait_helps（再等会更好）",
        "",
        "| 集 | 必须等/其余 | 必须等：4 步均值 | 其余：4 步均值 | 必须等 last>first | 其余 last>first | 必须等不降 | 其余不降 | 必须等有回落 | 其余有回落 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        xs = by[name]
        lines.append(shape_row(name, [x for x in xs if x["wait_helps"]], [x for x in xs if not x["wait_helps"]]))
    lines.append(shape_row("全体", [x for x in rows if x["wait_helps"]], [x for x in rows if not x["wait_helps"]]))
    lines.append(shape_row("全体低把握", [x for x in low_rows if x["wait_helps"]], [x for x in low_rows if not x["wait_helps"]]))
    lines.append(shape_row("全体混合", [x for x in mix_rows if x["wait_helps"]], [x for x in mix_rows if not x["wait_helps"]]))
    lines += [
        "",
        "### will_change（后面高把握换答）",
        "",
        "| 集 | 会换/其余 | 会换：4 步均值 | 其余：4 步均值 | 会换 last>first | 其余 last>first | 会换不降 | 其余不降 | 会换有回落 | 其余有回落 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        xs = by[name]
        lines.append(shape_row(name, [x for x in xs if x["will_change"]], [x for x in xs if not x["will_change"]]))
    lines.append(shape_row("全体", [x for x in rows if x["will_change"]], [x for x in rows if not x["will_change"]]))
    lines.append(shape_row("全体低把握", [x for x in low_rows if x["will_change"]], [x for x in low_rows if not x["will_change"]]))
    lines += [
        "",
        "## 2. 形状 AUROC",
        "",
        "越大越像正类。last−first / 回落次数 / 标准差才是「窗内变化」；最后一步 / 平均是旧把握。",
        "",
        "### leftover_ok",
        "",
        "| 集 | 对/错 | 最后一步 | 第一步 | 4 步平均 | 4 步最差 | last−first | max−min | 回落次数 | 4 步标准差 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        lines.append(auroc_row(name, by[name], "leftover_ok"))
    lines.append(auroc_row("全体", rows, "leftover_ok"))
    lines.append(auroc_row("全体低把握", low_rows, "leftover_ok"))
    lines.append(auroc_row("全体混合", mix_rows, "leftover_ok"))
    lines += [
        "",
        "### wait_helps",
        "",
        "| 集 | 必须等/其余 | 最后一步 | 第一步 | 4 步平均 | 4 步最差 | last−first | max−min | 回落次数 | 4 步标准差 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        lines.append(auroc_row(name, by[name], "wait_helps"))
    lines.append(auroc_row("全体", rows, "wait_helps"))
    lines.append(auroc_row("全体低把握", low_rows, "wait_helps"))
    lines.append(auroc_row("全体混合", mix_rows, "wait_helps"))
    lines += [
        "",
        "### will_change",
        "",
        "| 集 | 会换/其余 | 最后一步 | 第一步 | 4 步平均 | 4 步最差 | last−first | max−min | 回落次数 | 4 步标准差 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        lines.append(auroc_row(name, by[name], "will_change"))
    lines.append(auroc_row("全体", rows, "will_change"))
    lines.append(auroc_row("全体低把握", low_rows, "will_change"))
    lines += ["", "读法：变化量（last−first / 回落 / 标准差）若贴 0.5，窗内走法分不开该不该停。", ""]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
