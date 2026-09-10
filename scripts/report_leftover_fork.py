#!/usr/bin/env python3
"""假平台改口前：下一步把握 / 试答熵是不是分叉点。不训、不抽新前向。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_hyps as hy
import report_k4_second_lock as sl
import report_leftover_after as after
import train_leftover_commit_probe as tp

TABLE = AE / "tables/leftover_fork.md"
LENS = AE / "results/confcal_judge/v2/dense_lens"


def med(xs: list[float]) -> float:
    finite = [x for x in xs if x == x]
    return float(np.median(finite)) if finite else float("nan")


def mean(xs: list[float]) -> float:
    finite = [x for x in xs if x == x]
    return float(np.mean(finite)) if finite else float("nan")


def lens_dirs(model: str, dataset: str, seed: int) -> list[Path]:
    out = []
    if dataset in ("aime24", "aime25"):
        out.append(LENS / model / f"{dataset}_s{seed}")
    else:
        out.append(LENS / model / dataset)
        if model == "r1_7b":
            out.append(LENS / dataset)
    return out


def load_entropy(model: str, dataset: str, seed: int) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for folder in lens_dirs(model, dataset, seed):
        if not folder.is_dir():
            continue
        for path in folder.glob("scores_shard*.jsonl"):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                qi = row.get("question_idx")
                step = row.get("decision_step")
                ent = row.get("ans_entropy")
                if qi is None or step is None or ent is None:
                    continue
                out[(int(qi), int(step))] = float(ent)
    return out


def fork_fields(rows: list[dict[str, Any]], left: dict[str, Any]) -> dict[str, Any]:
    ans = left["ans"]
    after_rows = rows[left["end"] + 1 :]
    c_next = float("nan")
    c_break = float("nan")
    next_same = False
    next_step = None
    break_step = None
    if after_rows:
        nxt = after_rows[0]
        c_next = rg.finite(nxt.get("confidence"))
        next_step = int(nxt["stopped_len"])
        next_same = bool(rg.same(nxt.get("final_answer"), ans))
        for row in after_rows:
            if rg.same(row.get("final_answer"), ans):
                continue
            c_break = rg.finite(row.get("confidence"))
            break_step = int(row["stopped_len"])
            break
    left_c = float(left.get("c") if left.get("c") == left.get("c") else float("nan"))
    return {
        "left_c": left_c,
        "c_next": c_next,
        "c_break": c_break,
        "drop_next": left_c - c_next if c_next == c_next else float("nan"),
        "drop_break": left_c - c_break if c_break == c_break else float("nan"),
        "next_same": next_same,
        "next_step": next_step,
        "break_step": break_step,
    }


def load_left_fork(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    xs = after.load_left(model, dataset, seed)
    if not xs:
        return []
    trials_path = room.dense_trial_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    qis = sorted(official) if official else sorted(set(gmap) & set(by) if gmap else by)
    ents = load_entropy(model, dataset, seed)
    out = []
    left_by_qi = {x["question_idx"]: x for x in xs}
    for qi in qis:
        rec = left_by_qi.get(qi)
        if rec is None:
            continue
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        wins = sl.same_windows(rows)
        left = next((w for w in wins if hy.leftover(w)), None)
        if left is None:
            continue
        rec = dict(rec)
        rec.update(fork_fields(rows, left))
        rec["ent_left"] = ents.get((qi, rec["left_step"]), float("nan"))
        rec["ent_next"] = (
            ents.get((qi, rec["next_step"]), float("nan")) if rec["next_step"] else float("nan")
        )
        rec["ent_break"] = (
            ents.get((qi, rec["break_step"]), float("nan")) if rec["break_step"] else float("nan")
        )
        out.append(rec)
    return out


def auroc_key(xs: list[dict[str, Any]], yfn, key: str) -> float:
    y = np.asarray([int(yfn(x)) for x in xs])
    s = np.asarray([float(x.get(key, float("nan"))) for x in xs])
    return tp.auroc(y, s)


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    rows: list[dict[str, Any]] = []
    for _zh, model in after.MODELS:
        for _ds_zh, dataset in after.DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                part = load_left_fork(model, dataset, seed)
                rows.extend(part)
                print(f"{model} {dataset} s{seed} leftover={len(part)}", flush=True)
    wait_ch = [x for x in rows if x["wait_helps"] and x["will_change"]]
    ok_same = [x for x in rows if x["left_ok"] and x["same_as_high"]]
    wait_nv = [x for x in rows if x["wait_helps"] and x["never_high"]]
    ok_nv = [x for x in rows if x["left_ok"] and x["never_high"]]
    # 下一步仍同答：改口还没发生，看把握有没有先掉
    wait_ch_hold = [x for x in wait_ch if x["next_same"]]
    ok_same_hold = [x for x in ok_same if x["next_same"]]

    def line(name: str, xs: list[dict[str, Any]]) -> str:
        return (
            f"| {name} | {len(xs)} | {mean([x['left_c'] for x in xs]):.3f} | "
            f"{mean([x['c_next'] for x in xs]):.3f} | {mean([x['drop_next'] for x in xs]):+.3f} | "
            f"{mean([x['c_break'] for x in xs]):.3f} | {mean([x['ent_left'] for x in xs]):.3f} | "
            f"{mean([x['ent_next'] for x in xs]):.3f} | {mean([x['ent_break'] for x in xs]):.3f} |"
        )

    lines = [
        "# 假平台改口是不是分叉点",
        "",
        "只看第一扇剩窗之后。把握来自逐步 boxed 试答。",
        "试答熵来自已有 dense_lens（有的格子没有，该项为 —）。",
        "下一步仍同答 = 改口还没发生，这是还能干预的一步。",
        "",
        "## 1. 改口前把握掉不掉",
        "",
        "| 切片 | 题 | 剩窗把握 | 下一步把握 | 把握下降 | 改口当步把握 | 剩窗试答熵 | 下一步试答熵 | 改口当步熵 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        line("误杀·后面换答", wait_ch),
        line("误杀·换答、下一步仍同答", wait_ch_hold),
        line("已对·后面同答锁", ok_same),
        line("已对·同答锁、下一步仍同答", ok_same_hold),
        line("误杀·写完才对", wait_nv),
        line("已对·写完才停", ok_nv),
    ]
    # 分：下一步就改口 vs 先再同答一拍
    wait_now = [x for x in wait_ch if not x["next_same"]]
    lines += [
        line("误杀·换答、下一步就改口", wait_now),
        "",
        "## 2. 下一步把握还能不能分开「会换答」",
        "",
        "正类 = 误杀且后面换答。分数越大越像正类。留一集防认数据集。",
        "",
        "| 尺子 | 全体 | 只看下一步仍同答 | 留一奥赛 | 留一 GPQA | 留一 AIME |",
        "|---|---|---|---|---|---|",
    ]
    hold = [x for x in rows if x["next_same"]]

    def loto(xs: list[dict[str, Any]], key: str) -> str:
        parts = []
        groups = {
            "奥赛": [x for x in xs if x["dataset"] == "olympiadbench"],
            "GPQA": [x for x in xs if x["dataset"] == "gpqa-diamond"],
            "AIME": [x for x in xs if x["dataset"] in ("aime24", "aime25")],
        }
        for name, test in groups.items():
            y = np.asarray([int(x["wait_helps"] and x["will_change"]) for x in test])
            s = np.asarray([float(x.get(key, float("nan"))) for x in test])
            if y.min() == y.max() or len(test) < 8:
                parts.append("—")
            else:
                parts.append(tp.fmt(tp.auroc(y, s)))
        return " | ".join(parts)

    y_all = lambda x: x["wait_helps"] and x["will_change"]
    for name, key in (
        ("剩窗把握（对照，越大越不像误杀）", "left_c"),
        ("下一步把握下降", "drop_next"),
        ("1−下一步把握", "c_next_inv"),
        ("1−改口当步把握", "c_break_inv"),
        ("下一步试答熵", "ent_next"),
        ("改口当步试答熵", "ent_break"),
    ):
        for x in rows:
            x["c_next_inv"] = -x["c_next"] if x["c_next"] == x["c_next"] else float("nan")
            x["c_break_inv"] = -x["c_break"] if x["c_break"] == x["c_break"] else float("nan")
        lines.append(
            f"| {name} | {tp.fmt(auroc_key(rows, y_all, key))} | "
            f"{tp.fmt(auroc_key(hold, y_all, key))} | {loto(rows if '下一步仍' not in name else hold, key)} |"
        )
    n_ent = sum(1 for x in rows if x["ent_next"] == x["ent_next"])
    lines += [
        "",
        f"有下一步试答熵的剩窗 {n_ent}/{len(rows)}。",
        "",
        "## 3. 读法",
        "",
        "若误杀换答在「下一步仍同答」时把握已经明显掉、熵已经明显高，才有分叉点可干预。",
        "若只在改口当步才分开，干预已经晚了。若留一集掉到 0.5，还是认数据集。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(rows)} wait_ch={len(wait_ch)} hold={len(wait_ch_hold)}", flush=True)


if __name__ == "__main__":
    main()
