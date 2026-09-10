#!/usr/bin/env python3
"""剥完薄平台之后：就算厚平台已对能完美分开，整集上限在哪。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room
import report_k4_hyps as hy
import report_k4_second_lock as sl
import report_leftover_after as after
import report_leftover_thick as thick

TABLE = AE / "tables/remain_ceiling.md"


def load_full(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not trials_path.is_file():
        return []
    puma_path = dd.puma_stat_path(model, dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)} if regen_path.is_file() else {}
    )
    by: dict[int, list[dict[str, Any]]] = {}
    from collections import defaultdict

    by = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    qis = sorted(official) if official else sorted(set(gmap) & set(by) if gmap else by)
    lefts = {x["question_idx"]: x for x in after.load_left(model, dataset, seed)}
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
        orig_ok = bool(info.get("original_correct")) if "original_correct" in info else False
        orig_tok = int(info.get("original_tokens") or last.get("count_reasoning_tokens") or 0)
        host = regen.get(qi)
        if host:
            host_ok = bool(host.get("compressed_correct"))
            host_tok = int(host.get("compressed_tokens") or 0) + int(
                host.get("tokens_trial_answers") or 0
            )
        elif official:
            host_ok = bool(info.get("compressed_correct"))
            host_tok = int(info.get("compressed_tokens") or 0) + int(
                info.get("tokens_trial_answers") or 0
            )
        else:
            host_ok = orig_ok
            host_tok = float(orig_tok)
        rec = lefts.get(qi)
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "host_ok": host_ok,
                "host_tok": float(host_tok),
                "has_left": rec is not None,
                "left": rec,
            }
        )
    return out


def apply(qs: list[dict[str, Any]], pred) -> dict[str, float]:
    acc = tok = fire = host_acc = host_tok = 0.0
    for q in qs:
        host_acc += int(q["host_ok"])
        host_tok += q["host_tok"]
        left = q["left"]
        if left is not None and pred(left):
            acc += int(left["left_ok"])
            tok += left["left_tok"]
            fire += 1
        else:
            acc += int(q["host_ok"])
            tok += q["host_tok"]
    n = float(len(qs)) or 1.0
    return {
        "d_acc": 100.0 * (acc / n - host_acc / n),
        "d_tok": tok / n - host_tok / n,
        "n_fire": fire,
        "n": n,
        "host_acc": 100.0 * host_acc / n,
        "host_tok": host_tok / n,
    }


def cell(rec: dict[str, float]) -> str:
    return f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f}（{int(rec['n_fire'])}/{int(rec['n'])}）"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    rows: list[dict[str, Any]] = []
    for _zh, model in after.MODELS:
        for _ds_zh, dataset in after.DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                part = load_full(model, dataset, seed)
                rows.extend(part)
                print(f"{model} {dataset} s{seed} n={len(part)}", flush=True)
    rules = (
        ("看见剩窗且试答已对就切（旧上限）", lambda x: x["left_ok"]),
        ("厚平台且试答已对", lambda x: thick.is_thick(x) and x["left_ok"]),
        ("厚平台且已对且写完才停", lambda x: thick.is_thick(x) and x["left_ok"] and x["never_high"]),
        ("厚平台且已对且后面同答锁", lambda x: thick.is_thick(x) and x["left_ok"] and x["same_as_high"]),
        ("厚平台且已对且后面会换答", lambda x: thick.is_thick(x) and x["left_ok"] and x["will_change"]),
    )
    lines = [
        "# 厚平台就算能分开：整集上限",
        "",
        "对照是整集密探（无则 PUMA）。不是只在剩窗题上。",
        "下面全是先知：只切该切片，交剩窗试答，其余走密探。",
        "「还有的做」= 奥赛 / GPQA / AIME 都能 Acc 不降，且 token 少到值得写进去。",
        "",
        "## 1. 整集合计",
        "",
        "| 先知切谁 | 全体难集 | 奥赛+AIME | GPQA |",
        "|---|---|---|---|",
    ]
    oly = [q for q in rows if q["dataset"] != "gpqa-diamond"]
    gpq = [q for q in rows if q["dataset"] == "gpqa-diamond"]
    for name, pred in rules:
        lines.append(
            f"| {name} | {cell(apply(rows, pred))} | {cell(apply(oly, pred))} | {cell(apply(gpq, pred))} |"
        )

    lines += [
        "",
        "## 2. 分集：只看「厚平台已对」和「其中写完才停」",
        "",
        "| 集 | 题 / 密探 | 厚平台已对 | 其中写完才停 | 其中后面同答锁 |",
        "|---|---|---|---|---|",
    ]
    from collections import defaultdict

    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in rows:
        by[f"{q['model']}\t{q['dataset']}"].append(q)
    for key in sorted(by):
        model, ds = key.split("\t")
        xs = by[key]
        host = apply(xs, lambda _x: False)
        lines.append(
            f"| {after.MODEL_ZH[model]} {after.DS_ZH[ds]} | "
            f"{int(host['n'])} / {host['host_acc']:.1f}% · {host['host_tok']:.0f} | "
            f"{cell(apply(xs, rules[1][1]))} | "
            f"{cell(apply(xs, rules[2][1]))} | "
            f"{cell(apply(xs, rules[3][1]))} |"
        )
    lines += [
        "",
        "## 3. 读法",
        "",
        "写完才停那一刀，是密探收不走的。后面同答锁那一刀，密探很快会收，上限里的 token 会缩。",
        "若写完才停只在 GPQA 大、奥赛/AIME 接近 0，就还是一摊题一个方法，不是一句难集故事。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
