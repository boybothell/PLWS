#!/usr/bin/env python3
"""剩窗之后：同答还坚持几步、何时改口、延迟再决定能不能躲开换答误杀。"""
from __future__ import annotations

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

TABLE = AE / "tables/leftover_after.md"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
    ("14B", "r1_14b"),
    ("Qwen3-4B", "qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b"),
)
DS = (
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)
MODEL_ZH = {tag: zh for zh, tag in MODELS}
DS_ZH = {tag: zh for zh, tag in DS}
DELAYS = (1, 2, 4, 8, 16)


def med(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def fate_of(row: dict[str, Any]) -> str:
    if row["will_change"]:
        return "later_change"
    if row["same_as_high"]:
        return "later_same"
    return "never_high"


def cls_of(row: dict[str, Any]) -> str:
    if row["wait_helps"]:
        return "wait"
    if row["left_ok"]:
        return "ok"
    return "both_wrong"


def timeline(rows: list[dict[str, Any]], left: dict[str, Any]) -> dict[str, Any]:
    ans = left["ans"]
    persist = 0
    first_diff = None
    after = rows[left["end"] + 1 :]
    for row in after:
        step = int(row["stopped_len"])
        if rg.same(row.get("final_answer"), ans):
            if first_diff is None:
                persist += 1
            continue
        if first_diff is None:
            first_diff = step
        break
    wins_after = [w for w in sl.same_windows(rows) if w["step"] > left["step"]]
    new_lock = next((w for w in wins_after if not rg.same(w["ans"], ans)), None)
    high = next((w for w in wins_after if w["kind"] == "high"), None)
    extra_c = []
    for row in after:
        if extra_c and not rg.same(row.get("final_answer"), ans):
            break
        if not rg.same(row.get("final_answer"), ans):
            break
        extra_c.append(rg.finite(row.get("confidence")))
        if len(extra_c) >= 4:
            break
    extra_kind = None
    if len(extra_c) >= 4:
        chunk = extra_c[:4]
        if rg.is_high(chunk):
            extra_kind = "high"
        elif rg.is_low(chunk):
            extra_kind = "low"
        else:
            extra_kind = "mix"
    return {
        "persist": persist,
        "first_diff": first_diff,
        "gap_diff": None if first_diff is None else first_diff - left["step"],
        "new_lock": None if new_lock is None else new_lock["step"] - left["step"],
        "high_gap": None if high is None else high["step"] - left["step"],
        "broke_le": {d: first_diff is not None and first_diff <= left["step"] + d for d in DELAYS},
        "same_for": {d: persist >= d for d in DELAYS},
        "left_c": float(left.get("c") if left.get("c") == left.get("c") else float("nan")),
        "c4": extra_c[3] if len(extra_c) >= 4 else float("nan"),
        "rise4": (extra_c[3] - float(left.get("c"))) if len(extra_c) >= 4 else float("nan"),
        "min4": min(extra_c[:4]) if len(extra_c) >= 4 else float("nan"),
        "extra_kind": extra_kind,
    }


def load_left(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
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
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    qis = sorted(official) if official else sorted(set(gmap) & set(by) if gmap else by)
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
        orig_ok = bool(info.get("original_correct")) if "original_correct" in info else bool(
            low.credit(original, gt, original, True)
        )
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
            host_tok = float(orig_tok or int(last.get("count_reasoning_tokens") or 0))
        wins = sl.same_windows(rows)
        left = next((w for w in wins if hy.leftover(w)), None)
        if left is None:
            continue
        high_after = next((w for w in wins if w["kind"] == "high" and w["step"] > left["step"]), None)
        left_ok = bool(low.credit(left["ans"], gt, original, orig_ok))
        wait_helps = bool(host_ok and not left_ok)
        packed = rg.pack(trials, rows, left["end"], "rescue", original_tokens=orig_tok)
        rec = {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "question_idx": qi,
            "kind": left["kind"],
            "left_step": int(left["step"]),
            "left_ok": left_ok,
            "wait_helps": wait_helps,
            "host_ok": host_ok,
            "host_tok": float(host_tok),
            "left_tok": float(packed["tokens"]),
            "will_change": bool(high_after is not None and not rg.same(left["ans"], high_after["ans"])),
            "same_as_high": bool(high_after is not None and rg.same(left["ans"], high_after["ans"])),
            "never_high": high_after is None,
        }
        rec.update(timeline(rows, left))
        out.append(rec)
    return out


def bucket_line(name: str, xs: list[dict[str, Any]]) -> str:
    if not xs:
        return f"| {name} | 0 | — | — | — | — | — |"
    diffs = [x["gap_diff"] for x in xs if x["gap_diff"] is not None]
    never = sum(1 for x in xs if x["gap_diff"] is None)
    le1 = sum(1 for x in xs if x["gap_diff"] is not None and x["gap_diff"] <= 1)
    le4 = sum(1 for x in xs if x["gap_diff"] is not None and x["gap_diff"] <= 4)
    le8 = sum(1 for x in xs if x["gap_diff"] is not None and x["gap_diff"] <= 8)
    n = len(xs)
    return (
        f"| {name} | {n} | {med([x['persist'] for x in xs]):.0f} | "
        f"{med(diffs):.0f}（无改口 {never}） | "
        f"{100.0 * le1 / n:.0f}% | {100.0 * le4 / n:.0f}% | {100.0 * le8 / n:.0f}% |"
    )


def eval_delay(xs: list[dict[str, Any]], delay: int) -> dict[str, float]:
    """运行时：剩窗后再走 delay 步，仍是同一答才交剩窗。"""
    if not xs:
        return {"d_acc": 0.0, "d_tok": 0.0, "n_fire": 0.0, "wait_fire": 0.0, "n": 0.0, "ok": True}
    acc = tok = fire = wait = 0.0
    host_acc = host_tok = 0.0
    for x in xs:
        host_acc += int(x["host_ok"])
        host_tok += x["host_tok"]
        persist = bool(x["same_for"][delay])
        # 若 delay 步内已经出现高把握同答锁，密探会先收，这扇剩窗交不上
        high_first = x["high_gap"] is not None and x["high_gap"] <= delay and x["same_as_high"]
        do = persist and not high_first
        if do:
            acc += int(x["left_ok"])
            tok += x["left_tok"]  # 近似：切点仍是剩窗；实际还多走了 delay 步
            fire += 1
            wait += int(x["wait_helps"])
        else:
            acc += int(x["host_ok"])
            tok += x["host_tok"]
    n = float(len(xs))
    d_acc = 100.0 * (acc / n - host_acc / n)
    return {
        "d_acc": d_acc,
        "d_tok": tok / n - host_tok / n,
        "n_fire": fire,
        "wait_fire": wait,
        "n": n,
        "ok": acc + 1e-12 >= host_acc,
    }


def cell(rec: dict[str, float]) -> str:
    flag = "" if rec["ok"] else " 伤"
    return (
        f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f}"
        f"（{int(rec['n_fire'])}/{int(rec['n'])}，误杀 {int(rec['wait_fire'])}）{flag}"
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    rows: list[dict[str, Any]] = []
    for _zh, model in MODELS:
        for _ds_zh, dataset in DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                part = load_left(model, dataset, seed)
                rows.extend(part)
                print(f"{model} {dataset} s{seed} leftover={len(part)}", flush=True)
    lines = [
        "# 剩窗之后：同答还坚持多久、改口来得有多快",
        "",
        "探针只打在第一扇非高把握连答窗。这里看窗后的轨迹，不训头。",
        "坚持 = 剩窗后再连续几步仍是这一答。改口间隔 = 第一次出现不同答距剩窗几步。",
        "无改口 = 这条轨迹后头再也没换过试答。",
        "",
        "## 1. 误杀 / 已对 / 两边都错，改口来得有多快",
        "",
        "| 切片 | 题 | 中位坚持步 | 中位改口间隔 | ≤1 步改口 | ≤4 步 | ≤8 步 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    groups = [
        ("误杀，后面换答", lambda x: x["wait_helps"] and x["will_change"]),
        ("误杀，写完才对", lambda x: x["wait_helps"] and x["never_high"]),
        ("已对，后面同答锁", lambda x: x["left_ok"] and x["same_as_high"]),
        ("已对，写完才停", lambda x: x["left_ok"] and x["never_high"]),
        ("已对，低把握", lambda x: x["left_ok"] and x["kind"] == "low"),
        ("已对，混合", lambda x: x["left_ok"] and x["kind"] == "mix"),
        ("两边都错，后面换答", lambda x: (not x["left_ok"]) and (not x["wait_helps"]) and x["will_change"]),
        ("全体剩窗", lambda x: True),
    ]
    for name, pred in groups:
        lines.append(bucket_line(name, [x for x in rows if pred(x)]))

    lines += [
        "",
        "## 2. 按数据集：误杀什么时候改口",
        "",
        "| 集 | 误杀·换答 中位改口 | ≤4 步 | 误杀·写完 中位改口 | 已对·同答锁 中位坚持 |",
        "|---|---|---:|---|---|",
    ]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in rows:
        by_cell[f"{x['model']}\t{x['dataset']}"].append(x)
    for key in sorted(by_cell):
        model, ds = key.split("\t")
        xs = by_cell[key]
        chg = [x for x in xs if x["wait_helps"] and x["will_change"]]
        nv = [x for x in xs if x["wait_helps"] and x["never_high"]]
        ok_same = [x for x in xs if x["left_ok"] and x["same_as_high"]]
        chg_gap = [x["gap_diff"] for x in chg if x["gap_diff"] is not None]
        nv_gap = [x["gap_diff"] for x in nv if x["gap_diff"] is not None]
        le4 = 100.0 * sum(1 for g in chg_gap if g <= 4) / max(len(chg), 1)
        lines.append(
            f"| {MODEL_ZH.get(model, model)} {DS_ZH.get(ds, ds)} | "
            f"{len(chg)} 题 / {med(chg_gap):.0f} 步 | {le4:.0f}% | "
            f"{len(nv)} 题 / {med(nv_gap):.0f} 步 | "
            f"{len(ok_same)} 题 / {med([x['persist'] for x in ok_same]):.0f} 步 |"
        )

    lines += [
        "",
        "## 3. 运行时延迟：剩窗后再走 D 步，仍同答才交",
        "",
        "不看金标、不看后面会不会高把握换答。只看已经走出来的 D 步还是不是这一答。",
        "若 D 步内密探已经高把握锁上同一答，这扇交不上，走密探。",
        "数字是剩窗题相对密探。伤 = Acc 低于密探。token 用切点近似，延迟本身还多走了 D 步，省得会更少。",
        "",
        "| 延迟 | 全体难集剩窗 | 奥赛+AIME | GPQA |",
        "|---|---|---|---|",
    ]
    oly = [x for x in rows if x["dataset"] != "gpqa-diamond"]
    gpq = [x for x in rows if x["dataset"] == "gpqa-diamond"]
    for d in DELAYS:
        lines.append(
            f"| 再走 {d} 步仍同答才交 | {cell(eval_delay(rows, d))} | "
            f"{cell(eval_delay(oly, d))} | {cell(eval_delay(gpq, d))} |"
        )
    lines.append(
        f"| 看见剩窗就交（对照） | {cell(eval_delay_now(rows))} | "
        f"{cell(eval_delay_now(oly))} | {cell(eval_delay_now(gpq))} |"
    )

    def delay_on(xs: list[dict[str, Any]], delay: int, pred) -> dict[str, float]:
        picked = [x for x in xs if pred(x)]
        rest = [x for x in xs if not pred(x)]
        rec = eval_delay(picked, delay)
        # 不满足 pred 的题一律留密探，把它们并回合计
        if not xs:
            return rec
        acc_host = sum(int(x["host_ok"]) for x in xs)
        tok_host = sum(x["host_tok"] for x in xs)
        acc = tok = fire = wait = 0.0
        for x in xs:
            persist = bool(x["same_for"][delay])
            high_first = x["high_gap"] is not None and x["high_gap"] <= delay and x["same_as_high"]
            do = pred(x) and persist and not high_first
            if do:
                acc += int(x["left_ok"])
                tok += x["left_tok"]
                fire += 1
                wait += int(x["wait_helps"])
            else:
                acc += int(x["host_ok"])
                tok += x["host_tok"]
        n = float(len(xs))
        d_acc = 100.0 * (acc / n - acc_host / n)
        return {
            "d_acc": d_acc,
            "d_tok": tok / n - tok_host / n,
            "n_fire": fire,
            "wait_fire": wait,
            "n": n,
            "ok": acc + 1e-12 >= acc_host,
        }

    lines += [
        "",
        "## 3b. 延迟 4 步再收口",
        "",
        "| 规则 | 全体 | 奥赛+AIME | GPQA |",
        "|---|---|---|---|",
        f"| 再走 4 步仍同答，且这扇是混合窗 | {cell(delay_on(rows, 4, lambda x: x['kind'] == 'mix'))} | "
        f"{cell(delay_on(oly, 4, lambda x: x['kind'] == 'mix'))} | "
        f"{cell(delay_on(gpq, 4, lambda x: x['kind'] == 'mix'))} |",
        f"| 再走 4 步仍同答，且这扇是低把握 | {cell(delay_on(rows, 4, lambda x: x['kind'] == 'low'))} | "
        f"{cell(delay_on(oly, 4, lambda x: x['kind'] == 'low'))} | "
        f"{cell(delay_on(gpq, 4, lambda x: x['kind'] == 'low'))} |",
    ]
    remain = [
        x
        for x in rows
        if x["wait_helps"] and x["same_for"][4] and not (x["high_gap"] is not None and x["high_gap"] <= 4 and x["same_as_high"])
    ]
    lines += [
        "",
        f"延迟 4 步后仍会误杀的 {len(remain)} 题：",
        f"低把握 {sum(1 for x in remain if x['kind'] == 'low')}，混合 {sum(1 for x in remain if x['kind'] == 'mix')}；",
        f"后面换答 {sum(1 for x in remain if x['will_change'])}，写完才对 {sum(1 for x in remain if x['never_high'])}，",
        f"后面同答锁 {sum(1 for x in remain if x['same_as_high'])}。",
        "",
        "## 4. 读法",
        "",
        "换答误杀中位 2 步就改口；已对且后面同答锁中位再坚持 13 步。",
        "19 步是新高把握锁，不是第一次改口。延迟能砍掉大部分薄假平台，砍不掉坚持 ≥4 步的低把握假平台。",
        "混合窗 + 延迟 4 步几乎不伤 Acc，但几乎不省 token。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={len(rows)}", flush=True)


def eval_delay_now(xs: list[dict[str, Any]]) -> dict[str, float]:
    if not xs:
        return {"d_acc": 0.0, "d_tok": 0.0, "n_fire": 0.0, "wait_fire": 0.0, "n": 0.0, "ok": True}
    acc = mean([int(x["left_ok"]) for x in xs])
    host = mean([int(x["host_ok"]) for x in xs])
    rec = {
        "d_acc": 100.0 * (acc - host),
        "d_tok": mean([x["left_tok"] for x in xs]) - mean([x["host_tok"] for x in xs]),
        "n_fire": float(len(xs)),
        "wait_fire": float(sum(int(x["wait_helps"]) for x in xs)),
        "n": float(len(xs)),
        "ok": acc + 1e-12 >= host,
    }
    return rec


if __name__ == "__main__":
    main()
