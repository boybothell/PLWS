#!/usr/bin/env python3
"""高把握锁错了：再等 / 写完还能救回多少。对照密探（无则 PUMA）。"""
from __future__ import annotations

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
import report_k4_second_lock as sl

TABLE = AE / "tables/high_wrong_rescue.md"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
    ("14B", "r1_14b"),
    ("32B", "r1_32b"),
    ("Qwen3-4B", "qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b"),
)
DS = (
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def load_cell(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    trials_path = room.dense_trial_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    if not trials_path.is_file():
        return []
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
        a_final = g.get("A_final") or original
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
            host_name = "密探k4"
        elif official:
            host_ok = bool(info.get("compressed_correct"))
            host_tok = int(info.get("compressed_tokens") or 0) + int(
                info.get("tokens_trial_answers") or 0
            )
            host_name = "PUMA"
        else:
            host_ok = orig_ok
            host_tok = float(orig_tok)
            host_name = "写完"
        wins = sl.same_windows(rows)
        highs = [w for w in wins if w["kind"] == "high"]
        first_high = highs[0] if highs else None
        if first_high is None:
            high_ok = False
            later_diff_high_ok = False
            later_diff_any_ok = False
            high_eq_final = False
        else:
            high_ok = bool(low.credit(first_high["ans"], gt, original, orig_ok))
            high_eq_final = bool(rg.same(first_high["ans"], a_final))
            later_diff_high_ok = False
            for w in highs[1:]:
                if rg.same(w["ans"], first_high["ans"]):
                    continue
                if low.credit(w["ans"], gt, original, orig_ok):
                    later_diff_high_ok = True
                    break
            later_diff_any_ok = False
            for w in wins:
                if w["step"] <= first_high["step"]:
                    continue
                if rg.same(w["ans"], first_high["ans"]):
                    continue
                if low.credit(w["ans"], gt, original, orig_ok):
                    later_diff_any_ok = True
                    break
        # 挽救：宿主错，但这条轨迹后面还能对
        continue_end = (not host_ok) and orig_ok
        continue_high = (not host_ok) and later_diff_high_ok
        continue_any = (not host_ok) and later_diff_any_ok
        regen_saved = bool(first_high is not None and (not high_ok) and host_ok)
        dead = bool(first_high is not None and (not high_ok) and (not host_ok) and (not orig_ok) and (not later_diff_any_ok))
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "host_ok": host_ok,
                "host_tok": float(host_tok),
                "orig_ok": orig_ok,
                "orig_tok": float(orig_tok),
                "host_name": host_name,
                "has_high": first_high is not None,
                "high_ok": high_ok,
                "high_eq_final": high_eq_final,
                "regen_saved": regen_saved,
                "continue_end": continue_end,
                "continue_high": continue_high,
                "continue_any": continue_any,
                "dead": dead,
                "high_wrong": bool(first_high is not None and not high_ok),
            }
        )
    return out


def apply_rescue(qs: list[dict[str, Any]], pred) -> dict[str, float]:
    """宿主错且 pred 为真 → 改走写完。"""
    acc = tok = fire = host_acc = host_tok = 0.0
    for q in qs:
        host_acc += int(q["host_ok"])
        host_tok += q["host_tok"]
        if pred(q):
            acc += int(q["orig_ok"])
            tok += q["orig_tok"]
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
    for _zh, model in MODELS:
        for _ds_zh, dataset in DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                part = load_cell(model, dataset, seed)
                rows.extend(part)
                print(f"{model} {dataset} s{seed} n={len(part)}", flush=True)
    lines = [
        "# 高把握锁错了：再等还能不能救",
        "",
        "只看已经出现过高把握四步同答的题。密探会在这里停（有重写就走重写）。",
        "挽救 = 宿主金标是错的，但这条轨迹写完或后面另一把锁是对的。",
        "死锁 = 高把握试答错，宿主也错，写完也错，后面也没有另一答是对的。",
        "对照整集宿主。先知拒绝这把高把握锁、改走写完。token 会变多。",
        "",
    ]
    n = len(rows)
    has = [q for q in rows if q["has_high"]]
    wrong = [q for q in has if q["high_wrong"]]
    lines += [
        f"难集 {n} 题，其中有高把握锁 {len(has)}（{100.0 * len(has) / max(n, 1):.0f}%），",
        f"锁上的试答就错 {len(wrong)}（占有锁题 {100.0 * len(wrong) / max(len(has), 1):.0f}%）。",
        "",
        "## 1. 错锁之后还能不能救",
        "",
        "| 集 | 题 | 有高把握锁 | 试答就错 | 重写已经救回 | 再等写完能救 | 后面另一把高把握能救 | 后面换答能救 | 死锁 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in rows:
        by[f"{q['model']}\t{q['dataset']}"].append(q)
    zh = {t: z for z, t in MODELS}
    dszh = {t: z for z, t in DS}
    for key in sorted(by):
        model, ds = key.split("\t")
        xs = by[key]
        h = [q for q in xs if q["has_high"]]
        w = [q for q in h if q["high_wrong"]]
        lines.append(
            f"| {zh[model]} {dszh[ds]} | {len(xs)} | {len(h)} | {len(w)} | "
            f"{sum(int(q['regen_saved']) for q in xs)} | "
            f"{sum(int(q['continue_end']) for q in xs)} | "
            f"{sum(int(q['continue_high']) for q in xs)} | "
            f"{sum(int(q['continue_any']) for q in xs)} | "
            f"{sum(int(q['dead']) for q in xs)} |"
        )
    lines += [
        "",
        f"合计：试答就错 {sum(int(q['high_wrong']) for q in rows)}，"
        f"重写已救 {sum(int(q['regen_saved']) for q in rows)}，"
        f"写完能救 {sum(int(q['continue_end']) for q in rows)}，"
        f"后一把高把握能救 {sum(int(q['continue_high']) for q in rows)}，"
        f"后面换答能救 {sum(int(q['continue_any']) for q in rows)}，"
        f"死锁 {sum(int(q['dead']) for q in rows)}。"
        f" 错锁里试答已经等于写完终答 {sum(int(q['high_wrong'] and q['high_eq_final']) for q in rows)}。",
        "",
        "## 2. 先知上限：拒绝这把错锁，改走写完",
        "",
        "只在「宿主错且写完对」的题上改走写完。这是 Acc 上限，token 必增。",
        "",
        "| 先知救谁 | 全体难集 | 奥赛+AIME | GPQA |",
        "|---|---|---|---|",
    ]
    oly = [q for q in rows if q["dataset"] != "gpqa-diamond"]
    gpq = [q for q in rows if q["dataset"] == "gpqa-diamond"]
    rules = (
        ("有高把握锁、宿主错、写完对", lambda q: q["has_high"] and q["continue_end"]),
        ("高把握试答错、宿主错、写完对", lambda q: q["high_wrong"] and q["continue_end"]),
        ("后面另一把高把握能救（仍改走写完）", lambda q: q["continue_high"]),
        ("凡是写完能救都救（含从没高把握锁）", lambda q: q["continue_end"]),
    )
    for name, pred in rules:
        lines.append(
            f"| {name} | {cell(apply_rescue(rows, pred))} | "
            f"{cell(apply_rescue(oly, pred))} | {cell(apply_rescue(gpq, pred))} |"
        )
    lines += [
        "",
        "## 3. 分集：有高把握锁且写完能救",
        "",
        "| 集 | 密探 Acc / token | 拒绝错锁改走写完 | 凡是写完能救都救 |",
        "|---|---|---|---|",
    ]
    for key in sorted(by):
        model, ds = key.split("\t")
        xs = by[key]
        host = apply_rescue(xs, lambda _q: False)
        lines.append(
            f"| {zh[model]} {dszh[ds]} | {host['host_acc']:.1f}% / {host['host_tok']:.0f} | "
            f"{cell(apply_rescue(xs, lambda q: q['has_high'] and q['continue_end']))} | "
            f"{cell(apply_rescue(xs, lambda q: q['continue_end']))} |"
        )
    lines += [
        "",
        "读法：单条早停要省 token；高把握错挽救要加 token 换 Acc。",
        "若写完能救的题很少，或几乎都是「试答已经等于写完」（死锁），这条也没有门。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE} n={n}", flush=True)


if __name__ == "__main__":
    main()
