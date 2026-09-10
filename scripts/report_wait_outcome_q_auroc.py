#!/usr/bin/env python3
"""Question-level Wait outcome AUROC under the user's 8/22 label.

Denom: has a low-conf consecutive window, and 0.995+Wait did not stop on high-conf.
Pos: Wait stopped and Acc >= Full-CoT, or Wait did not stop and Full-CoT is correct.
Score: min-of-4 stop_margin at the Wait stop window, else brightest low-conf window.
Gate: 0.995 + Wait, no FS. τ is the same Acc-first pick as the threeway Wait column.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_tau995_wait_threeway as tw

TABLE = AE / "tables/wait_outcome_q_auroc.md"
WAIT = "stop_margin"
WANT = {
    "7B MATH",
    "7B 奥赛",
    "7B GPQA",
    "7B AIME24",
    "32B MATH",
    "Qwen3-4B 奥赛",
    "Qwen3-4B GPQA",
    "Qwen3-8B GPQA",
    "Qwen3-8B AIME24",
}


def has_low(q: dict[str, Any]) -> bool:
    return any((not ev["high"]) and (not ev["mixed"]) for ev in q["events"])


def win_min(ev: dict[str, Any]) -> float:
    vals = ev["vals"].get(WAIT) or []
    usable = [v for v in vals if math.isfinite(v)]
    return min(usable) if usable else float("nan")


def brightest(q: dict[str, Any]) -> float:
    best = float("nan")
    for ev in q["events"]:
        if ev["high"] or ev["mixed"]:
            continue
        score = win_min(ev)
        if score == score and (best != best or score > best):
            best = score
    return best


def collect(pack: dict[str, Any], thr: float) -> dict[str, Any]:
    pos, neg = [], []
    n_denom = 0
    bucket = {
        "停且更好": 0,
        "停且都对": 0,
        "停且都错": 0,
        "停且更差": 0,
        "未停且对": 0,
        "未停且错": 0,
        "无Wait分": 0,
    }
    for q in pack["questions"]:
        wait = layers.decide(q, signal=WAIT, threshold=thr, need="all", use_fs=False)
        if wait["branch"] == "conf":
            continue
        if not has_low(q):
            continue
        n_denom += 1
        full_ok = bool(q["orig_ok"])
        stopped = wait["branch"] == "rescue"
        if stopped:
            wait_ok = bool(rg.hit(wait["answer"], q["gt"], q["a_final"], q["orig_ok"]))
            if wait_ok and not full_ok:
                bucket["停且更好"] += 1
                good = True
            elif wait_ok and full_ok:
                bucket["停且都对"] += 1
                good = True
            elif (not wait_ok) and (not full_ok):
                bucket["停且都错"] += 1
                good = True
            else:
                bucket["停且更差"] += 1
                good = False
            score = rg.finite(wait.get("score"))
            if score != score:
                score = brightest(q)
        else:
            good = full_ok
            bucket["未停且对" if full_ok else "未停且错"] += 1
            score = brightest(q)
        if score != score:
            bucket["无Wait分"] += 1
            continue
        (pos if good else neg).append(score)
    return {
        "n_denom": n_denom,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "auroc": rg.auroc(pos, neg) if pos and neg else float("nan"),
        "bucket": bucket,
        "thr": thr,
    }


def fmt_row(name: str, rec: dict[str, Any]) -> str:
    b = rec["bucket"]
    auc = f"{rec['auroc']:.3f}" if rec["auroc"] == rec["auroc"] else "—"
    thr = rec["thr"]
    thr_s = "不开" if not math.isfinite(thr) else f"{thr:.3f}"
    return (
        f"| {name} | {thr_s} | {rec['n_denom']} | {auc} "
        f"| {rec['n_pos']} | {rec['n_neg']} "
        f"| {b['停且更好']} | {b['停且都对']} | {b['停且都错']} | {b['停且更差']} "
        f"| {b['未停且对']} | {b['未停且错']} |"
    )


def main() -> None:
    rg.TAU = 0.995
    header = (
        "| 集 | Wait τ | 分母 | 题级 AUROC | 正 | 负 | "
        "停且更好 | 停且都对 | 停且都错 | 停且更差 | 未停且对 | 未停且错 |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# 题级 AUROC：Wait 结果相对 Full-CoT",
        "",
        "闸门：0.995 + Wait，不开后路强停。门槛和四列表 `0.995+Wait` 同一扇。",
        "分母：有低置信连答窗，且 0.995+Wait 实际没停在高置信（Wait 先停或跑完）。",
        "正类：Wait 停了且这题 Acc ≥ Full-CoT（含都对、都错、救回来），或 Wait 没停且 Full-CoT 对。",
        "负类：Wait 停了但伤了 Full-CoT，或 Wait 没停且 Full-CoT 错。",
        "分数：停在那扇窗的 4 步最差；没停则取还开着的低置信窗里最亮的一扇。",
        "AUROC ≠ 正类占比。正类里同时有「分高该停」和「分低该跑完」，这个数只作对照。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, cells in tw.all_jobs():
        if name not in WANT:
            continue
        loaded = []
        for cell in cells:
            pack = tw.load_one(cell)
            if pack is None:
                print(f"skip {cell['name']} 无读数", flush=True)
                continue
            print(f"load {cell['name']} n={len(pack['questions'])} wait={pack['_wait_cov']:.2f}", flush=True)
            loaded.append(pack)
        if not loaded or len(loaded) != len(cells):
            print(f"skip {name} 未齐 {len(loaded)}/{len(cells)}", flush=True)
            continue
        pack = accfirst.merge_packs(loaded, name) if len(loaded) > 1 else loaded[0]
        pack["_wait_cov"] = min(p["_wait_cov"] for p in loaded)
        nofs_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=False)
        nofs_sum = rg.summarize(nofs_rows)
        wait_only = tw.sweep_wait(pack, nofs_rows, nofs_sum, use_fs=False)
        if wait_only is None:
            print(f"skip {name} sweep 空", flush=True)
            continue
        rec = collect(pack, float(wait_only["threshold"]))
        row = fmt_row(name, rec)
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
