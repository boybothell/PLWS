#!/usr/bin/env python3
"""Wait 4-step shape + aggregation: all / last / k3 / mean / first vs 0.995."""
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
import report_conf_fs_stop_margin as cmp
import report_tau995_wait_threeway as tw

TABLE = AE / "tables/wait_step_agg.md"
WAIT = "stop_margin"
NEEDS = ("all", "last", "k3", "mean", "first")


def _passed(vals: list[float], threshold: float, need: str) -> bool:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return False
    ok = [math.isfinite(v) and v >= threshold for v in vals]
    if need == "all":
        return bool(ok) and all(ok)
    if need == "last":
        return bool(ok) and ok[-1]
    if need == "k3":
        return sum(ok) >= 3
    if need == "mean":
        return (sum(finite) / len(finite)) >= threshold
    if need == "first":
        return math.isfinite(vals[0]) and vals[0] >= threshold
    raise ValueError(need)


def agg_score(vals: list[float], how: str) -> float:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return float("nan")
    if how == "min":
        return min(finite)
    if how == "last":
        return finite[-1] if math.isfinite(vals[-1]) else finite[-1]
    if how == "mean":
        return sum(finite) / len(finite)
    if how == "first":
        return vals[0] if math.isfinite(vals[0]) else finite[0]
    if how == "k3":
        if len(finite) < 3:
            return float("nan")
        return sorted(finite, reverse=True)[2]
    return float("nan")


def windows(pack: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for q in pack["questions"]:
        seen = 0
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            if ev["tag"] not in ("R", "L"):
                continue
            vals = ev["vals"].get(WAIT) or []
            if len(vals) < 2 or any(v != v for v in vals):
                continue
            seen += 1
            out.append(
                {
                    "tag": ev["tag"],
                    "vals": vals,
                    "first_low": seen == 1,
                    "n": len(vals),
                }
            )
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def shape(wins: list[dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {"n": len(wins), "R": 0, "L": 0}
    for tag in ("R", "L", "all"):
        sub = wins if tag == "all" else [w for w in wins if w["tag"] == tag]
        rec[tag] = len(sub)
        if tag in ("R", "L"):
            rec[f"{tag}"] = len(sub)
        pos = {i: [] for i in range(4)}
        last_gt_first = 0
        nondec = 0
        strict = 0
        last_is_max = 0
        first_is_min = 0
        dips = 0
        deltas = []
        usable = 0
        for w in sub:
            v = w["vals"]
            if len(v) < 4:
                continue
            usable += 1
            for i in range(4):
                pos[i].append(v[i])
            last_gt_first += int(v[-1] > v[0] + 1e-12)
            nondec += int(all(v[i] <= v[i + 1] + 1e-9 for i in range(3)))
            strict += int(all(v[i] < v[i + 1] - 1e-12 for i in range(3)))
            last_is_max += int(v[-1] + 1e-12 >= max(v))
            first_is_min += int(v[0] - 1e-12 <= min(v))
            dips += int(any(v[i + 1] + 1e-12 < v[i] for i in range(3)))
            deltas.extend(v[i + 1] - v[i] for i in range(3))
        rec[f"{tag}_n4"] = usable
        rec[f"{tag}_pos"] = [mean(pos[i]) for i in range(4)]
        rec[f"{tag}_last_gt_first"] = last_gt_first / max(usable, 1)
        rec[f"{tag}_nondec"] = nondec / max(usable, 1)
        rec[f"{tag}_strict"] = strict / max(usable, 1)
        rec[f"{tag}_last_max"] = last_is_max / max(usable, 1)
        rec[f"{tag}_first_min"] = first_is_min / max(usable, 1)
        rec[f"{tag}_dip"] = dips / max(usable, 1)
        rec[f"{tag}_ddelta"] = mean(deltas)
    return rec


def auroc_how(wins: list[dict[str, Any]], how: str) -> str:
    pos, neg = [], []
    for w in wins:
        s = agg_score(w["vals"], how)
        if s != s:
            continue
        (pos if w["tag"] == "R" else neg).append(s)
    if not pos or not neg:
        return "—"
    return f"{rg.auroc(pos, neg):.3f}（{len(pos)}/{len(neg)}）"


def sweep_need(pack, base_rows, base_sum, need: str):
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            vals = ev["vals"].get(WAIT) or []
            s = agg_score(vals, "mean" if need == "mean" else "first" if need == "first" else "last" if need == "last" else "min" if need == "all" else "k3")
            if s == s:
                xs.append(s)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = []
    for thr in thrs:
        ours = layers.run_pack(pack, WAIT, thr, need, use_fs=False)
        rec = layers.contrast(base_rows, ours, base_sum)
        rec["threshold"] = thr
        rec["need"] = need
        points.append(rec)
    return accfirst.pick_acc_first_both(points, base_sum["acc"], base_sum["tok"])


def fmt_pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def fmt_rec(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "不开"
    thr = rec["threshold"]
    thr_s = "不开" if thr == float("inf") else f"{thr:.2f}"
    return f"{fmt_pair(rec['acc'], rec['tok'])}（{thr_s}；{rec.get('rescue_R', 0)}/{rec.get('rescue_L', 0)}）"


def pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def main() -> None:
    rg.TAU = 0.995
    layers.passed = _passed
    lines = [
        "# Wait 窗内逐步 + 聚法（PUMA Wait，相对 0.995）",
        "",
        "只看纯低置信度连答窗。分数 = 收口 logp − Wait logp，越大越想停。",
        "聚法门槛各自扫一遍：相对只开 0.995，正确率不降且 token 不多时先取正确率最高。",
        "括号：门槛；放进停对/停错。不开 = 这扇门一题都不放。",
        "",
    ]
    print("\n".join(lines[:6]), flush=True)

    shape_rows = [
        "| 集 | 窗（对/错） | 对：4 步均值 | 错：4 步均值 | 对 last>first | 错 last>first | 对不降 | 错不降 | 对有回落 | 错有回落 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    auc_rows = [
        "| 集 | 4 步最差=都过 | 最后一步 | 4 步里第 3 高 | 4 步平均 | 第一步 |",
        "|---|---|---|---|---|---|",
    ]
    acc_rows = [
        "| 集 | 0.995 | 都过 | 最后一步 | 3/4 | 平均 | 第一步 |",
        "|---|---|---|---|---|---|---|",
    ]
    first_shape = [
        "| 集 | 第一扇对 last>first | 第一扇错 last>first | 第一扇对不降 | 第一扇错不降 |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, cells in tw.all_jobs():
        loaded = []
        for cell in cells:
            pack = tw.load_one(cell)
            if pack is None:
                continue
            print(f"load {cell['name']} n={len(pack['questions'])} wait={pack['_wait_cov']:.2f}", flush=True)
            loaded.append(pack)
        if not loaded or len(loaded) != len(cells):
            print(f"skip {name} seeds={len(loaded)}/{len(cells)}", flush=True)
            continue
        pack = accfirst.merge_packs(loaded, name) if len(loaded) > 1 else loaded[0]
        pack["_wait_cov"] = min(p["_wait_cov"] for p in loaded)
        if pack["_wait_cov"] < 0.5:
            print(f"skip {name} wait={pack['_wait_cov']:.2f}", flush=True)
            continue
        wins = windows(pack)
        first = [w for w in wins if w["first_low"]]
        sh = shape(wins)
        sh1 = shape(first)
        rpos = " / ".join(f"{x:.2f}" for x in sh["R_pos"])
        lpos = " / ".join(f"{x:.2f}" for x in sh["L_pos"])
        shape_rows.append(
            f"| {name} | {sh['R']}/{sh['L']} | {rpos} | {lpos} | "
            f"{pct(sh['R_last_gt_first'])} | {pct(sh['L_last_gt_first'])} | "
            f"{pct(sh['R_nondec'])} | {pct(sh['L_nondec'])} | "
            f"{pct(sh['R_dip'])} | {pct(sh['L_dip'])} |"
        )
        first_shape.append(
            f"| {name} | {pct(sh1['R_last_gt_first'])} | {pct(sh1['L_last_gt_first'])} | "
            f"{pct(sh1['R_nondec'])} | {pct(sh1['L_nondec'])} |"
        )
        auc_rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    auroc_how(wins, "min"),
                    auroc_how(wins, "last"),
                    auroc_how(wins, "k3"),
                    auroc_how(wins, "mean"),
                    auroc_how(wins, "first"),
                ]
            )
            + " |"
        )
        nofs_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=False)
        nofs_sum = rg.summarize(nofs_rows)
        picked = {need: sweep_need(pack, nofs_rows, nofs_sum, need) for need in NEEDS}
        acc_rows.append(
            f"| {name} | {fmt_pair(nofs_sum['acc'], nofs_sum['tok'])} | "
            + " | ".join(fmt_rec(picked[need]) for need in NEEDS)
            + " |"
        )
        print(shape_rows[-1], flush=True)
        print(acc_rows[-1], flush=True)

    body = [
        "## 窗内 4 步长什么样",
        "",
        *shape_rows,
        "",
        "只看每题第一扇低置信窗：",
        "",
        *first_shape,
        "",
        "## 窗级分开程度（AUROC，对窗 vs 错窗）",
        "",
        *auc_rows,
        "",
        "## 相对 0.995 的 Acc / token（各自选门槛）",
        "",
        *acc_rows,
        "",
    ]
    TABLE.write_text("\n".join(lines + body))
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
