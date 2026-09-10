#!/usr/bin/env python3
"""Threshold sensitivity and shared/normalized freeze for last-layer 好写."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import screen_puma_layers as sc
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/screen_dola_thr_share.md"


def dola_name(pack: dict[str, Any]) -> str:
    return f"dola_mean_l{max(pack['_ids'])}"


def last_scores(pack: dict[str, Any], sig: str) -> list[float]:
    xs = []
    for q in pack["questions"]:
        for ev in q["_act"]:
            vals = ev["vals"].get(sig) or []
            if vals and vals[-1] == vals[-1]:
                xs.append(vals[-1])
    return xs


def stats(xs: list[float]) -> tuple[float, float]:
    xs = sorted(x for x in xs if x == x)
    if len(xs) < 4:
        return float("nan"), float("nan")
    mid = xs[len(xs) // 2]
    q1 = xs[len(xs) // 4]
    q3 = xs[(3 * len(xs)) // 4]
    spread = max(q3 - q1, 1e-6)
    return mid, spread


def eval_at(pack: dict[str, Any], sig: str, scores: list[list[float]], thr: float) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = puma_acc = puma_tok = 0.0
    for q, xs in zip(pack["questions"], scores):
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        hit = None
        for ev, s in zip(q["_act"], xs):
            if s == s and s >= thr:
                hit = ev
                break
        if hit is None:
            acc += int(q["puma_ok"])
            tok += q["puma_tok"]
            continue
        sim = rg.pack(
            q["trials"], q["rows"], hit["end"], "rescue",
            original_tokens=q["orig_tok"], label=hit["tag"],
        )
        ok = rg.same(sim["answer"], q["gt"])
        acc += int(ok)
        tok += sim["tokens"]
    return {
        "acc": acc / n,
        "tok": tok / n,
        "d_acc": 100.0 * (acc / n - puma_acc / n),
        "d_tok": tok / n - puma_tok / n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
    }


def per_q_raw(pack: dict[str, Any], sig: str) -> list[list[float]]:
    out = []
    for q in pack["questions"]:
        out.append([
            (ev["vals"].get(sig) or [float("nan")])[-1]
            if (ev["vals"].get(sig) or [float("nan")])
            else float("nan")
            for ev in q["_act"]
        ])
    return out


def per_q_norm(raw: list[list[float]], mid: float, spread: float) -> list[list[float]]:
    return [[(s - mid) / spread if s == s else float("nan") for s in xs] for xs in raw]


def main() -> None:
    packs = [(n, p) for n, p in sc.load_ready()]
    items = []
    for name, pack in packs:
        sig = dola_name(pack)
        raw = per_q_raw(pack, sig)
        xs = last_scores(pack, sig)
        mid, spread = stats(xs)
        local = sc.sweep_cfg(pack, sig, "last", False)
        items.append({
            "name": name,
            "pack": pack,
            "sig": sig,
            "raw": raw,
            "mid": mid,
            "spread": spread,
            "local": local,
            "xs": xs,
        })
        print(
            f"{name} mid={mid:.3f} iqr={spread:.3f} "
            f"local_thr={local['threshold']:.3f} {rg.fmt_pp(local['d_acc'])}",
            flush=True,
        )

    seven = [x for x in items if x["name"].startswith("7B ")]
    eight = [x for x in items if x["name"].startswith("8B ")]

    def mean_d(xs: list[dict[str, Any]], recs: list[dict[str, Any]]) -> float:
        return sum(r["d_acc"] for r in recs) / max(len(recs), 1)

    lines = [
        "# 好写门槛：敏不敏感、能不能共用、能不能先归一化",
        "",
        "分数 = 最后一层对试答正文的平均对数概率，只看低置信窗最后一步。",
        "本集最好 = 每集自己选门槛。冻 = 所有集用同一个数。",
        "归一化 = (分数 − 该模型低窗中位数) / 四分位距，再用同一个归一化门槛。",
        "",
    ]

    # sensitivity: local best vs nearby
    lines += ["## 本集门槛附近敏不敏感", "", "| 集 | 本集最好 | 门槛略松 | 门槛略紧 | 冻成 8B 中间档 −0.08 |", "|---|---|---|---|---|"]
    print(lines[-2], flush=True)
    for x in items:
        loc = x["local"]
        # nearby: ±0.2 in log space if scale allows, else ±20% of |thr|
        step = 0.2 if abs(loc["threshold"]) < 2 else 0.25 * abs(loc["threshold"])
        loose = eval_at(x["pack"], x["sig"], x["raw"], loc["threshold"] - step)
        tight = eval_at(x["pack"], x["sig"], x["raw"], loc["threshold"] + step)
        freeze = eval_at(x["pack"], x["sig"], x["raw"], -0.08)
        row = (
            f"| {x['name']} | {loc['threshold']:.3f} → {rg.fmt_pp(loc['d_acc'])} | "
            f"{rg.fmt_pp(loose['d_acc'])} | {rg.fmt_pp(tight['d_acc'])} | "
            f"{rg.fmt_pp(freeze['d_acc'])} |"
        )
        print(row, flush=True)
        lines.append(row)

    def pick_shared(group: list[dict[str, Any]], kind: str) -> tuple[float, list[dict[str, Any]]]:
        # grid from pooled quantiles of raw or z
        pool = []
        cached = []
        for x in group:
            if kind == "raw":
                cached.append(x["raw"])
                pool.extend(x["xs"])
            else:
                z = per_q_norm(x["raw"], x["mid"], x["spread"])
                cached.append(z)
                for xs in z:
                    pool.extend(s for s in xs if s == s)
        thrs = rg.quantiles(pool, n=41)
        best_thr, best_recs, best_mean = None, None, -1e9
        for thr in thrs:
            recs = [eval_at(x["pack"], x["sig"], c, thr) for x, c in zip(group, cached)]
            mean = sum(r["d_acc"] for r in recs) / len(recs)
            if best_thr is None or mean > best_mean + 1e-9:
                best_thr, best_recs, best_mean = thr, recs, mean
        return best_thr, best_recs

    lines += ["", "## 冻一个数（原始尺子）", "", "| 范围 | 冻的门槛 | 各集 vs PUMA | 平均 | 本集最好平均 |", "|---|---|---|---:|---:|"]
    for title, group in (("7B 五集", seven), ("8B 五集", eight), ("7B+8B 十集", items)):
        thr, recs = pick_shared(group, "raw")
        local_mean = sum(x["local"]["d_acc"] for x in group) / len(group)
        bits = ", ".join(f"{x['name']} {rg.fmt_pp(r['d_acc'])}" for x, r in zip(group, recs))
        row = f"| {title} | {thr:.3f} | {bits} | {sum(r['d_acc'] for r in recs)/len(recs):+.1f}pp | {local_mean:+.1f}pp |"
        print(row, flush=True)
        lines.append(row)

    lines += ["", "## 先按模型低窗中位数/四分位距归一化，再冻一个数", "", "| 范围 | 归一化门槛 | 各集 vs PUMA | 平均 | 本集最好平均 |", "|---|---|---|---:|---:|"]
    for title, group in (("7B 五集", seven), ("8B 五集", eight), ("7B+8B 十集", items)):
        thr, recs = pick_shared(group, "z")
        local_mean = sum(x["local"]["d_acc"] for x in group) / len(group)
        bits = ", ".join(f"{x['name']} {rg.fmt_pp(r['d_acc'])}" for x, r in zip(group, recs))
        row = f"| {title} | {thr:.3f} | {bits} | {sum(r['d_acc'] for r in recs)/len(recs):+.1f}pp | {local_mean:+.1f}pp |"
        print(row, flush=True)
        lines.append(row)

    # exp-scale shared (geo-conf equivalent), only makes sense where scores aren't wildly negative
    lines += [
        "",
        "## 先 e^好写 变成 0–1，再冻一个把握门槛",
        "",
        "| 冻的把握 | 7B 平均 | 8B 平均 | 全体平均 | 8B 各集 |",
        "|---:|---:|---:|---:|---|",
    ]
    for geo in (0.50, 0.70, 0.80, 0.85, 0.88, 0.90, 0.93, 0.95, 0.97):
        thr = math.log(geo)
        recs = {x["name"]: eval_at(x["pack"], x["sig"], x["raw"], thr) for x in items}
        m7 = sum(recs[x["name"]]["d_acc"] for x in seven) / 5
        m8 = sum(recs[x["name"]]["d_acc"] for x in eight) / 5
        mall = sum(recs[x["name"]]["d_acc"] for x in items) / 10
        bits = ", ".join(f"{x['name'][3:]} {rg.fmt_pp(recs[x['name']]['d_acc'])}" for x in eight)
        row = f"| {geo:.2f} | {m7:+.1f} | {m8:+.1f} | {mall:+.1f} | {bits} |"
        print(row, flush=True)
        lines.append(row)

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
