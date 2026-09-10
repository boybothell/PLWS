#!/usr/bin/env python3
"""One shared layer+agg recipe, ranked by overall Acc vs half-depth and vs PUMA.

Per-set threshold. Same definition across 7B/8B (relative depth).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import screen_puma_layers as sc
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/screen_overall_layer.md"


def aligned_map(pack: dict[str, Any]) -> dict[str, str]:
    ids = pack["_ids"]
    half = pack.get("_half_layer")
    last = max(ids) if ids else None
    q1 = sc.rel_layer(ids, 0.25) if ids else None
    q3 = sc.rel_layer(ids, 0.75) if ids else None
    raw = {
        "末层减一半深": "exit_minus_half",
        "末层整段好写": "last_mean_logp",
        "试答用词有多集中": "neg_ans_entropy",
        "浅层到深层抬了多少": "dola_mean_rise",
        "末层减前半平均": "exit_minus_mean_shallow",
        "最后一层 DoLA 均值": f"dola_mean_l{last}" if last is not None else "",
        "一半深 DoLA 均值": f"dola_mean_l{half}" if half is not None else "",
        "四分之三深 DoLA 均值": f"dola_mean_l{q3}" if q3 is not None else "",
        "一半深好写": f"lens_l{half}" if half is not None else "",
        "四分之三深好写": f"lens_l{q3}" if q3 is not None else "",
        "最后一层好写": f"lens_l{last}" if last is not None else "",
        "末层减四分之一深": f"exit_minus_l{q1}" if q1 is not None else "",
        "末层减四分之三深": f"exit_minus_l{q3}" if q3 is not None else "",
        "DoLA 一半深→最后": f"dola_mean_{min(half, last)}_{max(half, last)}"
        if half is not None and last is not None
        else "",
    }
    have = set(pack["_names"])
    return {zh: sig for zh, sig in raw.items() if sig and sig in have}


def cache_scores(pack: dict[str, Any], sig: str, agg: str, flip: bool) -> list[list[tuple[float, dict[str, Any]]]]:
    out = []
    for q in pack["questions"]:
        rows = []
        for ev in q["_act"]:
            score = sc.window_score(ev["vals"].get(sig) or [], agg, flip)
            if score == score:
                rows.append((score, ev))
        out.append(rows)
    return out


def sweep_cached(pack: dict[str, Any], cached: list[list[tuple[float, dict[str, Any]]]]) -> dict[str, Any] | None:
    xs = [s for rows in cached for s, _ in rows]
    if len(xs) < 8:
        return None
    n = len(pack["questions"])
    puma_acc = sum(int(q["puma_ok"]) for q in pack["questions"]) / n
    puma_tok = sum(q["puma_tok"] for q in pack["questions"]) / n
    points = []
    for thr in [float("inf"), *rg.quantiles(xs, n=41)]:
        acc = tok = 0.0
        n_r = n_l = 0
        for q, rows in zip(pack["questions"], cached):
            hit = next((ev for score, ev in rows if score >= thr), None)
            if hit is None:
                acc += int(q["puma_ok"])
                tok += q["puma_tok"]
                continue
            sim = rg.pack(
                q["trials"],
                q["rows"],
                hit["end"],
                "rescue",
                original_tokens=q["orig_tok"],
                label=hit["tag"],
            )
            ok = rg.same(sim["answer"], q["gt"])
            acc += int(ok)
            tok += sim["tokens"]
            if hit["tag"] == "R":
                n_r += 1
            else:
                n_l += 1
        points.append(
            {
                "threshold": thr,
                "acc": acc / n,
                "tok": tok / n,
                "d_acc": 100.0 * (acc / n - puma_acc),
                "d_tok": tok / n - puma_tok,
                "rescue_R": n_r,
                "rescue_L": n_l,
            }
        )
    chosen = accfirst.pick_acc_first_both(points, puma_acc, puma_tok)
    if chosen is None:
        return None
    chosen["puma_acc"] = puma_acc
    chosen["puma_tok"] = puma_tok
    return chosen


def main() -> None:
    packs = sc.load_ready()
    names = [n for n, _ in packs]
    eight = [n for n in names if n.startswith("8B ")]
    seven = [n for n in names if n.startswith("7B ")]
    maps = {n: aligned_map(p) for n, p in packs}
    shared = sorted(set.intersection(*(set(m) for m in maps.values())))
    print(f"共用 {len(shared)}: {shared}", flush=True)

    half: dict[str, dict[str, Any]] = {}
    grid: dict[tuple[str, str, bool], dict[str, dict[str, Any]]] = {}
    for name, pack in packs:
        amap = maps[name]
        cached_half = cache_scores(pack, "exit_minus_half", "min", False)
        half[name] = sweep_cached(pack, cached_half)
        print(f"half {name} {rg.fmt_pp(half[name]['d_acc'])}", flush=True)
        for zh in shared:
            sig = amap[zh]
            for agg in sc.AGGS:
                for flip in (False, True):
                    got = sweep_cached(pack, cache_scores(pack, sig, agg, flip))
                    if got is None:
                        continue
                    grid.setdefault((zh, agg, flip), {})[name] = got

    rows = []
    for key, recs in grid.items():
        if len(recs) != len(names):
            continue
        zh, agg, flip = key
        d_half = {n: 100.0 * (recs[n]["acc"] - half[n]["acc"]) for n in names}
        mean_vs_half = sum(d_half.values()) / len(names)
        mean_vs_puma = sum(recs[n]["d_acc"] for n in names) / len(names)
        mean8 = sum(recs[n]["d_acc"] for n in eight) / max(len(eight), 1)
        mean7 = sum(recs[n]["d_acc"] for n in seven) / max(len(seven), 1)
        half8 = sum(half[n]["d_acc"] for n in eight) / max(len(eight), 1)
        n_beat = sum(int(d_half[n] > 1e-9) for n in names)
        n_lose = sum(int(d_half[n] < -1e-9) for n in names)
        gpqa7 = recs["7B GPQA"]["d_acc"]
        rows.append(
            {
                "zh": zh,
                "agg": agg,
                "flip": flip,
                "mean_vs_half": mean_vs_half,
                "mean_vs_puma": mean_vs_puma,
                "mean8": mean8,
                "mean7": mean7,
                "half8": half8,
                "d8_vs_half": mean8 - half8,
                "n_beat": n_beat,
                "n_lose": n_lose,
                "gpqa7": gpqa7,
                "recs": recs,
                "d_half": d_half,
            }
        )

    # overall: help 8B, don't dump 7B GPQA below +20, then max mean vs half
    usable = [r for r in rows if r["gpqa7"] + 1e-9 >= 20.0]
    ranked = sorted(usable, key=lambda r: (-r["d8_vs_half"], -r["mean_vs_half"], -r["mean_vs_puma"]))
    any_ranked = sorted(rows, key=lambda r: (-r["d8_vs_half"], -r["mean_vs_half"], -r["mean_vs_puma"]))

    def line(r: dict[str, Any]) -> str:
        flip = "取反" if r["flip"] else "不取反"
        return (
            f"| {r['zh']} | {sc.AGG_ZH[r['agg']]} | {flip} | "
            f"{r['mean_vs_puma']:+.1f} | {r['mean7']:+.1f} | {r['mean8']:+.1f} | "
            f"{r['d8_vs_half']:+.1f} | {r['mean_vs_half']:+.1f} | "
            f"{r['n_beat']}/{len(names)} | {r['gpqa7']:+.1f} |"
        )

    lines = [
        "# 共用一条：整体更好（尤其 8B）",
        "",
        "同一条 = 相对深度对齐后的同一种分数 + 同一种 4 步收法 + 同一种是否取反。门槛本集自选。正确率只对金标。",
        "对照 = 末层减一半深、4 步都过。不要求每集都赢，按 8B 平均涨点、再看全体平均。",
        "下面只留 7B GPQA 仍至少 +20pp（相对 PUMA）的，避免把现在唯一大头赔掉。",
        "",
        f"一半深 4 步都过：7B 平均 {sum(half[n]['d_acc'] for n in seven)/len(seven):+.1f}pp，"
        f"8B 平均 {sum(half[n]['d_acc'] for n in eight)/len(eight):+.1f}pp。",
        "",
        "| 定义 | 聚法 | 取反 | 全体vs PUMA | 7B | 8B | 8B比一半深 | 全体比一半深 | 赢几集 | 7B GPQA |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    print(lines[-2], flush=True)
    seen = 0
    for r in ranked:
        if seen >= 12:
            break
        row = line(r)
        print(row, flush=True)
        lines.append(row)
        seen += 1

    best = ranked[0] if ranked else any_ranked[0]
    lines += ["", "## 最好这一条各集", ""]
    lines.append("| 集 | 一半深 | 这一条 | 比一半深 |")
    lines.append("|---|---|---|---|")
    for n in names:
        h = half[n]
        r = best["recs"][n]
        lines.append(
            f"| {n} | {sw.fmt_pair(h['acc'], h['tok'])}（{rg.fmt_pp(h['d_acc'])}） | "
            f"{sw.fmt_pair(r['acc'], r['tok'])}（{rg.fmt_pp(r['d_acc'])}） | "
            f"{rg.fmt_pp(best['d_half'][n])} / {rg.fmt_tok(r['tok'] - h['tok'])} |"
        )
    print(
        f"最好 {best['zh']} {sc.AGG_ZH[best['agg']]} "
        f"{'取反' if best['flip'] else '不取反'} "
        f"8B {best['mean8']:+.1f} 比一半深 {best['d8_vs_half']:+.1f}",
        flush=True,
    )
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
