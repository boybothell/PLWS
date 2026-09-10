#!/usr/bin/env python3
"""Is one aligned layer+agg better than half-depth-all on every 7B/8B set?"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import screen_puma_layers as sc
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/screen_shared_layer.md"


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
        "末层减前半最浅": "exit_minus_min_shallow",
        "末层减前半平均": "exit_minus_mean_shallow",
        "末层减后半平均": "exit_minus_mean_deep",
        "最后一层 DoLA 均值": f"dola_mean_l{last}" if last is not None else "",
        "一半深 DoLA 均值": f"dola_mean_l{half}" if half is not None else "",
        "四分之三深 DoLA 均值": f"dola_mean_l{q3}" if q3 is not None else "",
        "一半深好写": f"lens_l{half}" if half is not None else "",
        "四分之三深好写": f"lens_l{q3}" if q3 is not None else "",
        "最后一层好写": f"lens_l{last}" if last is not None else "",
        "末层减四分之一深": f"exit_minus_l{q1}" if q1 is not None else "",
        "末层减四分之三深": f"exit_minus_l{q3}" if q3 is not None else "",
        "DoLA 一半深→最后": f"dola_mean_{half}_{last}" if half is not None and last is not None else "",
        "DoLA 四分之三→最后": f"dola_mean_{q3}_{last}" if q3 is not None and last is not None else "",
        "lens 一半深→最后": f"lens_{half}_{last}" if half is not None and last is not None else "",
        "四分之三深 JSD": f"dola_jsd_l{q3}" if q3 is not None else "",
    }
    have = set(pack["_names"])
    out = {}
    for zh, sig in raw.items():
        if sig and sig in have:
            out[zh] = sig
    return out


def main() -> None:
    packs = sc.load_ready()
    names = [n for n, _ in packs]
    # aligned keys that exist on every pack
    maps = {n: aligned_map(p) for n, p in packs}
    shared_zh = sorted(set.intersection(*(set(m) for m in maps.values())))
    print(f"共用定义 {len(shared_zh)}: {shared_zh}", flush=True)

    grid: dict[tuple[str, str, bool], dict[str, dict[str, Any]]] = {}
    half: dict[str, dict[str, Any]] = {}
    for name, pack in packs:
        amap = maps[name]
        rec = sc.sweep_cfg(pack, "exit_minus_half", "min", False)
        half[name] = rec
        print(f"half {name} {sw.fmt_pair(rec['acc'], rec['tok'])} {rg.fmt_pp(rec['d_acc'])}", flush=True)
        for zh in shared_zh:
            sig = amap[zh]
            for agg in sc.AGGS:
                for flip in (False, True):
                    got = sc.sweep_cfg(pack, sig, agg, flip)
                    if got is None:
                        continue
                    grid.setdefault((zh, agg, flip), {})[name] = got

    lines = [
        "# 有没有同一条在 7B/8B 全部集上更好",
        "",
        "同一条 = 相对深度对齐后的同一种分数 + 同一种 4 步收法 + 同一种是否取反。",
        "门槛仍本集自选。对照 = 末层减一半深、4 步都过、不取反。",
        "更好 = 正确率高于对照；持平 = 正确率一样。",
        "",
    ]
    winners = []
    for key, recs in sorted(grid.items(), key=lambda kv: kv[0]):
        zh, agg, flip = key
        if len(recs) != len(names):
            continue
        beats = []
        ties = []
        loses = []
        for name in names:
            h = half[name]
            r = recs[name]
            if r["acc"] > h["acc"] + 1e-12:
                beats.append(name)
            elif r["acc"] + 1e-12 >= h["acc"]:
                ties.append(name)
            else:
                loses.append(name)
        if not loses and (beats or ties):
            winners.append((key, recs, beats, ties, loses))

    strict = [w for w in winners if len(w[2]) == len(names)]
    no_lose = [w for w in winners if not w[4] and w[2]]
    print(f"全赢 {len(strict)}  不输且至少一集赢 {len(no_lose)}", flush=True)

    lines.append(f"现有格子：{', '.join(names)}。")
    lines.append(f"全赢（每集正确率都高于一半深4步都过）：**{len(strict)} 条**。")
    lines.append(f"不输且至少一集更好：**{len(no_lose)} 条**。")
    lines.append("")
    if no_lose:
        lines += [
            "| 定义 | 聚法 | 取反 | 赢的集 | 持平的集 | 各集比一半深 |",
            "|---|---|---|---|---|---|",
        ]
        def rank(item):
            key, recs, beats, ties, _ = item
            mean_d = sum(recs[n]["acc"] - half[n]["acc"] for n in names) / len(names)
            return (-len(beats), -mean_d)

        for (zh, agg, flip), recs, beats, ties, _ in sorted(no_lose, key=rank):
            deltas = ", ".join(
                f"{n} {rg.fmt_pp(100.0 * (recs[n]['acc'] - half[n]['acc']))}"
                for n in names
            )
            row = (
                f"| {zh} | {sc.AGG_ZH[agg]} | {'是' if flip else '否'} | "
                f"{len(beats)} | {len(ties)} | {deltas} |"
            )
            print(row, flush=True)
            lines.append(row)
    else:
        lines.append("没有一条能在全部集上不输给一半深、4 步都过。")
        # closest: fewest losses, then most beats
        scored = []
        for key, recs in grid.items():
            if len(recs) != len(names):
                continue
            lose = sum(int(recs[n]["acc"] + 1e-12 < half[n]["acc"]) for n in names)
            beat = sum(int(recs[n]["acc"] > half[n]["acc"] + 1e-12) for n in names)
            mean_d = sum(recs[n]["acc"] - half[n]["acc"] for n in names) / len(names)
            scored.append((lose, -beat, -mean_d, key, recs))
        scored.sort()
        lines += [
            "",
            "最接近的几条（仍有集更差）：",
            "",
            "| 定义 | 聚法 | 取反 | 赢 | 输 | 输在哪 |",
            "|---|---|---|---:|---:|---|",
        ]
        for lose, nbeat, _, (zh, agg, flip), recs in scored[:8]:
            beat = -nbeat
            lost = [
                f"{n} {rg.fmt_pp(100.0 * (recs[n]['acc'] - half[n]['acc']))}"
                for n in names
                if recs[n]["acc"] + 1e-12 < half[n]["acc"]
            ]
            row = (
                f"| {zh} | {sc.AGG_ZH[agg]} | {'是' if flip else '否'} | "
                f"{beat} | {lose} | {', '.join(lost) or '—'} |"
            )
            print(row, flush=True)
            lines.append(row)

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
