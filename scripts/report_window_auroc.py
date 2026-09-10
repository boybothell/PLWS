#!/usr/bin/env python3
"""Window-level AUROC: all low-conf locks, R vs L. Gate score = min of 4 steps."""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp

SIGS = (
    ("exit_minus_half", "出口减一半深"),
    ("stop_margin", "收口比 Wait"),
)


def win_score(ev, sig) -> float:
    vals = ev["vals"].get(sig) or []
    if not vals or any(v != v for v in vals):
        return float("nan")
    return min(vals)


def collect(pack, sig):
    pos, neg = [], []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            score = win_score(ev, sig)
            if score != score:
                continue
            if ev["tag"] == "R":
                pos.append(score)
            elif ev["tag"] == "L":
                neg.append(score)
    return pos, neg


def row(name, pack):
    cells = [name, str(len(pack["questions"]))]
    for sig, _zh in SIGS:
        pos, neg = collect(pack, sig)
        auc = rg.auroc(pos, neg)
        if not pos or not neg:
            cells.append("—")
            continue
        cells.append(
            f"{auc:.2f}（对{len(pos)}/错{len(neg)}；中位 {rg.p50(pos):.2f} / {rg.p50(neg):.2f}）"
        )
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    named = []
    for cell in list(rg.CELLS) + [more.EXTRA]:
        pack = cmp.load_ready(cell)
        if pack is None:
            print(f"skip {cell['name']}", flush=True)
            continue
        print(f"load {cell['name']}", flush=True)
        named.append((cell["name"], pack))
    for ds, zh in (("aime24", "7B AIME24"), ("aime25", "7B AIME25")):
        parts = []
        for seed in accfirst.SEEDS:
            pack = cmp.load_ready(accfirst.aime_cell(ds, seed))
            if pack is None:
                continue
            pack["_cov"] = 1.0
            parts.append(pack)
            print(f"load {ds}-s{seed}", flush=True)
        if len(parts) == 4:
            named.append((zh, accfirst.merge_packs(parts, zh)))

    header = "| 集 | 题数 | 出口减一半深 | 收口比 Wait |"
    sep = "|---|---:|---|---|"
    print("\n所有低置信窗，连对 vs 连错。分数=窗内 4 步最差。0.5 是猜。")
    print(header)
    print(sep)
    lines = [
        "# 低置信窗：对锁 vs 错锁",
        "",
        "只这一条：所有低置信连答窗摊开。正=连对，负=连错。分数=4 步最差（门用的数）。",
        "",
        header,
        sep,
    ]
    for name, pack in named:
        line = row(name, pack)
        print(line)
        lines.append(line)
    out = AE / "tables/window_auroc_r_vs_l.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n写成 {out}")


if __name__ == "__main__":
    main()
