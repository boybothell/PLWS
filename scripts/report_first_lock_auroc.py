#!/usr/bin/env python3
"""AUROC on first low-conf lock: R=pos, L=neg. One question, one score."""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp

SIGS = (("exit_minus_half", "出口减一半深"), ("stop_margin", "收口比 Wait"))


def win_score(ev, sig) -> float:
    vals = ev["vals"].get(sig) or []
    if not vals or any(v != v for v in vals):
        return float("nan")
    return min(vals)


def first_lock(pack, sig):
    pos, neg = [], []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"] or ev["tag"] not in {"R", "L"}:
                continue
            score = win_score(ev, sig)
            if score != score:
                continue
            if ev["tag"] == "R":
                pos.append(score)
            else:
                neg.append(score)
            break
    return pos, neg


def cell(pos, neg):
    if not pos or not neg:
        return "—"
    return (
        f"{rg.auroc(pos, neg):.2f}（先对{len(pos)}/先错{len(neg)}；"
        f"中位 {rg.p50(pos):.2f} / {rg.p50(neg):.2f}）"
    )


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    named = []
    for c in list(rg.CELLS) + [more.EXTRA]:
        pack = cmp.load_ready(c)
        if pack is None:
            print(f"skip {c['name']}", flush=True)
            continue
        print(f"load {c['name']}", flush=True)
        named.append((c["name"], pack))
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

    header = "| 集 | 出口减一半深 | 收口比 Wait |"
    print("\n每题第一扇低置信锁。正=这扇连对，负=这扇连错。分数=4 步最差。")
    print(header)
    print("|---|---|---|")
    for name, pack in named:
        bits = []
        for sig, _ in SIGS:
            bits.append(cell(*first_lock(pack, sig)))
        print(f"| {name} | {bits[0]} | {bits[1]} |")


if __name__ == "__main__":
    main()
