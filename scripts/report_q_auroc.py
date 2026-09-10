#!/usr/bin/env python3
"""Within-question AUROC: on the same problem, R locks vs L locks."""
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


def q_aucs(pack, sig):
    aucs = []
    for q in pack["questions"]:
        pos, neg = [], []
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
        if pos and neg:
            aucs.append(rg.auroc(pos, neg))
    return aucs


def cell_text(aucs):
    if not aucs:
        return "—"
    win = sum(a > 0.5 for a in aucs)
    tie = sum(a == 0.5 for a in aucs)
    return (
        f"{sum(aucs)/len(aucs):.2f}（{len(aucs)}题有对也有错；"
        f"题内对锁更高 {win}/{len(aucs)}，平 {tie}）"
    )


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

    header = "| 集 | 出口减一半深 | 收口比 Wait |"
    sep = "|---|---|---|"
    print("\n只算一道题里同时有低置信连对和连错的题。每题一票。0.5 是猜。")
    print(header)
    print(sep)
    for name, pack in named:
        half = cell_text(q_aucs(pack, "exit_minus_half"))
        wait = cell_text(q_aucs(pack, "stop_margin"))
        print(f"| {name} | {half} | {wait} |")


if __name__ == "__main__":
    main()
