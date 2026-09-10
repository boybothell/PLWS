#!/usr/bin/env python3
"""Other already-extracted scalars for high-conf 停对 vs 停错.

Question-level only: one vote = the window the high-conf door would take.
"""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_14b_vs_puma as r14
import report_8b_extra_vs_puma as extra8
import screen_gw_dynamics as dyn
import screen_gw_qvote as qv
import screen_gw_veto as gw

RAW = (
    "hidden_nn_cos",
    "hidden_norm",
    "prefix_ans_cos",
    "last_min_logp",
    "last_mean_logp",
    "ans_entropy",
    "lens_best",
    "lens_best_layer",
    "lens_rise",
    "eigen_k2",
    "eigen_k4",
    "neg_eigen_k4",
    "mid_neg_eigen_k4",
    "emerge_layer",
    "layer_agree",
    "lb_cot",
    "lb_rev",
    "lb_recency5",
    "lb_ans",
    "lookback_ratio",
    "geo_conf",
    "margin",
    "reasoning_pmi",
    "evidence_gain",
    "wrapup_nll",
    "increment_nll",
    "hist_forget",
    "stop_logp",
    "wait_logp",
    "stop_margin",
)


def win_vals(ev, sig):
    return [v for v in (ev["vals"].get(sig) or []) if v == v]


def pack_feat(ev, rows, end):
    rec = {"tag": ev["tag"]}
    for sig in RAW:
        xs = win_vals(ev, sig)
        rec[f"{sig}|last"] = xs[-1] if xs else float("nan")
        rec[f"{sig}|min"] = min(xs) if xs else float("nan")
        rec[f"{sig}|d"] = xs[-1] - xs[0] if len(xs) >= 2 else float("nan")
    window = rows[end + 1 - rg.K : end + 1]
    confs = [rg.finite(x.get("confidence")) for x in window]
    confs = [c for c in confs if c == c]
    rec["conf|min"] = min(confs) if confs else float("nan")
    rec["conf|first"] = confs[0] if confs else float("nan")
    rec["conf|d"] = (confs[-1] - confs[0]) if len(confs) >= 2 else float("nan")
    rec["conf|drop"] = (confs[0] - min(confs)) if confs else float("nan")
    ans = str(window[-1].get("final_answer") or "")
    rec["ans_chars"] = float(len(ans)) if ans else float("nan")
    rec["step"] = float(int(rows[end]["stopped_len"]))
    last = rec["last_mean_logp|last"]
    mn = rec["last_min_logp|last"]
    rec["logp_spread"] = last - mn if last == last and mn == mn else float("nan")
    return rec


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = [rg.CELLS[0], rg.CELLS[2], extra8.CELLS[0], extra8.CELLS[1], r14.CELLS[0]]
    print(
        "一题一票。高置信门会停的那一扇。连对=停对，连错=停错。"
        "只扫现成读数，不新抽。",
        flush=True,
    )
    feat_names = None
    polarity = {}
    for cell in cells:
        pack = gw.load(cell)
        if pack is None:
            continue
        layers.precompute_events(pack, list(RAW))
        votes = []
        for q in pack["questions"]:
            ev = qv.first_high_vote(q)
            if ev is None:
                continue
            votes.append(pack_feat(ev, q["rows"], ev["end"]))
        if feat_names is None:
            feat_names = [k for k in votes[0] if k != "tag"]
            for k in feat_names:
                polarity[k] = []
        pos_n = sum(1 for v in votes if v["tag"] == "G")
        neg_n = sum(1 for v in votes if v["tag"] == "W")
        print(f"\n=== {cell['name']}  票 {len(votes)}  停对{pos_n}/停错{neg_n} ===", flush=True)
        if pos_n < 8 or neg_n < 5:
            print("  停错太少", flush=True)
            continue
        ranked = []
        for key in feat_names:
            pos = [v[key] for v in votes if v["tag"] == "G" and v[key] == v[key]]
            neg = [v[key] for v in votes if v["tag"] == "W" and v[key] == v[key]]
            if len(pos) < 8 or len(neg) < 5:
                continue
            roc = rg.auroc(pos, neg)
            rocf = rg.auroc([-x for x in pos], [-x for x in neg])
            use = max(roc, rocf)
            polarity[key].append((cell["name"], roc - 0.5, len(pos), len(neg)))
            ranked.append((use, roc, rocf, key, dyn.med(pos), dyn.med(neg), len(pos), len(neg)))
        ranked.sort(reverse=True)
        print("  本集最好（一题一票）", flush=True)
        for use, roc, rocf, key, pm, nm, np, nn in ranked[:8]:
            which = "正" if roc >= rocf else "反"
            print(
                f"    {key:22} 停对{np}/停错{nn}  中位 {pm:8.3f}/{nm:8.3f}  "
                f"{roc:.3f} 反{rocf:.3f} ({which})",
                flush=True,
            )
        weak = sum(1 for use, *_ in ranked if use < 0.60)
        print(f"  {len(ranked)} 条里 {weak} 条最高一边仍 <0.60", flush=True)

    print("\n跨集同向且至少 4 集有票、平均|AUROC-0.5|≥0.08", flush=True)
    hits = []
    for key, xs in polarity.items():
        if len(xs) < 4:
            continue
        signs = [1 if d > 0 else -1 if d < 0 else 0 for _, d, *_ in xs]
        if not all(s == signs[0] and s != 0 for s in signs):
            continue
        mean_abs = sum(abs(d) for _, d, *_ in xs) / len(xs)
        if mean_abs < 0.08:
            continue
        hits.append((mean_abs, key, xs))
    hits.sort(reverse=True)
    if not hits:
        print("  没有", flush=True)
    for mean_abs, key, xs in hits:
        bits = "  ".join(f"{n}:{0.5+d:.2f}" for n, d, *_ in xs)
        print(f"  {key:22} 均离0.5 {mean_abs:.3f}  {bits}", flush=True)


if __name__ == "__main__":
    main()
