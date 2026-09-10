#!/usr/bin/env python3
"""G vs W change features. Question-level AUROC only.

One vote per question that the high-conf door would actually stop.
Vote = that first high-conf lock (step≥10). G=停对, W=停错.
Do not pool later windows, steps, or layer-pairs as extra votes.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_default_dense_gate as old
import replay_rescue_R_gate as rg
import report_14b_vs_puma as r14
import report_8b_extra_vs_puma as extra8
import screen_gw_dynamics as dyn
import screen_gw_veto as gw


def first_high_vote(q):
    """Same walk as the running high-conf door. One window or none."""
    events_by_end = {ev["end"]: ev for ev in q["events"]}
    trials, rows = q["trials"], q["rows"]
    lens = old.step_char_lens(trials)
    red_run = 0
    best_conf = float("-inf")
    probed = []
    walked = 0
    for end, row in enumerate(rows):
        step = int(row["stopped_len"])
        while walked < len(trials) and int(trials[walked]["stopped_len"]) <= step:
            tstep = int(trials[walked]["stopped_len"])
            if tstep >= rg.FS_MIN_STEP:
                red_run = red_run + 1 if old.is_red(tstep, lens) else 0
            walked += 1
        conf = rg.finite(row.get("confidence"))
        answer = str(row.get("final_answer") or "")
        if answer and conf == conf:
            best_conf = max(best_conf, conf)
            probed.append({"step": step, "answer": answer, "conf": conf})
        ev = events_by_end.get(end)
        if ev is not None and step >= rg.MSS and ev["high"]:
            if ev["tag"] in {"G", "W"}:
                return ev
            return None
        pick = rg.fs_ready(step, red_run, best_conf, probed)
        if pick is not None:
            return None
    return None


def auroc_q(pos, neg):
    if len(pos) < 8 or len(neg) < 5:
        return float("nan")
    return rg.auroc(pos, neg)


def split(votes, key):
    pos = [v[key] for v in votes if v["tag"] == "G" and v[key] == v[key]]
    neg = [v[key] for v in votes if v["tag"] == "W" and v[key] == v[key]]
    return pos, neg


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = [rg.CELLS[0], rg.CELLS[2], extra8.CELLS[0], extra8.CELLS[1], r14.CELLS[0]]
    print(
        "一题一票。只有高置信门会停的题投票；票=那一扇窗。"
        "连对=停对，连错=停错。AUROC 只在这些票上算。",
        flush=True,
    )
    keys = [
        ("wait_d", "Wait 4步末−首"),
        ("wait_s", "Wait 后半−前半"),
        ("stop_d", "</think> 4步变化"),
        ("q_to_h", "层 ¼→½"),
        ("h_to_t", "层 ½→¾"),
        ("t_to_L", "层 ¾→末"),
        ("h_to_L", "层 ½→末"),
        ("late_vs_mid", "后半跳−前半跳"),
        ("peak_rel", "最大跳有多深"),
        ("peak_jump", "最大一跳多大"),
        ("shape_qh", "去高低后 ¼→½"),
        ("shape_ht", "去高低后 ½→¾"),
        ("shape_tl", "去高低后 ¾→末"),
        ("dola_qh", "首词 ¼→½"),
        ("dola_ht", "首词 ½→¾"),
        ("dola_tl", "首词 ¾→末"),
    ]
    polarity = {k: [] for k, _ in keys}

    for cell in cells:
        pack = gw.load(cell)
        if pack is None:
            continue
        ids = more.discover_layers(pack["scores"])
        last_L = max(ids)
        n = last_L + 1
        qL = min(ids, key=lambda L: abs(L - n // 4))
        hL = min(ids, key=lambda L: abs(L - n // 2))
        tL = min(ids, key=lambda L: abs(L - (3 * n) // 4))
        names = [f"lens_l{L}" for L in ids] + [f"dola_logp_l{L}" for L in ids]
        names += ["stop_margin", "stop_logp", "wait_logp", "last_mean_logp", f"exit_minus_l{hL}"]
        layers.precompute_events(pack, names)
        votes = []
        n_q = len(pack["questions"])
        for q in pack["questions"]:
            ev = first_high_vote(q)
            if ev is None:
                continue
            rec = dyn.feat_row(ev, ids, last_L, qL, hL, tL)
            rec["tag"] = ev["tag"]
            # shape: subtract mean of the 4 bins so only the bend remains
            bins = [rec["lens_q"], rec["lens_h"], rec["lens_t"], rec["lens_L"]]
            if all(x == x for x in bins):
                mu = sum(bins) / 4
                sq, sh, st, sl = (x - mu for x in bins)
                rec["shape_qh"] = sh - sq
                rec["shape_ht"] = st - sh
                rec["shape_tl"] = sl - st
            else:
                rec["shape_qh"] = rec["shape_ht"] = rec["shape_tl"] = float("nan")

            def last_s(sig):
                xs = dyn.series(ev, sig)
                return xs[-1] if xs else float("nan")

            dq, dh, dt, dl = (last_s(f"dola_logp_l{L}") for L in (qL, hL, tL, last_L))
            rec["dola_qh"] = dh - dq if dh == dh and dq == dq else float("nan")
            rec["dola_ht"] = dt - dh if dt == dt and dh == dh else float("nan")
            rec["dola_tl"] = dl - dt if dl == dl and dt == dt else float("nan")
            votes.append(rec)
        pos_n = sum(1 for v in votes if v["tag"] == "G")
        neg_n = sum(1 for v in votes if v["tag"] == "W")
        print(
            f"\n=== {cell['name']}  会停高置信 {len(votes)}/{n_q}  "
            f"停对{pos_n}/停错{neg_n}  层{qL}/{hL}/{tL}/{last_L} ===",
            flush=True,
        )
        if pos_n < 8 or neg_n < 5:
            print("  停错票太少，不算 AUROC", flush=True)
            continue
        print(
            f"  层曲线中位  ¼/½/¾/末  "
            f"停对 {dyn.med([v['lens_q'] for v in votes if v['tag']=='G']):.2f}/"
            f"{dyn.med([v['lens_h'] for v in votes if v['tag']=='G']):.2f}/"
            f"{dyn.med([v['lens_t'] for v in votes if v['tag']=='G']):.2f}/"
            f"{dyn.med([v['lens_L'] for v in votes if v['tag']=='G']):.2f}  "
            f"停错 {dyn.med([v['lens_q'] for v in votes if v['tag']=='W']):.2f}/"
            f"{dyn.med([v['lens_h'] for v in votes if v['tag']=='W']):.2f}/"
            f"{dyn.med([v['lens_t'] for v in votes if v['tag']=='W']):.2f}/"
            f"{dyn.med([v['lens_L'] for v in votes if v['tag']=='W']):.2f}",
            flush=True,
        )
        early_g = sum(1 for v in votes if v["tag"] == "G" and v["peak_rel"] == v["peak_rel"] and v["peak_rel"] <= 0.55)
        late_g = sum(1 for v in votes if v["tag"] == "G" and v["peak_rel"] == v["peak_rel"] and v["peak_rel"] >= 0.75)
        early_w = sum(1 for v in votes if v["tag"] == "W" and v["peak_rel"] == v["peak_rel"] and v["peak_rel"] <= 0.55)
        late_w = sum(1 for v in votes if v["tag"] == "W" and v["peak_rel"] == v["peak_rel"] and v["peak_rel"] >= 0.75)
        print(
            f"  最大跳在前半/后段  停对 {early_g}/{late_g}  停错 {early_w}/{late_w}",
            flush=True,
        )
        print("  一题一票 AUROC（停对分是否高于停错）", flush=True)
        for key, zh in keys:
            pos, neg = split(votes, key)
            roc = auroc_q(pos, neg)
            rocf = auroc_q([-x for x in pos], [-x for x in neg])
            if roc != roc:
                continue
            better = "正" if roc >= rocf else "反"
            use = max(roc, rocf)
            polarity[key].append((cell["name"], roc - 0.5))
            mark = "  ←" if use >= 0.65 else ""
            print(
                f"    {zh:16} 停对{len(pos)}/停错{len(neg)}  "
                f"中位 {dyn.med(pos):7.3f}/{dyn.med(neg):7.3f}  "
                f"{roc:.3f} 反{rocf:.3f} ({better}){mark}",
                flush=True,
            )

    print("\n跨集同向（一题一票，差=AUROC-0.5，同号才算同向）", flush=True)
    for key, zh in keys:
        xs = polarity[key]
        if len(xs) < 4:
            continue
        signs = [1 if d > 0 else -1 if d < 0 else 0 for _, d in xs]
        if all(s == signs[0] and s != 0 for s in signs):
            print(f"  {zh}: 五集同向  " + " ".join(f"{n}:{d:+.2f}" for n, d in xs), flush=True)
        else:
            print(f"  {zh}: 不同向", flush=True)


if __name__ == "__main__":
    main()
