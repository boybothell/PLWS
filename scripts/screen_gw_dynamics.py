#!/usr/bin/env python3
"""G vs W by *change*: layer-to-layer shape and Wait over the 4-step lock.

Not the old static min-of-4 threshold. First high-conf window only.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_14b_vs_puma as r14
import report_8b_extra_vs_puma as extra8
import screen_gw_veto as gw


def med(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


def series(ev, sig):
    return [v for v in (ev["vals"].get(sig) or []) if v == v]


def delta(xs):
    if len(xs) < 2:
        return float("nan")
    return xs[-1] - xs[0]


def slope(xs):
    if len(xs) < 2:
        return float("nan")
    # last half minus first half
    mid = len(xs) // 2
    a = statistics.mean(xs[: max(mid, 1)])
    b = statistics.mean(xs[mid:])
    return b - a


def feat_row(ev, ids, last_L, qL, hL, tL):
    def last_s(sig):
        xs = series(ev, sig)
        return xs[-1] if xs else float("nan")

    lens = [last_s(f"lens_l{L}") for L in ids]
    jumps = []
    for i in range(len(ids) - 1):
        a, b = lens[i], lens[i + 1]
        if a == a and b == b:
            jumps.append((ids[i + 1], b - a))
    if jumps:
        peak_L, peak_j = max(jumps, key=lambda x: x[1])
        peak_rel = peak_L / max(last_L, 1)
    else:
        peak_L, peak_j, peak_rel = float("nan"), float("nan"), float("nan")
    lq, lh, lt, ll = (last_s(f"lens_l{L}") for L in (qL, hL, tL, last_L))
    out = {
        "wait_d": delta(series(ev, "stop_margin")),
        "wait_s": slope(series(ev, "stop_margin")),
        "wait_last": last_s("stop_margin"),
        "stop_d": delta(series(ev, "stop_logp")),
        "wlog_d": delta(series(ev, "wait_logp")),
        "exit_d": delta(series(ev, "last_mean_logp")),
        "exit_s": slope(series(ev, "last_mean_logp")),
        "half_d": delta(series(ev, f"lens_l{hL}")),
        "lastL_d": delta(series(ev, f"lens_l{last_L}")),
        "rise_d": delta(series(ev, f"exit_minus_l{hL}")),
        "q_to_h": (lh - lq) if lh == lh and lq == lq else float("nan"),
        "h_to_t": (lt - lh) if lt == lt and lh == lh else float("nan"),
        "t_to_L": (ll - lt) if ll == ll and lt == lt else float("nan"),
        "h_to_L": (ll - lh) if ll == ll and lh == lh else float("nan"),
        "late_vs_mid": (
            (ll - lh) - (lh - lq)
            if all(x == x for x in (ll, lh, lq))
            else float("nan")
        ),
        "peak_rel": peak_rel,
        "peak_jump": peak_j,
        "lens_q": lq,
        "lens_h": lh,
        "lens_t": lt,
        "lens_L": ll,
    }
    return out


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = [rg.CELLS[0], rg.CELLS[2], extra8.CELLS[0], extra8.CELLS[1], r14.CELLS[0]]
    print(
        "高置信第一扇：看「怎么变」，不看旧的 4 步最差门槛。"
        "层间跳在哪、Wait/出口在 4 步里升还是降。",
        flush=True,
    )
    change_keys = [
        ("wait_d", "Wait 4步末−首"),
        ("wait_s", "Wait 后半−前半"),
        ("stop_d", "</think> 4步变化"),
        ("wlog_d", "Wait词 4步变化"),
        ("exit_d", "出口 4步变化"),
        ("half_d", "一半深 4步变化"),
        ("lastL_d", "最深层 4步变化"),
        ("rise_d", "出口减一半深 4步变化"),
        ("q_to_h", "层：¼→½"),
        ("h_to_t", "层：½→¾"),
        ("t_to_L", "层：¾→末"),
        ("h_to_L", "层：½→末"),
        ("late_vs_mid", "后半跳 − 前半跳"),
        ("peak_rel", "最大跳在多深（0浅1深）"),
        ("peak_jump", "最大一跳有多大"),
    ]
    for cell in cells:
        pack = gw.load(cell)
        if pack is None:
            continue
        ids = more.discover_layers(pack["scores"])
        if len(ids) < 4:
            print(f"skip {cell['name']} layers={ids}", flush=True)
            continue
        last_L = max(ids)
        n = last_L + 1
        qL = min(ids, key=lambda L: abs(L - n // 4))
        hL = min(ids, key=lambda L: abs(L - n // 2))
        tL = min(ids, key=lambda L: abs(L - (3 * n) // 4))
        names = (
            [f"lens_l{L}" for L in ids]
            + [f"exit_minus_l{hL}"]
            + ["stop_margin", "stop_logp", "wait_logp", "last_mean_logp"]
        )
        layers.precompute_events(pack, names)
        g, w = [], []
        for q in pack["questions"]:
            for ev in q["events"]:
                if not ev["high"] or ev["tag"] not in ("G", "W"):
                    continue
                if int(q["rows"][ev["end"]]["stopped_len"]) < rg.MSS:
                    continue
                rec = feat_row(ev, ids, last_L, qL, hL, tL)
                rec["tag"] = ev["tag"]
                (w if ev["tag"] == "W" else g).append(rec)
                break
        print(
            f"\n=== {cell['name']}  层 {qL}/{hL}/{tL}/{last_L}  G={len(g)} W={len(w)} ===",
            flush=True,
        )
        if len(w) < 5:
            print("  W 太少", flush=True)
            continue
        print("  层曲线（窗最后一步，中位）", flush=True)
        print(
            f"    {'':8} {'¼':>8} {'½':>8} {'¾':>8} {'末':>8}",
            flush=True,
        )
        for tag, rows in (("G", g), ("W", w)):
            print(
                f"    {tag:8} {med([r['lens_q'] for r in rows]):8.2f} "
                f"{med([r['lens_h'] for r in rows]):8.2f} "
                f"{med([r['lens_t'] for r in rows]):8.2f} "
                f"{med([r['lens_L'] for r in rows]):8.2f}",
                flush=True,
            )
        print("  变化量  G中位 / W中位  AUROC(G>W)  反过来", flush=True)
        ranked = []
        for key, zh in change_keys:
            gp = [r[key] for r in g]
            wp = [r[key] for r in w]
            if sum(x == x for x in gp) < 8 or sum(x == x for x in wp) < 5:
                continue
            roc = rg.auroc(gp, wp)
            rocf = rg.auroc([-x if x == x else x for x in gp], [-x if x == x else x for x in wp])
            ranked.append((max(roc, rocf), roc, rocf, key, zh, med(gp), med(wp)))
            mark = "  ←较好" if max(roc, rocf) >= 0.65 else ""
            print(
                f"    {zh:18} G {med(gp):7.3f}  W {med(wp):7.3f}  "
                f"{roc:.3f} / 反{rocf:.3f}{mark}",
                flush=True,
            )
        best = [r for r in ranked if r[0] >= 0.65]
        if not best:
            print("  没有 ≥0.65 的变化量（和猜差不远或只在这一集略好）", flush=True)


if __name__ == "__main__":
    main()
