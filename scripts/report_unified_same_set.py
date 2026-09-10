#!/usr/bin/env python3
"""Same-set best for last-minus-half. High door = PUMA-style ε, k=4, mss=10. No freeze."""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_acctok as acc
import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg

SIG = "exit_minus_half"


def pick_flat(points, conf_acc, conf_tok=None):
    """Acc first among Acc-flat and tok-not-worse. Later-window stops count."""
    if conf_tok is None:
        keep = [p for p in points if p["acc"] + 1e-12 >= conf_acc]
        if not keep:
            return None
        return sorted(keep, key=lambda p: (p["tok"], -p["acc"]))[0]
    both = [
        p for p in points if p["acc"] + 1e-12 >= conf_acc and p["tok"] <= conf_tok + 1e-6
    ]
    if both:
        return sorted(both, key=lambda p: (-p["acc"], p["tok"]))[0]
    keep = [p for p in points if p["acc"] + 1e-12 >= conf_acc]
    if not keep:
        return None
    return sorted(keep, key=lambda p: (p["tok"], -p["acc"]))[0]


def sweep(pack, conf_rows, conf_sum):
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            vals = ev["vals"].get(SIG) or []
            if vals and vals[-1] == vals[-1]:
                xs.append(vals[-1])
    points = []
    for thr in rg.quantiles(xs, n=17):
        rec = layers.contrast(conf_rows, layers.run_pack(pack, SIG, thr, "all"), conf_sum)
        rec["threshold"] = thr
        points.append(rec)
    return acc.pick_acc(points), pick_flat(points, conf_sum["acc"])


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    print(
        f"现行默认：高置信度门 k={rg.K} 第一次≥{rg.TAU} 后面≥第一次−{rg.EPS} 前{rg.MSS}步不许停；"
        f"默认加上后路强停 FS；滞后=最后一层减一半深。"
        f"选门槛按题：正确率不降且 token 不多时先取正确率最高；不要求每题第一扇低置信窗就停，理想第一扇、后面停也算。"
        f"{' FS已开。' if rg.USE_FS else ' FS关着。'}",
        flush=True,
    )
    for cell in list(rg.CELLS) + [more.EXTRA]:
        pack = layers.load_pack(cell)
        cons.attach_shared(pack)
        layers.precompute_events(pack, [SIG, "stop_margin", "late_rise", "lens_rise"])
        conf_rows = layers.run_pack(pack, None, float("inf"), "all")
        conf_sum = rg.summarize(conf_rows)
        best, flat = sweep(pack, conf_rows, conf_sum)
        print(f"\n======== {cell['name']}  {len(pack['questions'])}题 ========", flush=True)
        print(
            f"  官方 PUMA                 {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f}"
        )
        print(
            f"  高置信度+FS               {rg.fmt_pct(conf_sum['acc'])} / {conf_sum['tok']:.0f}  "
            f"vs PUMA {rg.fmt_pp(100*(conf_sum['acc']-pack['puma_acc']))} / {rg.fmt_tok(conf_sum['tok']-pack['puma_tok'])}"
        )
        if best:
            print(
                f"  最后一层减一半深 · 正确率优先  {rg.fmt_pct(best['acc'])} / {best['tok']:.0f}  "
                f"门槛{best['threshold']:.3f}  vs只看置信度 {rg.fmt_pp(best['d_acc_pp_conf'])} / {rg.fmt_tok(best['d_tok_conf'])}  "
                f"vs PUMA {rg.fmt_pp(100*(best['acc']-pack['puma_acc']))} / {rg.fmt_tok(best['tok']-pack['puma_tok'])}  "
                f"放进{best['rescue_R']}/{best['rescue_L']} 伤{best['damaged']}"
            )
        if flat:
            print(
                f"  最后一层减一半深 · 正确率不降  {rg.fmt_pct(flat['acc'])} / {flat['tok']:.0f}  "
                f"门槛{flat['threshold']:.3f}  vs只看置信度 {rg.fmt_pp(flat['d_acc_pp_conf'])} / {rg.fmt_tok(flat['d_tok_conf'])}  "
                f"vs PUMA {rg.fmt_pp(100*(flat['acc']-pack['puma_acc']))} / {rg.fmt_tok(flat['tok']-pack['puma_tok'])}  "
                f"放进{flat['rescue_R']}/{flat['rescue_L']} 伤{flat['damaged']}"
            )


if __name__ == "__main__":
    main()
