#!/usr/bin/env python3
"""Sweep k and geo-conf threshold + last-minus-half lag. Same-set Acc-flat first."""
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
KS = (2, 3, 4, 5)
TAUS = (0.95, 0.97, 0.98, 0.99)


def pick_flat(points, conf_acc):
    keep = [p for p in points if p["acc"] + 1e-12 >= conf_acc]
    if not keep:
        return None
    return sorted(keep, key=lambda p: (p["tok"], -p["acc"], -p.get("rescue_R", 0)))[0]


def sweep_lag(pack, conf_rows, conf_sum):
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            vals = ev["vals"].get(SIG) or []
            if vals and vals[-1] == vals[-1]:
                xs.append(vals[-1])
    points = []
    for need in ("all", "last"):
        for thr in rg.quantiles(xs, n=17):
            rec = layers.contrast(conf_rows, layers.run_pack(pack, SIG, thr, need), conf_sum)
            rec.update(threshold=thr, need=need)
            points.append(rec)
    return acc.pick_acc(points), pick_flat(points, conf_sum["acc"])


def beat(acc_v, tok, puma_acc, puma_tok):
    return acc_v + 1e-12 >= puma_acc and tok <= puma_tok + 1e-6


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    packs = {}
    for cell in list(rg.CELLS) + [more.EXTRA]:
        print(f"load {cell['name']}", flush=True)
        pack = layers.load_pack(cell)
        cons.attach_shared(pack)
        packs[cell["name"]] = pack

    print(
        f"\n高置信度：k 次同一猜测，第一次≥门槛，后面≥第一次−{rg.EPS}，前{rg.MSS}步不许停。"
        f"滞后：最后一层减一半深，各集按正确率不降再少 token 选门槛（不冻）。\n",
        flush=True,
    )
    header = f"{'集':<8} {'k':>2} {'门槛':>5}  {'只看置信度':>16}  {'+滞后(正确率不降)':>20}  {'对PUMA':>16}  过线"
    print(header, flush=True)
    wins = []
    for k in KS:
        for tau in TAUS:
            rg.K = k
            rg.TAU = tau
            for name, pack in packs.items():
                layers.precompute_events(pack, [SIG])
                conf_rows = layers.run_pack(pack, None, float("inf"), "all")
                conf_sum = rg.summarize(conf_rows)
                best, flat = sweep_lag(pack, conf_rows, conf_sum)
                use = flat or best
                flag = "是" if beat(use["acc"], use["tok"], pack["puma_acc"], pack["puma_tok"]) else "否"
                if flag == "是":
                    wins.append((name, k, tau, use, pack["puma_acc"], pack["puma_tok"]))
                print(
                    f"{name:<8} {k:2d} {tau:5.2f}  "
                    f"{rg.fmt_pct(conf_sum['acc'])} / {conf_sum['tok']:.0f}  "
                    f"{rg.fmt_pct(use['acc'])} / {use['tok']:.0f} {use['need']}@{use['threshold']:.2f}  "
                    f"{rg.fmt_pp(100*(use['acc']-pack['puma_acc']))} / {rg.fmt_tok(use['tok']-pack['puma_tok'])}  "
                    f"{flag}",
                    flush=True,
                )

    print("\n相对 PUMA：正确率不低、token 不多 的 (k, 门槛)：", flush=True)
    if not wins:
        print("  没有一档四格都过。下面按集列出能过的。", flush=True)
    by = {}
    for name, k, tau, use, pa, pt in wins:
        by.setdefault(name, []).append((k, tau, use, pa, pt))
    for name, pack in packs.items():
        rows = by.get(name, [])
        print(f"  {name} PUMA {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f}  过线 {len(rows)} 档", flush=True)
        for k, tau, use, pa, pt in rows[:8]:
            print(
                f"    k={k} 门槛{tau:.2f}  {rg.fmt_pct(use['acc'])} / {use['tok']:.0f}  "
                f"{rg.fmt_pp(100*(use['acc']-pa))} / {rg.fmt_tok(use['tok']-pt)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
