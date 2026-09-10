#!/usr/bin/env python3
"""Why median gap / AUROC does not equal Acc: room + full threshold sweep."""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_acctok as acc
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg

FOCUS = {
    "奥赛": ["lens_20_27", "dola_mean_20_27", "exit_minus_l20", "stop_margin", "dola_mean_rise"],
    "8B-MATH": ["dola_mean_l24", "dola_mean_8_24", "stop_margin", "dola_logp_8_24", "dola_mean_24_31"],
}


def qtile(xs, p):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def room(pack):
    n = len(pack["questions"])
    r_only = r_then_g = r_full_wrong = r_only_full_ok = 0
    for q in pack["questions"]:
        labs = set(q["labels"])
        if "R" not in labs:
            continue
        if "G" in labs:
            r_then_g += 1
        else:
            r_only += 1
            if q["orig_ok"]:
                r_only_full_ok += 1
            else:
                r_full_wrong += 1
    print(
        f"  {n} 题里：低置信连对后又出现高置信连对 {r_then_g}（早停只少 token，正确率不变）\n"
        f"  只有低置信连对、没有高置信连对 {r_only}："
        f"写完也对 {r_only_full_ok}（早停正确率不变），"
        f"写完反而不对 {r_full_wrong}（只有这些能加正确率）"
    )
    return r_full_wrong


def dist(pack, sig):
    bins = defaultdict(list)
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["tag"] not in {"G", "R", "L"}:
                continue
            vals = ev["vals"].get(sig) or []
            if vals and vals[-1] == vals[-1]:
                bins[ev["tag"]].append(vals[-1])
    print(f"  {sig} 四分位（25/50/75）")
    for tag, name in (("G", "高置信对"), ("R", "低置信对"), ("L", "低置信错")):
        xs = bins[tag]
        print(
            f"    {name}: n={len(xs)}  "
            f"{qtile(xs,0.25):.2f} / {qtile(xs,0.5):.2f} / {qtile(xs,0.75):.2f}"
        )
    # overlap: share of R below L median, L above R median
    r, l = bins["R"], bins["L"]
    if r and l:
        lm, rm = qtile(l, 0.5), qtile(r, 0.5)
        print(
            f"    低置信对里有 {sum(x<=lm for x in r)/len(r):.0%} 不超过错的中位；"
            f"错的里有 {sum(x>=rm for x in l)/len(l):.0%} 不低于对的中位"
        )


def sweep_sig(pack, sig, conf_rows, conf_sum, needs=("all", "last", "k3")):
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["tag"] in {"R", "L"}:
                vals = ev["vals"].get(sig) or []
                if vals and vals[-1] == vals[-1]:
                    xs.append(vals[-1])
    rows = []
    for need in needs:
        for thr in rg.quantiles(xs, n=33):
            rec = layers.contrast(
                conf_rows, layers.run_pack(pack, sig, thr, need), conf_sum
            )
            rec.update(signal=sig, need=need, threshold=thr)
            rows.append(rec)
    best = acc.pick_acc(rows)
    safe = acc.pick_safe(rows, conf_sum["acc"])
    return rows, best, safe


def show_curve(rows, need="all", k=8):
    sub = [r for r in rows if r["need"] == need]
    sub = sorted(sub, key=lambda r: r["threshold"], reverse=True)
    print(f"  门槛从高到低（{need}，摘正确率最好和开门较多的几档）")
    # unique-ish operating points
    picked = []
    for r in sub:
        key = (round(r["acc"], 4), round(r["tok"]), r["rescue_R"], r["rescue_L"])
        if not picked or key != (round(picked[-1]["acc"], 4), round(picked[-1]["tok"]), picked[-1]["rescue_R"], picked[-1]["rescue_L"]):
            picked.append(r)
    # always include first (almost closed) and acc-best
    acc_best = acc.pick_acc(sub)
    show = []
    for r in picked:
        if r["rescue_R"] + r["rescue_L"] == 0:
            continue
        show.append(r)
    # head, middle, tail + best
    idxs = {0, len(show)//4, len(show)//2, 3*len(show)//4, len(show)-1} if show else set()
    chosen = [show[i] for i in sorted(idxs) if 0 <= i < len(show)]
    if acc_best and acc_best not in chosen:
        chosen.append(acc_best)
        chosen.sort(key=lambda r: -r["threshold"])
    for r in chosen[:k]:
        print(
            f"    门槛 {r['threshold']:.3f}  {rg.fmt_pct(r['acc'])} / {r['tok']:.0f}  "
            f"{rg.fmt_pp(r['d_acc_pp_conf'])} / {rg.fmt_tok(r['d_tok_conf'])}  "
            f"放进 {r['rescue_R']}/{r['rescue_L']} 伤 {r['damaged']}"
        )


def main():
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = {c["name"]: c for c in list(rg.CELLS) + [more.EXTRA]}
    for name in ("奥赛", "8B-MATH"):
        print(f"\n======== {name} ========", flush=True)
        pack = layers.load_pack(cells[name])
        ids = more.discover_layers(pack["scores"])
        names = more.pair_names(ids)
        layers.precompute_events(pack, names)
        conf_rows = layers.run_pack(pack, None, float("inf"), "all")
        conf_sum = rg.summarize(conf_rows)
        print(f"只看置信度 {rg.fmt_pct(conf_sum['acc'])} / {conf_sum['tok']:.0f}")
        room(pack)
        print("\n中位数看着远，分布叠多少：")
        for sig in FOCUS[name]:
            dist(pack, sig)

        print("\n所有层对按全集正确率重选门槛（4 步都过 / 只看最后一步 / 3 步过，取最好）")
        board = []
        for sig in names:
            rows, best, safe = sweep_sig(pack, sig, conf_rows, conf_sum)
            if not best:
                continue
            board.append((best["acc"], -best["tok"], sig, best, safe, rows))
        board.sort(reverse=True)
        print("  读数 | Acc最好 | 门槛/窗口 | 相对只看置信度 | 放进对/错 | 伤")
        for acc_, _, sig, best, safe, rows in board[:10]:
            print(
                f"  {sig:22s}  {rg.fmt_pct(best['acc'])} / {best['tok']:.0f}  "
                f"{best['need']}@{best['threshold']:.3f}  "
                f"{rg.fmt_pp(best['d_acc_pp_conf'])} / {rg.fmt_tok(best['d_tok_conf'])}  "
                f"{best['rescue_R']}/{best['rescue_L']}  伤{best['damaged']}"
            )
        print("\n指定读数的门槛曲线（不是只报一个点）：")
        for sig in FOCUS[name][:3]:
            rows, best, safe = sweep_sig(pack, sig, conf_rows, conf_sum)
            print(f"\n  [{sig}] Acc最好 {rg.fmt_pct(best['acc'])} / {best['tok']:.0f} "
                  f"({best['need']}@{best['threshold']:.3f})")
            show_curve(rows, need=best["need"])


if __name__ == "__main__":
    main()
