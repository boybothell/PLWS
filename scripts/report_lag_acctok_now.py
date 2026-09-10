#!/usr/bin/env python3
"""Acc + token for current gates. Both wins count: Acc up, or Acc flat + fewer tokens."""
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

METHODS = (
    ("late_shared", "出口前一段到出口，试答好写了多少"),
    ("exit_minus_half", "最后一层比一半深"),
    ("stop_margin", "写完猜测后，更想收工还是再想想"),
    ("and_min_z", "上面这条、收工、用词集中，都像高置信连对"),
)


def pick_flat(points, conf_acc):
    keep = [p for p in points if p["acc"] + 1e-12 >= conf_acc]
    if not keep:
        return None
    return sorted(keep, key=lambda p: (p["tok"], -p["acc"]))[0]


def oracle_r(q):
    trials, rows = q["trials"], q["rows"]
    for ev in q["events"]:
        if ev["high"]:
            return rg.pack(trials, rows, ev["end"], "conf", original_tokens=q["orig_tok"], label=ev["tag"])
        if ev["mixed"]:
            continue
        if ev["tag"] == "R":
            return rg.pack(trials, rows, ev["end"], "rescue", original_tokens=q["orig_tok"], label="R")
    return rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=q["orig_tok"])


def sweep(pack, sig, conf_rows, conf_sum):
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["tag"] in {"R", "L"}:
                vals = ev["vals"].get(sig) or []
                if vals and vals[-1] == vals[-1]:
                    xs.append(vals[-1])
    points = []
    for thr in rg.quantiles(xs, n=17):
        rec = layers.contrast(conf_rows, layers.run_pack(pack, sig, thr, "all"), conf_sum)
        rec.update(threshold=thr)
        points.append(rec)
    return acc.pick_acc(points), pick_flat(points, conf_sum["acc"])


def line(name, acc_v, tok, d_conf=None, t_conf=None, d_puma=None, t_puma=None, extra=""):
    bit = f"{rg.fmt_pct(acc_v)} / {tok:.0f}"
    if d_conf is not None:
        bit += f"  vs只看置信度 {rg.fmt_pp(d_conf)} / {rg.fmt_tok(t_conf)}"
    if d_puma is not None:
        bit += f"  vs PUMA {rg.fmt_pp(d_puma)} / {rg.fmt_tok(t_puma)}"
    if extra:
        bit += f"  {extra}"
    print(f"  {name:<28} {bit}")


def main():
    orig = layers.enrich
    layers.enrich = lambda scores: cons.enrich_all(more.enrich_all(orig(scores)))  # type: ignore
    # enrich_all is on more
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = list(rg.CELLS) + [more.EXTRA]
    packs = {}
    for cell in cells:
        print(f"load {cell['name']}", flush=True)
        pack = layers.load_pack(cell)
        cons.attach_shared(pack)
        names = [cons.SHARED, "exit_minus_half", "stop_margin", "neg_ans_entropy", "late_rise", "lens_rise"]
        layers.precompute_events(pack, names)
        packs[cell["name"]] = pack

    ref = cons.g_refs(packs["MATH"], [cons.SHARED, "stop_margin", "neg_ans_entropy"])
    stats = {k: cons.ref_stats(v) for k, v in ref.items()}
    for pack in packs.values():
        cons.add_combo(pack, stats)

    print()
    print("成功两种都算：正确率升高；或正确率不降、token 变少。")
    print("「对」= 和写完一致，或和标准答案一致。交卷仍是当时那个猜测。")
    print("除「事后上界」外，都是线上只用当前窗读数。")
    print()
    for name, pack in packs.items():
        conf_rows = layers.run_pack(pack, None, float("inf"), "all")
        conf_sum = rg.summarize(conf_rows)
        print(f"======== {name}  {len(pack['questions'])}题 ========")
        line("官方 PUMA（含改写）", pack["puma_acc"], pack["puma_tok"])
        line(
            "只看置信度",
            conf_sum["acc"],
            conf_sum["tok"],
            d_puma=100.0 * (conf_sum["acc"] - pack["puma_acc"]),
            t_puma=conf_sum["tok"] - pack["puma_tok"],
        )

        frozen = rg.FROZEN.get(pack["cell"]["dataset"])
        if frozen:
            sig, thr = frozen
            rec = layers.contrast(conf_rows, layers.run_pack(pack, sig, thr, "all"), conf_sum)
            line(
                "现用滞后门（冻在7B）",
                rec["acc"],
                rec["tok"],
                rec["d_acc_pp_conf"],
                rec["d_tok_conf"],
                100.0 * (rec["acc"] - pack["puma_acc"]),
                rec["tok"] - pack["puma_tok"],
                f"{sig}@{thr:g}",
            )

        for sig, zh in METHODS:
            best, flat = sweep(pack, sig, conf_rows, conf_sum)
            if best:
                line(
                    zh + " · 正确率优先",
                    best["acc"],
                    best["tok"],
                    best["d_acc_pp_conf"],
                    best["d_tok_conf"],
                    100.0 * (best["acc"] - pack["puma_acc"]),
                    best["tok"] - pack["puma_tok"],
                )
            if flat and (not best or abs(flat["tok"] - best["tok"]) > 1 or abs(flat["acc"] - best["acc"]) > 1e-9):
                line(
                    zh + " · 正确率不降尽量少token",
                    flat["acc"],
                    flat["tok"],
                    flat["d_acc_pp_conf"],
                    flat["d_tok_conf"],
                    100.0 * (flat["acc"] - pack["puma_acc"]),
                    flat["tok"] - pack["puma_tok"],
                )

        o_rows = []
        for q in pack["questions"]:
            sim = oracle_r(q)
            o_rows.append(
                {
                    **sim,
                    "ok": rg.hit(sim["answer"], q["gt"], q["a_final"], q["orig_ok"]),
                    "has_r": "R" in q["labels"],
                    "r_only": "R" in q["labels"] and "G" not in q["labels"],
                }
            )
        o = layers.contrast(conf_rows, o_rows, conf_sum)
        line(
            "事后上界：低置信连对就停",
            o["acc"],
            o["tok"],
            o["d_acc_pp_conf"],
            o["d_tok_conf"],
            100.0 * (o["acc"] - pack["puma_acc"]),
            o["tok"] - pack["puma_tok"],
            f"伤{o['damaged']}",
        )
        print()


if __name__ == "__main__":
    main()
