#!/usr/bin/env python3
"""Constructed metrics from existing dumps: within-question, G-relative, AND."""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_acctok as acc
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg

BASES = {
    "MATH": "lens_20_27",
    "GPQA": "exit_minus_l14",
    "奥赛": "lens_20_27",
    "8B-MATH": "dola_mean_l24",
}
SHARED = "late_shared"  # 3/4 → exit lens/dola-mean, same relative depth


def qtile(xs: list[float], p: float) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def iqr_z(x: float, mid: float, spread: float) -> float:
    if x != x or not math.isfinite(mid) or not math.isfinite(spread) or abs(spread) < 1e-6:
        return float("nan")
    return (x - mid) / spread


def ref_stats(xs: list[float]) -> tuple[float, float]:
    return qtile(xs, 0.5), qtile(xs, 0.75) - qtile(xs, 0.25)


def half_layer(ids: list[int]) -> int:
    """True half depth: 28-layer → 14, 32-layer → 16. Not the 2nd extracted id."""
    target = (max(ids) + 1) // 2
    return min(ids, key=lambda layer: (abs(layer - target), layer))


def attach_shared(pack: dict[str, Any]) -> str:
    ids = more.discover_layers(pack["scores"])
    if len(ids) < 2:
        pack["_half_layer"] = None
        return ""
    lo, hi = ids[-2], ids[-1]
    mid_id = half_layer(ids)
    pack["_half_layer"] = mid_id
    key = f"lens_{lo}_{hi}"
    alt = f"dola_mean_{lo}_{hi}"
    for row in pack["scores"].values():
        a = rg.finite(row.get(key))
        if a != a:
            a = rg.finite(row.get(alt))
        row[SHARED] = a
        last = rg.finite(row.get("last_mean_logp"))
        mid = rg.finite(row.get(f"lens_l{mid_id}"))
        row["exit_minus_half"] = last - mid if last == last and mid == mid else float("nan")
        lo_v = rg.finite(row.get(f"lens_l{lo}"))
        if a == a and lo_v == lo_v:
            row["norm_late"] = a / (abs(lo_v) + 1.0)
        else:
            row["norm_late"] = float("nan")
    return key


def add_question_deltas(pack: dict[str, Any], names: list[str]) -> None:
    for q in pack["questions"]:
        hist: dict[str, list[float]] = defaultdict(list)
        for ev in q["events"]:
            ev.setdefault("vals", {})
            for name in names:
                vals = ev["vals"].get(name) or []
                cur = vals[-1] if vals else float("nan")
                past = [x for x in hist[name] if x == x]
                if cur == cur and past:
                    ev["vals"].setdefault(f"qdelta_{name}", [cur - max(past)] * max(len(vals), 1))
                    ev["vals"].setdefault(
                        f"qrank_{name}",
                        [sum(cur >= x for x in past) / len(past)] * max(len(vals), 1),
                    )
                else:
                    ev["vals"].setdefault(f"qdelta_{name}", [float("nan")] * max(len(vals), 1))
                    ev["vals"].setdefault(f"qrank_{name}", [float("nan")] * max(len(vals), 1))
                if cur == cur:
                    hist[name].append(cur)


def add_combo(pack: dict[str, Any], stats: dict[str, tuple[float, float]]) -> None:
    parts = (SHARED, "stop_margin", "neg_ans_entropy")
    for q in pack["questions"]:
        for ev in q["events"]:
            zs = []
            width = 1
            for name in parts:
                vals = ev["vals"].get(name) or []
                width = max(width, len(vals))
                cur = vals[-1] if vals else float("nan")
                mid, spread = stats[name]
                zs.append(iqr_z(cur, mid, spread))
            combo = min(zs) if all(z == z for z in zs) else float("nan")
            ev["vals"]["and_min_z"] = [combo] * width
            late = (ev["vals"].get(SHARED) or [float("nan")])[-1]
            sm = (ev["vals"].get("stop_margin") or [float("nan")])[-1]
            ev["vals"]["z_late_mathg"] = [iqr_z(late, *stats[SHARED])] * width
            ev["vals"]["z_stop_mathg"] = [iqr_z(sm, *stats["stop_margin"])] * width


def g_refs(pack: dict[str, Any], names: list[str]) -> dict[str, list[float]]:
    out = defaultdict(list)
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["tag"] != "G":
                continue
            for name in names:
                vals = ev["vals"].get(name) or []
                if vals and vals[-1] == vals[-1]:
                    out[name].append(vals[-1])
    return out


def auc_tag(pack: dict[str, Any], sig: str) -> tuple[float, int, int]:
    pos, neg = [], []
    for q in pack["questions"]:
        for ev in q["events"]:
            vals = ev["vals"].get(sig) or []
            if not vals or vals[-1] != vals[-1]:
                continue
            if ev["tag"] == "R":
                pos.append(vals[-1])
            elif ev["tag"] == "L":
                neg.append(vals[-1])
    return rg.auroc(pos, neg), len(pos), len(neg)


def eligible(pack: dict[str, Any]) -> tuple[set[int], set[int]]:
    can_gain, can_hurt = set(), set()
    for q in pack["questions"]:
        labs = set(q["labels"])
        if "R" in labs and "G" not in labs and not q["orig_ok"]:
            can_gain.add(q["qi"])
        if "L" in q["labels"] and q["orig_ok"]:
            can_hurt.add(q["qi"])
    return can_gain, can_hurt


def sweep(pack, sig, conf_rows, conf_sum, need="all"):
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["tag"] in {"R", "L"}:
                vals = ev["vals"].get(sig) or []
                if vals and vals[-1] == vals[-1]:
                    xs.append(vals[-1])
    points = []
    for thr in rg.quantiles(xs, n=17):
        rec = layers.contrast(conf_rows, layers.run_pack(pack, sig, thr, need), conf_sum)
        rec.update(signal=sig, threshold=thr, need=need)
        points.append(rec)
    return acc.pick_acc(points), acc.pick_safe(points, conf_sum["acc"]), points


def freeze_apply(src_best, tgt_pack, tgt_conf, tgt_sum, sig):
    if not src_best:
        return None
    rec = layers.contrast(
        tgt_conf, layers.run_pack(tgt_pack, sig, float(src_best["threshold"]), src_best.get("need") or "all"), tgt_sum
    )
    rec.update(signal=sig, threshold=src_best["threshold"])
    return rec


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    cells = {c["name"]: c for c in list(rg.CELLS) + [more.EXTRA]}
    packs = {}
    for name in ("MATH", "GPQA", "奥赛", "8B-MATH"):
        print(f"load {name}", flush=True)
        pack = layers.load_pack(cells[name])
        attach_shared(pack)
        names = list(
            dict.fromkeys(
                [
                    SHARED,
                    "norm_late",
                    "exit_minus_half",
                    "stop_margin",
                    "neg_ans_entropy",
                    "last_mean_logp",
                    BASES[name],
                ]
            )
        )
        layers.precompute_events(pack, names)
        add_question_deltas(pack, [SHARED, "stop_margin"])
        packs[name] = pack

    ref = g_refs(packs["MATH"], [SHARED, "stop_margin", "neg_ans_entropy"])
    stats = {k: ref_stats(v) for k, v in ref.items()}
    print(f"MATH 高置信对参照 n={len(ref[SHARED])} stats={stats}", flush=True)
    for pack in packs.values():
        add_combo(pack, stats)
        # also need and_min_z / z_* in decide: already on events

    constructed = [
        SHARED,
        "norm_late",
        "z_late_mathg",
        "qdelta_" + SHARED,
        "qrank_" + SHARED,
        "and_min_z",
        "stop_margin",
        "z_stop_mathg",
        "qdelta_stop_margin",
    ]

    print("\n== 全窗口对/错分开（越大越好，0.5 是乱猜）==")
    print("读数 | MATH | GPQA | 奥赛 | 8B-MATH")
    for sig in constructed:
        bits = []
        for name, pack in packs.items():
            a, nr, nl = auc_tag(pack, sig)
            bits.append(f"{a:.2f}({nr}/{nl})" if a == a else "—")
        print(f"  {sig:24s} " + "  ".join(bits))

    print("\n== 奥赛：能加正确率的题 vs 开门会伤的题，读数中位 ==")
    pack = packs["奥赛"]
    gain, hurt = eligible(pack)
    print(f"能加正确率 {len(gain)} 题（只有低置信连对，写完反而不对）")
    print(f"开门可能伤 {len(hurt)} 题（有低置信连错，而且写完是对的）")
    for sig in constructed:
        gxs, hxs = [], []
        for q in pack["questions"]:
            best = float("-inf")
            found = False
            for ev in q["events"]:
                if ev["tag"] not in {"R", "L"}:
                    continue
                vals = ev["vals"].get(sig) or []
                if vals and vals[-1] == vals[-1]:
                    best = max(best, vals[-1])
                    found = True
            if not found:
                continue
            if q["qi"] in gain:
                gxs.append(best)
            if q["qi"] in hurt:
                hxs.append(best)
        if len(gxs) >= 8 and len(hxs) >= 8:
            print(
                f"  {sig:24s}  能加分题中位 {qtile(gxs,0.5):.3f}  会伤的题中位 {qtile(hxs,0.5):.3f}  "
                f"分开 {rg.auroc(gxs, hxs):.2f}（{len(gxs)}/{len(hxs)}）"
            )

    print("\n== 同集 Acc 最好（正确率优先）==")
    store = {}
    for name, pack in packs.items():
        conf_rows = layers.run_pack(pack, None, float("inf"), "all")
        conf_sum = rg.summarize(conf_rows)
        store[name] = {"pack": pack, "conf_rows": conf_rows, "conf_sum": conf_sum, "best": {}}
        print(f"{name} 只看置信度 {rg.fmt_pct(conf_sum['acc'])} / {conf_sum['tok']:.0f}")
        for sig in constructed:
            best, safe, _ = sweep(pack, sig, conf_rows, conf_sum, "all")
            if not best:
                continue
            store[name]["best"][sig] = best
            print(
                f"  {sig:24s} {rg.fmt_pct(best['acc'])} / {best['tok']:.0f}  "
                f"{rg.fmt_pp(best['d_acc_pp_conf'])} / {rg.fmt_tok(best['d_tok_conf'])}  "
                f"放进 {best['rescue_R']}/{best['rescue_L']} 伤 {best['damaged']}"
            )

    print("\n== 冻在 MATH 的 Acc 最好门槛，原样测别的集/模型 ==")
    for sig in constructed:
        src = store["MATH"]["best"].get(sig)
        if not src:
            continue
        print(f"{sig}  冻 MATH @{src['threshold']:.3f}  本集 {rg.fmt_pp(src['d_acc_pp_conf'])}")
        for tgt in ("GPQA", "奥赛", "8B-MATH"):
            rec = freeze_apply(src, store[tgt]["pack"], store[tgt]["conf_rows"], store[tgt]["conf_sum"], sig)
            if not rec:
                continue
            print(
                f"  → {tgt}  {rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f}  "
                f"{rg.fmt_pp(rec['d_acc_pp_conf'])} / {rg.fmt_tok(rec['d_tok_conf'])}  伤 {rec['damaged']}"
            )


if __name__ == "__main__":
    main()
