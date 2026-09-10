#!/usr/bin/env python3
"""Shared-gate screen: other layer pairs / multi-layer relative scores vs default."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import screen_puma_layers as sc
import screen_relative_layer as rel
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/screen_shared_rel_layers.md"
EPS = 1e-6


def last(vals: list[float] | None) -> float:
    if not vals:
        return float("nan")
    return vals[-1] if vals[-1] == vals[-1] else float("nan")


def lens(ev: dict[str, Any], layer: int) -> float:
    return last(ev["vals"].get(f"lens_l{layer}"))


def dola(ev: dict[str, Any], layer: int) -> float:
    return last(ev["vals"].get(f"dola_mean_l{layer}"))


def ratio(deep: float, shallow: float) -> float:
    if deep != deep or shallow != shallow:
        return float("nan")
    return (deep - shallow) / (abs(deep) + EPS)


def mean_ok(xs: list[float]) -> float:
    good = [x for x in xs if x == x]
    return sum(good) / len(good) if good else float("nan")


def min_ok(xs: list[float]) -> float:
    good = [x for x in xs if x == x]
    return min(good) if good else float("nan")


def max_ok(xs: list[float]) -> float:
    good = [x for x in xs if x == x]
    return max(good) if good else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    rx = [sorted(xs).index(v) for v in xs]
    ry = [sorted(ys).index(v) for v in ys]
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx < EPS or deny < EPS:
        return float("nan")
    return num / (denx * deny)


def make_scores(pack: dict[str, Any]) -> dict[str, Callable[[dict[str, Any]], float]]:
    ids = pack["_ids"]
    q1 = sc.rel_layer(ids, 0.25)
    half = sc.rel_layer(ids, 0.50)
    q3 = sc.rel_layer(ids, 0.75)
    last_l = max(ids)
    earlier = [L for L in ids if L < last_l]

    def exit_s(ev):
        val = last(ev["vals"].get("last_mean_logp"))
        if val == val:
            return val
        return lens(ev, last_l)

    def vs_half(ev):
        return ratio(exit_s(ev), lens(ev, half))

    def vs_q1(ev):
        return ratio(exit_s(ev), lens(ev, q1))

    def vs_q3(ev):
        return ratio(exit_s(ev), lens(ev, q3))

    def q3_vs_half(ev):
        return ratio(lens(ev, q3), lens(ev, half))

    def q3_vs_q1(ev):
        return ratio(lens(ev, q3), lens(ev, q1))

    def half_vs_q1(ev):
        return ratio(lens(ev, half), lens(ev, q1))

    def vs_mean_early(ev):
        return ratio(exit_s(ev), mean_ok([lens(ev, L) for L in earlier]))

    def vs_max_early(ev):
        return ratio(exit_s(ev), max_ok([lens(ev, L) for L in earlier]))

    def vs_min_early(ev):
        return ratio(exit_s(ev), min_ok([lens(ev, L) for L in earlier]))

    def vs_three(ev):
        return ratio(exit_s(ev), mean_ok([lens(ev, q1), lens(ev, half), lens(ev, q3)]))

    def both_q1_half(ev):
        return min_ok([vs_half(ev), vs_q1(ev)])

    def both_half_q3(ev):
        return min_ok([vs_half(ev), vs_q3(ev)])

    def mean_q1_half(ev):
        return mean_ok([vs_half(ev), vs_q1(ev)])

    def mean_three_rise(ev):
        return mean_ok([vs_q1(ev), vs_half(ev), vs_q3(ev)])

    def stack_mono(ev):
        return min_ok([ratio(lens(ev, q3), lens(ev, half)), vs_q3(ev)])

    def rise_frac(ev):
        vals = [lens(ev, L) for L in ids]
        pairs = [
            1.0 if vals[i + 1] == vals[i + 1] and vals[i] == vals[i] and vals[i + 1] > vals[i] else 0.0
            for i in range(len(ids) - 1)
            if vals[i + 1] == vals[i + 1] and vals[i] == vals[i]
        ]
        return sum(pairs) / len(pairs) if pairs else float("nan")

    def depth_corr(ev):
        pairs = [(L, lens(ev, L)) for L in ids if lens(ev, L) == lens(ev, L)]
        if len(pairs) < 3:
            return float("nan")
        return spearman([p[0] for p in pairs], [p[1] for p in pairs])

    def dola_vs_half(ev):
        return ratio(dola(ev, last_l), dola(ev, half))

    def rise_and_last_best(ev):
        rise = vs_half(ev)
        best = rel.rel_best(ev, ids, "lens_l")
        if rise != rise or best != best:
            return float("nan")
        return rise if best >= 1.0 - 1e-9 else float("-inf")

    def rise_and_mono(ev):
        rise = vs_half(ev)
        a, b, c = lens(ev, half), lens(ev, q3), exit_s(ev)
        if rise != rise or a != a or b != b or c != c:
            return float("nan")
        return rise if c >= b >= a else float("-inf")

    return {
        "出口相对一半深（默认）": vs_half,
        "出口相对四分之一深": vs_q1,
        "出口相对四分之三深": vs_q3,
        "四分之三相对一半深": q3_vs_half,
        "四分之三相对四分之一深": q3_vs_q1,
        "一半深相对四分之一深": half_vs_q1,
        "出口相对前半平均": vs_mean_early,
        "出口相对前半最亮": vs_max_early,
        "出口相对前半最暗": vs_min_early,
        "出口相对三截平均": vs_three,
        "两段都抬（对¼与½取更小）": both_q1_half,
        "后两段都抬（对½与¾取更小）": both_half_q3,
        "两段抬升平均（¼与½）": mean_q1_half,
        "三段抬升平均": mean_three_rise,
        "一层层抬上来（¾−½ 与 末−¾ 取更小）": stack_mono,
        "越深越好写的层数占比": rise_frac,
        "层深和好写同向（Spearman）": depth_corr,
        "DoLA 出口相对一半深": dola_vs_half,
        "相对一半深 且 最好写在最后一层": rise_and_last_best,
        "相对一半深 且 后半一层层亮": rise_and_mono,
    }


def attach(pack: dict[str, Any], fns: dict[str, Callable[[dict[str, Any]], float]]) -> None:
    for q in pack["questions"]:
        for ev in q["_act"]:
            ev["rel"] = {name: fn(ev) for name, fn in fns.items()}


def cache(pack: dict[str, Any], name: str) -> list[list[tuple[float, dict[str, Any]]]]:
    out = []
    for q in pack["questions"]:
        rows = []
        for ev in q["_act"]:
            score = ev["rel"].get(name)
            if score == score:
                rows.append((score, ev))
        out.append(rows)
    return out


def eval_thr(pack: dict[str, Any], cached: list, thr: float) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = puma_acc = puma_tok = 0.0
    for q, rows in zip(pack["questions"], cached):
        puma_acc += int(q["puma_ok"])
        puma_tok += q["puma_tok"]
        hit = next((ev for score, ev in rows if score >= thr), None)
        if hit is None:
            acc += int(q["puma_ok"])
            tok += q["puma_tok"]
            continue
        sim = rg.pack(
            q["trials"], q["rows"], hit["end"], "rescue",
            original_tokens=q["orig_tok"], label=hit["tag"],
        )
        ok = rg.same(sim["answer"], q["gt"])
        acc += int(ok)
        tok += sim["tokens"]
    return {
        "acc": acc / n,
        "tok": tok / n,
        "d_acc": 100.0 * (acc / n - puma_acc / n),
        "d_tok": tok / n - puma_tok / n,
    }


def freeze(packs: list[tuple[str, dict[str, Any]]], cached: dict[str, Any], group: list[str]) -> tuple[float, dict[str, dict[str, Any]], float] | None:
    pack_map = dict(packs)
    pool = [s for n in group for rows in cached[n] for s, _ in rows if s != float("-inf")]
    if len(pool) < 8:
        return None
    best = None
    for thr in rg.quantiles(pool, n=41):
        recs = {n: eval_thr(pack_map[n], cached[n], thr) for n in group}
        mean = sum(r["d_acc"] for r in recs.values()) / len(recs)
        n_pos = sum(int(r["d_acc"] > 1e-9) for r in recs.values())
        n_neg = sum(int(r["d_acc"] < -1e-9) for r in recs.values())
        cand = (mean, n_pos, -n_neg, -sum(r["d_tok"] for r in recs.values()) / len(recs), thr, recs)
        if best is None or cand[:4] > best[:4]:
            best = cand
    assert best is not None
    mean, n_pos, n_neg, _, thr, recs = best
    return thr, recs, mean


def fmt_vs(rec: dict[str, Any]) -> str:
    return f"{rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])}"


def fmt_mean(recs: dict[str, dict[str, Any]]) -> str:
    mean = sum(r["d_acc"] for r in recs.values()) / len(recs)
    mean_tok = sum(r["d_tok"] for r in recs.values()) / len(recs)
    return f"{mean:+.1f} / {rg.fmt_tok(mean_tok)}"


def main() -> None:
    packs = []
    fns = None
    for name, pack in sc.load_ready():
        if fns is None:
            fns = make_scores(pack)
        attach(pack, make_scores(pack))
        packs.append((name, pack))
    assert fns is not None
    names = [n for n, _ in packs]
    pack_map = dict(packs)

    cached: dict[str, dict[str, Any]] = {sig: {} for sig in fns}
    for name, pack in packs:
        for sig in fns:
            cached[sig][name] = cache(pack, sig)

    default_recs = {
        n: eval_thr(pack_map[n], cached["出口相对一半深（默认）"][n], rg.REL_RISE_THR)
        for n in names
    }
    ranked = []
    for sig in fns:
        got = freeze(packs, cached[sig], names)
        if got is None:
            print(f"skip {sig}", flush=True)
            continue
        thr, recs, mean = got
        n_pos = sum(int(r["d_acc"] > 1e-9) for r in recs.values())
        n_neg = sum(int(r["d_acc"] < -1e-9) for r in recs.values())
        mean_tok = sum(r["d_tok"] for r in recs.values()) / len(recs)
        vs_def = mean - sum(r["d_acc"] for r in default_recs.values()) / len(default_recs)
        ranked.append((mean, n_pos, -n_neg, -mean_tok, sig, thr, recs, vs_def))
        print(
            f"{sig} thr={thr:.3f} {fmt_mean(recs)} win={n_pos}/{len(names)} hurt={n_neg} vs默认 {vs_def:+.1f}",
            flush=True,
        )
    ranked.sort(reverse=True)

    def_mean = sum(r["d_acc"] for r in default_recs.values()) / len(default_recs)
    def_tok = sum(r["d_tok"] for r in default_recs.values()) / len(default_recs)
    lines = [
        "# 共用闸：换层 / 多层相对量",
        "",
        "官方 PUMA 当底。只看停点前的低置信窗最后一步。正确率只对金标。",
        f"默认 = (末层好写 − 一半深好写) / |末层好写| ≥ {rg.REL_RISE_THR:.2f}，十集同一条。",
        "其余每条自己扫一个共用门槛（不是每集一只），再跟默认比。",
        "四分之一 / 一半 / 四分之三深按总层数对齐：7B 第 8/14/20 层，8B 第 8/16/24 层。",
        "",
        f"默认十集平均 **{def_mean:+.1f}pp / {rg.fmt_tok(def_tok)}**。",
        "",
        "## 各条共用闸（按全体平均正确率）",
        "",
        "| 分数 | 冻的门槛 | 平均 vs PUMA | 赢 | 伤 | 比默认 | 各集 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for mean, n_pos, n_neg, _, sig, thr, recs, vs_def in ranked:
        n_neg = -n_neg
        mark = " **默认**" if sig.startswith("出口相对一半深（默认）") else ""
        bits = ", ".join(f"{n} {fmt_vs(recs[n])}" for n in names)
        lines.append(
            f"| {sig}{mark} | {thr:.3f} | {fmt_mean(recs)} | "
            f"{n_pos}/{len(names)} | {n_neg}/{len(names)} | {vs_def:+.1f} | {bits} |"
        )

    better = [row for row in ranked if row[0] > def_mean + 1e-9 and -row[2] <= 1]
    lines += [
        "",
        "## 按集：默认 vs 扫出来更好的共用闸",
        "",
    ]
    show = [("默认（出口相对一半深 ≥ 2.85）", default_recs, rg.REL_RISE_THR)]
    extra = [row for row in ranked if row[4] != "出口相对一半深（默认）"][:4]
    for mean, n_pos, n_neg, _, sig, thr, recs, vs_def in extra:
        show.append((f"{sig}（≥{thr:.2f}）", recs, thr))
    header = "| 集 | PUMA | " + " | ".join(title for title, _, _ in show) + " |"
    lines += [header, "|---|" + "|".join(["---"] * (1 + len(show))) + "|"]
    for name, pack in packs:
        cells = [sw.fmt_pair(pack["puma_acc"], pack["puma_tok"])]
        for _, recs, _ in show:
            rec = recs[name]
            cells.append(f"{sw.fmt_pair(rec['acc'], rec['tok'])}（{fmt_vs(rec)}）")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    if better:
        names_b = "；".join(row[4] for row in better)
        lines += ["", f"伤不超过 1 集、平均正确率高于默认的：{names_b}。"]
    else:
        lines += ["", "没有一条在「伤不超过 1 集」的前提下，全体平均正确率高于默认。"]

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
