#!/usr/bin/env python3
"""All extracted layer pairs + all low-conf windows, plus 8B vs PUMA gap."""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_acctok as acc  # noqa: E402
import analyze_lag_layers as layers  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402

EXTRA = {
    "name": "8B-MATH",
    "dataset": "math-500",
    "trial": AE / "results/dense_G_nemotron_8b/math-500/dense_puma/trial_answers.json",
    "stat": AE / "results/puma_offline_nemotron_8b/math-500/statistics.json",
    "gpath": AE / "results/dense_G_nemotron_8b/math-500/per_sample.json",
    "scores": (
        AE / "results/confcal_judge/v2/dense_internal/nemotron_8b/math-500",
        AE / "results/confcal_judge/v2/dense_lens/nemotron_8b/math-500",
    ),
    "preferred": "late_rise",
}

EXTRA_SIGS = (
    "stop_margin",
    "neg_ans_entropy",
    "dola_score",
    "emerge_layer",
    "layer_agree",
    "dola_mean_rise",
)


def discover_layers(scores: dict) -> list[int]:
    found: set[int] = set()
    for row in scores.values():
        for key in row:
            if key.startswith("dola_logp_l"):
                found.add(int(key[len("dola_logp_l") :]))
            elif key.startswith("lens_l") and key.count("_") == 1:
                try:
                    found.add(int(key[len("lens_l") :]))
                except ValueError:
                    continue
    return sorted(found)


def enrich_all(scores: dict) -> dict:
    ids = discover_layers(scores)
    for row in scores.values():
        last = rg.finite(row.get("last_mean_logp"))
        dola = {L: rg.finite(row.get(f"dola_logp_l{L}")) for L in ids}
        mean = {L: rg.finite(row.get(f"dola_mean_l{L}")) for L in ids}
        lens = {L: rg.finite(row.get(f"lens_l{L}")) for L in ids}
        for L in ids:
            if math.isfinite(last) and math.isfinite(lens[L]):
                row[f"exit_minus_l{L}"] = last - lens[L]
        for i, lo in enumerate(ids):
            for hi in ids[i + 1 :]:
                if math.isfinite(dola[lo]) and math.isfinite(dola[hi]):
                    row[f"dola_logp_{lo}_{hi}"] = dola[hi] - dola[lo]
                if math.isfinite(mean[lo]) and math.isfinite(mean[hi]):
                    row[f"dola_mean_{lo}_{hi}"] = mean[hi] - mean[lo]
                if math.isfinite(lens[lo]) and math.isfinite(lens[hi]):
                    row[f"lens_{lo}_{hi}"] = lens[hi] - lens[lo]
    return scores


def pair_names(ids: list[int]) -> list[str]:
    out = ["last_mean_logp", *EXTRA_SIGS]
    for L in ids:
        out += [f"dola_logp_l{L}", f"dola_mean_l{L}", f"lens_l{L}", f"exit_minus_l{L}"]
    for i, lo in enumerate(ids):
        for hi in ids[i + 1 :]:
            out += [f"dola_logp_{lo}_{hi}", f"dola_mean_{lo}_{hi}", f"lens_{lo}_{hi}"]
    return list(dict.fromkeys(out))


def med(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[len(xs) // 2]


def collect_windows(pack: dict[str, Any], names: list[str]) -> dict[str, dict[str, list[float]]]:
    bins: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    first: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in pack["questions"]:
        seen = set()
        for ev in q["events"]:
            tag = ev["tag"]
            if tag not in {"G", "R", "W", "L"}:
                continue
            for name in names:
                vals = ev["vals"].get(name) or []
                last = vals[-1] if vals else float("nan")
                if last == last:
                    bins[name][tag].append(last)
                    if tag not in seen and tag in {"R", "L"}:
                        first[name][tag].append(last)
            if tag in {"R", "L"}:
                seen.add(tag)
    return {"all": bins, "first": first}


def auc(pos: list[float], neg: list[float]) -> float:
    return rg.auroc(pos, neg)


def sweep_best(pack, sig, conf_rows, conf_sum):
    xs = []
    for ev in pack["questions"][0]["events"]:
        break
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["tag"] in {"R", "L"}:
                vals = ev["vals"].get(sig) or []
                if vals and vals[-1] == vals[-1]:
                    xs.append(vals[-1])
    points = []
    for thr in rg.quantiles(xs):
        rows = layers.run_pack(pack, sig, thr, "all")
        rec = layers.contrast(conf_rows, rows, conf_sum)
        rec["threshold"] = thr
        points.append(rec)
    return acc.pick_acc(points)


def gap_8b(pack) -> None:
    n = len(pack["questions"])
    conf_rows = layers.run_pack(pack, None, float("inf"), "all")
    both_ok = puma_only = conf_only = both_bad = 0
    early_wrong = late_miss = 0
    for q, row in zip(pack["questions"], conf_rows):
        puma = q["puma_ok"]
        ours = row["ok"]
        if puma and ours:
            both_ok += 1
        elif puma and not ours:
            puma_only += 1
            if row["early"] and row["branch"] == "conf":
                early_wrong += 1
            else:
                late_miss += 1
        elif (not puma) and ours:
            conf_only += 1
        else:
            both_bad += 1
    print(
        f"\n8B MATH 只看置信度 vs 官方 PUMA（{n} 题）\n"
        f"  两边都对 {both_ok}  只 PUMA 对 {puma_only}  只置信度对 {conf_only}  两边都错 {both_bad}\n"
        f"  只 PUMA 对的 {puma_only} 题里：高置信门提前交了错试答 {early_wrong}，"
        f"走到最后仍不对 {late_miss}\n"
        f"  净差 {(puma_only - conf_only)/n*100:.1f}pp  "
        f"（官方 PUMA 含改写终答；这边交的是试答本身）"
    )


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: enrich_all(orig(scores))
    cells = list(rg.CELLS) + [EXTRA]
    for cell in cells:
        print(f"\n======== {cell['name']} ========", flush=True)
        pack = layers.load_pack(cell)
        ids = discover_layers(pack["scores"])
        names = pair_names(ids)
        print(f"抽出的层：{ids}  信号 {len(names)} 条", flush=True)
        layers.precompute_events(pack, names)
        got = collect_windows(pack, names)
        conf_rows = layers.run_pack(pack, None, float("inf"), "all")
        conf_sum = rg.summarize(conf_rows)
        print(
            f"只看置信度 {rg.fmt_pct(conf_sum['acc'])} / {conf_sum['tok']:.0f}  "
            f"官方 PUMA {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f}",
            flush=True,
        )
        oracle = layers.run_pack  # placeholder
        # oracle R via rg.simulate path: reuse decide with a fake always-pass on R only
        n_r = sum(int("R" in q["labels"]) for q in pack["questions"])
        n_l = sum(int("L" in q["labels"]) for q in pack["questions"])
        n_g = sum(int("G" in q["labels"]) for q in pack["questions"])
        print(f"有过 G 的题 {n_g}  有过 R 的题 {n_r}  有过 L 的题 {n_l}")

        ranked = []
        for sig in names:
            a = got["all"][sig]
            f = got["first"][sig]
            a_all = auc(a["R"], a["L"])
            a_first = auc(f["R"], f["L"])
            if (len(a["R"]) + len(a["L"])) < 20 or a_all != a_all:
                continue
            ranked.append(
                (
                    a_all,
                    a_first,
                    sig,
                    {t: (len(a[t]), med(a[t])) for t in "GRLW"},
                    len(a["R"]),
                    len(a["L"]),
                    len(f["R"]),
                    len(f["L"]),
                )
            )
        ranked.sort(key=lambda x: -abs(x[0] - 0.5))
        print("\n全部低置信窗口上，对/错分得最开的层对（按 |AUROC-0.5|）")
        print("  读数 | 全窗口分开 | 第一次低置信分开 | G/R/L/W 中位数")
        for a_all, a_first, sig, stats, nr, nl, nfr, nfl in ranked[:12]:
            g, r, l, w = (stats[t][1] for t in "GRLW")
            print(
                f"  {sig:24s}  全{a_all:.2f}（{nr}/{nl}）  首{a_first:.2f}（{nfr}/{nfl}）  "
                f"G {g:.2f}  R {r:.2f}  L {l:.2f}  W {w:.2f}"
            )

        print("\n同集 Acc 最好的门（不要求伤 0）")
        bests = []
        for a_all, a_first, sig, stats, nr, nl, nfr, nfl in ranked:
            if abs(a_all - 0.5) < 0.04:
                continue
            best = sweep_best(pack, sig, conf_rows, conf_sum)
            if not best:
                continue
            bests.append((best["acc"], -best["tok"], sig, a_all, best))
        bests.sort(reverse=True)
        for _, __, sig, a_all, b in bests[:8]:
            print(
                f"  {sig:24s}  分开{a_all:.2f}  {rg.fmt_pct(b['acc'])} / {b['tok']:.0f}  "
                f"相对只看置信度 {rg.fmt_pp(b['d_acc_pp_conf'])} / {rg.fmt_tok(b['d_tok_conf'])}  "
                f"放进 {b['rescue_R']}/{b['rescue_L']} 伤 {b['damaged']}"
            )
        if cell["name"] == "8B-MATH":
            gap_8b(pack)


if __name__ == "__main__":
    main()
