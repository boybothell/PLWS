#!/usr/bin/env python3
"""Snapshot: does the lag-door Acc/Tok gain show up on 7B Oly and 8B MATH?"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_acctok as acc  # noqa: E402
import analyze_lag_layers as layers  # noqa: E402
import replay_rescue_R_gate as rg  # noqa: E402

EXTRA = (
    {
        "name": "8B-MATH",
        "dataset": "math-500",
        "trial": AE / "results/dense_G_nemotron_8b/math-500/dense_puma/trial_answers.json",
        "stat": AE / "results/puma_offline_nemotron_8b/math-500/statistics.json",
        "gpath": AE / "results/dense_G_nemotron_8b/math-500/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/nemotron_8b/math-500",
            AE / "results/confcal_judge/v2/dense_lens/nemotron_8b/math-500",
        ),
        "preferred": "late_rise_rel",
    },
)

RELS = ("late_rise_rel", "exit_minus_rel_half", "exit_minus_rel_3q", "last_mean_logp")
ZH = {
    "late_rise_rel": "试答首词，3/4 深→出口抬了多少",
    "exit_minus_rel_half": "最后一层比一半深（整段）",
    "exit_minus_rel_3q": "最后一层比 3/4 深（整段）",
    "last_mean_logp": "最后一层，整段试答好写多少",
    "late_rise": "试答首词，20→27",
    "lens_rise": "最后一层比第 14 层",
    "exit_minus_l20": "最后一层比第 20 层",
    "margin": "当前试答比历史别的高",
    "reasoning_pmi": "有草稿比只看题更顺",
}


def add_rel(scores: dict) -> dict:
    for row in scores.values():
        dola, lens = {}, {}
        for key, value in row.items():
            if key.startswith("dola_logp_l"):
                dola[int(key[len("dola_logp_l") :])] = rg.finite(value)
            if key.startswith("lens_l") and key.count("_") == 1:
                try:
                    lens[int(key[len("lens_l") :])] = rg.finite(value)
                except ValueError:
                    continue
        ids = sorted(dola)
        last = rg.finite(row.get("last_mean_logp"))
        if len(ids) >= 2:
            hi, lo = ids[-1], ids[-2]
            if math.isfinite(dola[hi]) and math.isfinite(dola[lo]):
                row["late_rise_rel"] = dola[hi] - dola[lo]
            mid = ids[len(ids) // 2]
            if math.isfinite(last) and math.isfinite(lens.get(mid, float("nan"))):
                row["exit_minus_rel_half"] = last - lens[mid]
            if math.isfinite(last) and math.isfinite(lens.get(lo, float("nan"))):
                row["exit_minus_rel_3q"] = last - lens[lo]
    return scores


def window_mix(pack: dict[str, Any]) -> dict[str, int]:
    n = len(pack["questions"])
    has = {"G": 0, "R": 0, "W": 0, "L": 0}
    for q in pack["questions"]:
        tags = {ev["tag"] for ev in q["events"]}
        for name in has:
            if name in tags:
                has[name] += 1
    has["n"] = n
    return has


def sweep(pack: dict[str, Any], sig: str, conf_rows, conf_sum) -> dict[str, Any] | None:
    auc = layers.auroc_bins(pack["first_low"], sig)
    if auc["n_r"] + auc["n_l"] < 8 or auc["auroc"] != auc["auroc"]:
        return None
    xs = [rg.finite(r.get(sig)) for r in pack["first_low"]]
    points = []
    for thr in rg.quantiles(xs):
        rows = layers.run_pack(pack, sig, thr, "all")
        rec = layers.contrast(conf_rows, rows, conf_sum)
        rec["threshold"] = thr
        rec["d_acc_pp_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
        rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
        points.append(rec)
    best = acc.pick_acc(points)
    return {"auroc": auc, "best": best}


def main() -> None:
    orig = layers.enrich
    orig_sigs = layers.all_signals
    layers.enrich = lambda scores: add_rel(orig(scores))
    layers.all_signals = lambda: list(dict.fromkeys([*orig_sigs(), *RELS]))
    cells = list(rg.CELLS) + list(EXTRA)
    packs = {}
    for cell in cells:
        if not cell["trial"].is_file() or not cell["stat"].is_file():
            print(f"skip missing {cell['name']}", flush=True)
            continue
        if not any(Path(p).is_dir() for p in cell["scores"]):
            print(f"skip no scores {cell['name']}", flush=True)
            continue
        print(f"load {cell['name']}", flush=True)
        pack = layers.load_pack(cell)
        names = list(dict.fromkeys([*layers.all_signals(), *RELS, "late_rise", "lens_rise", "margin", "reasoning_pmi"]))
        layers.precompute_events(pack, names)
        packs[cell["name"]] = pack

    print("\n=== 题里有没有低置信度连对（R）===")
    for name, pack in packs.items():
        mix = window_mix(pack)
        n = mix["n"]
        print(
            f"{name}: {n} 题  高置信连对 {mix['G']/n:.0%}  "
            f"低置信连对且对 {mix['R']/n:.0%}  "
            f"高置信连错 {mix['W']/n:.0%}  "
            f"低置信连错 {mix['L']/n:.0%}"
        )

    print("\n=== 同集：相对深度门 vs 只看置信度 ===")
    report = {}
    for name, pack in packs.items():
        conf_rows = layers.run_pack(pack, None, float("inf"), "all")
        conf_sum = rg.summarize(conf_rows)
        print(
            f"{name} 只看置信度 {rg.fmt_pct(conf_sum['acc'])} / {conf_sum['tok']:.0f}  "
            f"PUMA {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f}"
        )
        item = {"conf": conf_sum, "puma_acc": pack["puma_acc"], "puma_tok": pack["puma_tok"], "sigs": {}}
        for sig in (*RELS, "late_rise", "exit_minus_l20", "lens_rise"):
            got = sweep(pack, sig, conf_rows, conf_sum)
            if not got or not got["best"]:
                continue
            b, auc = got["best"], got["auroc"]
            item["sigs"][sig] = got
            print(
                f"  {ZH[sig]}  分开 {auc['auroc']:.2f}（对{auc['n_r']}/错{auc['n_l']}）  "
                f"{rg.fmt_pct(b['acc'])} / {b['tok']:.0f}  "
                f"相对只看置信度 {rg.fmt_pp(b['d_acc_pp_conf'])} / {rg.fmt_tok(b['d_tok_conf'])}  "
                f"放进 {b['rescue_R']}/{b['rescue_L']} 伤 {b['damaged']}"
            )
        report[name] = {"pack": pack, "conf_rows": conf_rows, "conf_sum": conf_sum, **item}

    print("\n=== 换集：用相对深度、A 集最好门槛测 B ===")
    order = [n for n in ("MATH", "GPQA", "奥赛", "8B-MATH") if n in report]
    for sig in ("late_rise_rel", "exit_minus_rel_3q", "exit_minus_rel_half"):
        for src in order:
            got = report[src]["sigs"].get(sig)
            if not got:
                continue
            thr = float(got["best"]["threshold"])
            for tgt in order:
                if src == tgt:
                    continue
                rec = layers.contrast(
                    report[tgt]["conf_rows"],
                    layers.run_pack(report[tgt]["pack"], sig, thr, "all"),
                    report[tgt]["conf_sum"],
                )
                print(
                    f"  {ZH[sig]}  冻{src}→测{tgt}  {rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f}  "
                    f"相对只看置信度 {rg.fmt_pp(rec['d_acc_pp_conf'])} / {rg.fmt_tok(rec['d_tok_conf'])}  "
                    f"伤 {rec['damaged']}"
                )


if __name__ == "__main__":
    main()
