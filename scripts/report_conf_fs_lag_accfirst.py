#!/usr/bin/env python3
"""High-conf + FS + last-minus-half. Acc first among Acc-flat and tok-not-worse."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg

SIG = "exit_minus_half"
SEEDS = (42, 0, 1, 123)
TABLE = AE / "tables/conf_fs_lag_accfirst.md"


def pick_acc_first_both(points: list[dict[str, Any]], base_acc: float, base_tok: float):
    both = [
        p
        for p in points
        if p["acc"] + 1e-12 >= base_acc and p["tok"] <= base_tok + 1e-6
    ]
    if both:
        return sorted(both, key=lambda p: (-p["acc"], p["tok"]))[0]
    flat = [p for p in points if p["acc"] + 1e-12 >= base_acc]
    if not flat:
        return None
    return sorted(flat, key=lambda p: (p["tok"], -p["acc"]))[0]


def aime_cell(ds: str, seed: int) -> dict[str, Any]:
    root = AE / f"results/dense_G_r1_7b/{ds}/seed_{seed}"
    if seed == 42:
        stat = AE / f"results/puma_offline_r1_7b/{ds}/statistics.json"
    else:
        stat = AE / f"results/puma_offline_r1_7b_s{seed}/{ds}/statistics.json"
    return {
        "name": f"{ds}-s{seed}",
        "dataset": ds,
        "trial": root / "dense_puma/trial_answers.json",
        "stat": stat,
        "gpath": root / "per_sample.json",
        "scores": (
            AE / f"results/confcal_judge/v2/dense_internal/r1_7b/{ds}_s{seed}",
            AE / f"results/confcal_judge/v2/dense_lens/r1_7b/{ds}_s{seed}",
        ),
    }


def load_ready(cell: dict[str, Any]) -> dict[str, Any] | None:
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
        return None
    pack = layers.load_pack(cell)
    cons.attach_shared(pack)
    layers.precompute_events(pack, [SIG])
    n_sig = 0
    n_low = 0
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            n_low += 1
            vals = ev["vals"].get(SIG) or []
            if vals and vals[-1] == vals[-1]:
                n_sig += 1
    pack["_cov"] = n_sig / max(n_low, 1)
    if pack["_cov"] < 0.5:
        return None
    return pack


def merge_packs(packs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    questions = []
    for i, pack in enumerate(packs):
        off = (i + 1) * 100000
        for q in pack["questions"]:
            nq = dict(q)
            nq["qi"] = int(q["qi"]) + off
            questions.append(nq)
    n = max(len(questions), 1)
    return {
        "cell": {"name": name},
        "questions": questions,
        "puma_acc": sum(int(q["puma_ok"]) for q in questions) / n,
        "puma_tok": sum(q["puma_tok"] for q in questions) / n,
        "_cov": min(p["_cov"] for p in packs),
    }


def low_xs(pack: dict[str, Any]) -> list[float]:
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            vals = ev["vals"].get(SIG) or []
            if vals and vals[-1] == vals[-1]:
                xs.append(vals[-1])
    return xs


def eval_thr(pack: dict[str, Any], fs_rows: list[dict[str, Any]], fs_sum: dict[str, Any], thr: float):
    ours = layers.run_pack(pack, SIG, thr, "all")
    rec = layers.contrast(fs_rows, ours, fs_sum)
    rec["threshold"] = thr
    rec["d_acc_pp_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
    rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
    rec["both_fs"] = rec["acc"] + 1e-12 >= fs_sum["acc"] and rec["tok"] <= fs_sum["tok"] + 1e-6
    rec["both_puma"] = rec["acc"] + 1e-12 >= pack["puma_acc"] and rec["tok"] <= pack["puma_tok"] + 1e-6
    return rec


def sweep(pack: dict[str, Any]):
    fs_rows = layers.run_pack(pack, None, float("inf"), "all")
    fs_sum = rg.summarize(fs_rows)
    xs = low_xs(pack)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = [eval_thr(pack, fs_rows, fs_sum, thr) for thr in thrs]
    chosen = pick_acc_first_both(points, fs_sum["acc"], fs_sum["tok"])
    return fs_rows, fs_sum, points, chosen


def line(name: str, rec: dict[str, Any], pack: dict[str, Any], fs_sum: dict[str, Any], note: str = "") -> str:
    thr = rec["threshold"]
    thr_s = "不开" if thr == float("inf") else f"{thr:.3f}"
    both = "过" if rec["both_puma"] else "否"
    extra = f" {note}" if note else ""
    return (
        f"| {name} | {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f} "
        f"| {rg.fmt_pct(fs_sum['acc'])} / {fs_sum['tok']:.0f} "
        f"| {thr_s} | {rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f} "
        f"| {rg.fmt_pp(rec['d_acc_pp_puma'])} / {rg.fmt_tok(rec['d_tok_puma'])} "
        f"| {rg.fmt_pp(rec['d_acc_pp_conf'])} / {rg.fmt_tok(rec['d_tok_conf'])} "
        f"| {both} |{extra}"
    )


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    print(
        "全部是高置信度 + FS + 最后一层减一半深。试答即终答。"
        "选门槛：正确率不降且 token 不多时，先取正确率最高，再少 token。",
        flush=True,
    )

    singles: list[tuple[str, dict[str, Any]]] = []
    for cell in list(rg.CELLS) + [more.EXTRA]:
        pack = load_ready(cell)
        if pack is None:
            print(f"skip {cell['name']}", flush=True)
            continue
        print(f"load {cell['name']} cov={pack['_cov']:.2f}", flush=True)
        singles.append((cell["name"], pack))

    aime_merged: list[tuple[str, dict[str, Any]]] = []
    for ds, zh in (("aime24", "7B AIME24"), ("aime25", "7B AIME25")):
        parts = []
        for seed in SEEDS:
            pack = load_ready(aime_cell(ds, seed))
            if pack is None:
                print(f"skip {ds}-s{seed}", flush=True)
                continue
            print(f"load {ds}-s{seed} cov={pack['_cov']:.2f}", flush=True)
            parts.append(pack)
        if len(parts) == 4:
            aime_merged.append((zh, merge_packs(parts, zh)))

    named = singles + aime_merged
    # 8B 奥赛没有 lens，滞后读数不够，不进这张表。

    prepared = []
    all_thrs: list[float] = []
    for name, pack in named:
        print(f"sweep {name}", flush=True)
        fs_rows, fs_sum, points, chosen = sweep(pack)
        prepared.append((name, pack, fs_rows, fs_sum, points, chosen))
        all_thrs.extend(p["threshold"] for p in points if p["threshold"] != float("inf"))

    # 全局：同一原始门槛，在尽量多的集上「相对高置信+FS 正确率不降且 token 不多」，再加权正确率。
    grid = sorted(set(round(x, 3) for x in all_thrs))
    if not grid:
        raise SystemExit("no thresholds")

    def score_global(thr: float):
        recs = []
        n = 0
        ok_w = 0.0
        tok_w = 0.0
        n_both_fs = 0
        n_both_puma = 0
        n_flat = 0
        for name, pack, fs_rows, fs_sum, _points, _chosen in prepared:
            rec = eval_thr(pack, fs_rows, fs_sum, thr)
            recs.append((name, rec))
            w = len(pack["questions"])
            n += w
            ok_w += rec["acc"] * w
            tok_w += rec["tok"] * w
            n_both_fs += int(rec["both_fs"])
            n_both_puma += int(rec["both_puma"])
            n_flat += int(rec["acc"] + 1e-12 >= fs_sum["acc"])
        return {
            "threshold": thr,
            "n_both_fs": n_both_fs,
            "n_both_puma": n_both_puma,
            "n_flat": n_flat,
            "acc": ok_w / max(n, 1),
            "tok": tok_w / max(n, 1),
            "recs": recs,
        }

    ranked = [score_global(thr) for thr in grid]
    ranked.sort(key=lambda r: (-r["n_both_fs"], -r["acc"], r["tok"]))
    frozen = ranked[0]

    print(
        f"\n全局冻门槛 {frozen['threshold']:.3f}  "
        f"相对高置信+FS 两边过线 {frozen['n_both_fs']}/{len(prepared)} 集  "
        f"相对 PUMA 两边过线 {frozen['n_both_puma']}/{len(prepared)} 集  "
        f"加权 {rg.fmt_pct(frozen['acc'])} / {frozen['tok']:.0f}",
        flush=True,
    )

    header = (
        "| 集 | 官方 PUMA | 高置信度+FS | 门槛 | 再加滞后 | 相对 PUMA | 相对高置信+FS | 相对 PUMA 两边过线 |"
    )
    sep = "|---|---|---|---|---|---|---|---|"

    per_lines = [header, sep]
    print("\n======== 每集自己扫（正确率优先，且 token 不多于高置信+FS） ========", flush=True)
    print(header)
    print(sep)
    for name, pack, _fs_rows, fs_sum, _points, chosen in prepared:
        if not chosen:
            print(f"| {name} | 选不出 |")
            continue
        row = line(name, chosen, pack, fs_sum)
        print(row)
        per_lines.append(row)

    fr_lines = [header, sep]
    print(f"\n======== 全局冻 {frozen['threshold']:.3f} ========", flush=True)
    print(header)
    print(sep)
    for name, rec in frozen["recs"]:
        pack = next(p for n, p, *_ in prepared if n == name)
        fs_sum = next(fs for n, _p, _r, fs, *_ in prepared if n == name)
        row = line(name, rec, pack, fs_sum)
        print(row)
        fr_lines.append(row)

    # 把每集门槛冻到别的集，看谁最像「全局」
    print("\n======== 用某一集自己的门槛去测全部（对照） ========", flush=True)
    print("| 冻在 | 门槛 | 两边过线(相对高置信+FS) | 加权正确率 / token |")
    print("|---|---|---|---|")
    xfer_lines = [
        "| 冻在 | 门槛 | 两边过线(相对高置信+FS) | 加权正确率 / token |",
        "|---|---|---|---|",
    ]
    for src_name, _pack, _fs_rows, _fs_sum, _points, chosen in prepared:
        if not chosen or chosen["threshold"] == float("inf"):
            continue
        got = score_global(chosen["threshold"])
        row = (
            f"| {src_name} | {chosen['threshold']:.3f} | {got['n_both_fs']}/{len(prepared)} "
            f"| {rg.fmt_pct(got['acc'])} / {got['tok']:.0f} |"
        )
        print(row)
        xfer_lines.append(row)

    md = "\n".join(
        [
            "# 高置信度 + FS + 滞后：正确率优先（两边过线时）",
            "",
            "交卷：试答即终答。三扇门都开：高置信度连答、FS 后路强停、最后一层减一半深。",
            "选门槛：相对「高置信度+FS」，正确率不降且 token 不多的点里，先取正确率最高，再少 token。",
            "不要求每题第一扇低置信窗就停；理想第一扇，后面停也算。",
            "AIME 是四个 seed 合成 120 题再平均。8B 奥赛还没有中间层 lens，没进这张表。",
            "",
            f"## 全局冻门槛 `{frozen['threshold']:.3f}`",
            "",
            f"在已跑完的 {len(prepared)} 集上，同一原始门槛："
            f"相对高置信+FS 两边过线 {frozen['n_both_fs']} 集，"
            f"相对 PUMA 两边过线 {frozen['n_both_puma']} 集。",
            "",
            *fr_lines,
            "",
            "## 每集自己扫出来的最好（对照，不当冻门槛）",
            "",
            *per_lines,
            "",
            "## 用某一集的门槛去测全部",
            "",
            *xfer_lines,
            "",
        ]
    )
    TABLE.write_text(md)
    print(f"\n写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
