#!/usr/bin/env python3
"""Aligned layer pairs + Wait variants on 7B/8B. Acc-first among both vs high-conf+FS."""
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
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp
import report_8b_extra_vs_puma as extra8

FRACS = (
    (0.25, "四分之一深"),
    (0.50, "一半深"),
    (0.75, "四分之三深"),
)
WAIT = (
    ("stop_margin", "收口比 Wait"),
    ("stop_vs_cont", "收口比 Wait 或 Alternatively"),
    ("stop_margin_alt", "收口比 Alternatively"),
)
TABLE = AE / "tables/screen_layer_wait.md"


def rel_layer(ids: list[int], frac: float) -> int:
    n = max(ids) + 1
    target = n - 1 if frac >= 1 else int(round((n - 1) * frac))
    return min(ids, key=lambda layer: (abs(layer - target), layer))


def polarity(pack: dict[str, Any], sig: str) -> tuple[float, int, int]:
    pos, neg = [], []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"] or ev["tag"] not in {"R", "L"}:
                continue
            vals = ev["vals"].get(sig) or []
            if not vals or any(v != v for v in vals):
                continue
            score = min(vals)
            (pos if ev["tag"] == "R" else neg).append(score)
            break
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    return rg.p50(pos) - rg.p50(neg), len(pos), len(neg)


def sweep(pack, fs_rows, fs_sum, sig: str, need: str = "all"):
    if cmp.cov(pack, sig) < 0.5:
        return None
    xs = cmp.low_xs(pack, sig)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = []
    for thr in thrs:
        ours = layers.run_pack(pack, sig, thr, need)
        rec = layers.contrast(fs_rows, ours, fs_sum)
        rec["threshold"] = thr
        rec["signal"] = sig
        rec["need"] = need
        rec["d_acc_pp_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
        rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
        rec["d_acc_pp_fs"] = 100.0 * (rec["acc"] - fs_sum["acc"])
        rec["d_tok_fs"] = rec["tok"] - fs_sum["tok"]
        rec["both_fs"] = rec["acc"] + 1e-12 >= fs_sum["acc"] and rec["tok"] <= fs_sum["tok"] + 1e-6
        points.append(rec)
    return accfirst.pick_acc_first_both(points, fs_sum["acc"], fs_sum["tok"])


def load_all() -> list[tuple[str, dict[str, Any]]]:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    named: list[tuple[str, dict[str, Any]]] = []
    for cell in list(rg.CELLS) + [more.EXTRA] + list(extra8.CELLS):
        seen = {n for n, _ in named}
        if cell["name"] in seen:
            continue
        pack = cmp.load_ready(cell)
        if pack is None:
            print(f"skip {cell['name']}", flush=True)
            continue
        print(f"load {cell['name']} layers={more.discover_layers(pack['scores'])[:6]}...", flush=True)
        named.append((cell["name"], pack))
    for ds, zh in (("aime24", "7B AIME24"), ("aime25", "7B AIME25")):
        parts = []
        for seed in accfirst.SEEDS:
            cell = accfirst.aime_cell(ds, seed)
            pack = cmp.load_ready(cell)
            if pack is None:
                continue
            pack["_cov"] = 1.0
            parts.append(pack)
        if len(parts) == 4:
            print(f"load {zh}", flush=True)
            named.append((zh, parts))
    return named


def fmt(rec) -> str:
    if rec is None:
        return "—"
    thr = rec["threshold"]
    thr_s = "不开" if thr == float("inf") else f"{thr:.2f}"
    return (
        f"{rg.fmt_pct(rec['acc'])}/{rec['tok']:.0f} "
        f"（{rg.fmt_pp(rec['d_acc_pp_puma'])}/{rg.fmt_tok(rec['d_tok_puma'])}；"
        f"{thr_s}；{rec['rescue_R']}/{rec['rescue_L']}）"
    )


def beats(a, b) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    if a["acc"] > b["acc"] + 1e-12:
        return True
    if abs(a["acc"] - b["acc"]) <= 1e-12 and a["tok"] + 1e-6 < b["tok"]:
        return True
    return False


def main() -> None:
    named = load_all()
    jobs: list[tuple[str, str, str, str]] = []
    # filled per pack after discovering layers
    results: dict[str, dict[str, Any]] = {}
    lines = [
        "# 7B / 8B：对齐层对 + Wait 变体",
        "",
        "选门槛：相对高置信+后路强停，正确率不降且 token 不多时先取正确率最高，再少 token。",
        "括号：相对 PUMA 的正确率/token；门槛；放进对/错锁。不开 = 滞后一扇都不放。",
        "层按相对深度对齐：7B 四分之一/一半/四分之三 ≈ 8/14/20，8B ≈ 8/16/24。",
        "分数 = 末层 boxed 平均 logp − 那一层 lens。Wait 变体只改读数或 4 步怎么过。",
        "",
    ]

    for name, pack in named:
        parts = pack if isinstance(pack, list) else None
        src = parts[0] if parts else pack
        ids = more.discover_layers(src["scores"])
        if len(ids) < 2:
            print(f"no layers {name}", flush=True)
            continue
        last = max(ids)
        extras = []
        names = ["last_mean_logp", "exit_minus_half"]
        layer_jobs = [("exit_minus_half", "all", f"末层 − 一半深（第 {src.get('_half_layer')} 层）", "layer")]
        for frac, zh in FRACS:
            mid = rel_layer(ids, frac)
            sig = f"exit_minus_l{mid}"
            names.append(sig)
            layer_jobs.append((sig, "all", f"末层 − {zh}（第 {mid} 层）", "layer"))
            rise = f"lens_{mid}_{last}"
            names.append(rise)
            extras.append((rise, "all", f"{zh}→末层 lens（{mid}→{last}）", "layer"))
        for sig, zh in WAIT:
            names.append(sig)
            for need, need_zh in (("all", "4 步都过"), ("last", "只看最后一步"), ("k3", "4 步里 3 步过")):
                layer_jobs.append((sig, need, f"{zh}，{need_zh}", "wait"))
        seen = set()
        jobs = []
        for item in layer_jobs + extras:
            key = (item[0], item[1])
            if key in seen:
                continue
            seen.add(key)
            jobs.append(item)
        name_list = list(dict.fromkeys(names))
        if parts:
            for part in parts:
                layers.precompute_events(part, name_list)
            pack = accfirst.merge_packs(parts, name)
        else:
            layers.precompute_events(pack, name_list)
        fs_rows = layers.run_pack(pack, None, float("inf"), "all")
        fs_sum = rg.summarize(fs_rows)
        print(
            f"sweep {name} n={len(pack['questions'])} fs={fs_sum['acc']:.3f}/{fs_sum['tok']:.0f} "
            f"puma={pack['puma_acc']:.3f}/{pack['puma_tok']:.0f} jobs={len(jobs)}",
            flush=True,
        )
        row: dict[str, Any] = {
            "puma_acc": pack["puma_acc"],
            "puma_tok": pack["puma_tok"],
            "fs": fs_sum,
            "picks": {},
            "pol": {},
        }
        for sig, need, zh, kind in jobs:
            rec = sweep(pack, fs_rows, fs_sum, sig, need)
            pol, n_r, n_l = polarity(pack, sig)
            key = f"{sig}|{need}"
            row["picks"][key] = rec
            row["pol"][key] = (pol, n_r, n_l, zh, kind)
            mark = ""
            if rec and rec["threshold"] != float("inf"):
                mark = f"  {fmt(rec)}"
            print(f"  {zh:28} pol={pol:+.2f}{mark}", flush=True)
        results[name] = row

    # summary tables
    set_names = [n for n, _ in named if n in results]
    layer_keys = []
    wait_keys = []
    for name in set_names:
        for key, meta in results[name]["pol"].items():
            _, _, _, zh, kind = meta
            if kind == "layer" and (key, zh) not in layer_keys:
                layer_keys.append((key, zh))
            if kind == "wait" and (key, zh) not in wait_keys:
                wait_keys.append((key, zh))

    def table(title: str, keys: list[tuple[str, str]]) -> list[str]:
        header = "| 读数 | " + " | ".join(set_names) + " | 正号集 | 不差于一半深 |"
        sep = "|---|" + "|".join(["---"] * len(set_names)) + "|---:|---:|"
        out = [f"## {title}", "", header, sep]
        # baseline half
        base_key = "exit_minus_half|all"
        for key, zh in keys:
            cells = []
            pos = 0
            n_pol = 0
            ge = 0
            n_cmp = 0
            for name in set_names:
                rec = results[name]["picks"].get(key)
                pol, _, _, _, _ = results[name]["pol"].get(key, (float("nan"), 0, 0, "", ""))
                cells.append(fmt(rec))
                if pol == pol:
                    n_pol += 1
                    if pol > 0:
                        pos += 1
                base = results[name]["picks"].get(base_key)
                if rec is not None and base is not None:
                    n_cmp += 1
                    if rec["acc"] + 1e-12 >= base["acc"] and rec["tok"] <= base["tok"] + 1e-6:
                        ge += 1
                    elif beats(rec, base):
                        ge += 1
            out.append(
                f"| {zh} | " + " | ".join(cells) + f" | {pos}/{n_pol} | {ge}/{n_cmp} |"
            )
        return out + [""]

    lines.extend(table("层对（相对深度对齐）", layer_keys))
    lines.extend(table("Wait 变体", wait_keys))

    # per-set winner
    lines += ["## 每集最好（正确率优先，再少 token）", ""]
    lines += ["| 集 | PUMA | 高置信+FS | 最好层对 | 最好 Wait | Wait 是否压过最好层对 |"]
    lines += ["|---|---|---|---|---|---|"]
    for name in set_names:
        row = results[name]
        best_l = None
        best_l_zh = "—"
        best_w = None
        best_w_zh = "—"
        for key, meta in row["pol"].items():
            _, _, _, zh, kind = meta
            rec = row["picks"].get(key)
            if rec is None:
                continue
            if kind == "layer" and beats(rec, best_l):
                best_l, best_l_zh = rec, zh
            if kind == "wait" and beats(rec, best_w):
                best_w, best_w_zh = rec, zh
        win = "是" if beats(best_w, best_l) else "否"
        lines.append(
            f"| {name} | {rg.fmt_pct(row['puma_acc'])}/{row['puma_tok']:.0f} "
            f"| {rg.fmt_pct(row['fs']['acc'])}/{row['fs']['tok']:.0f} "
            f"| {best_l_zh} {fmt(best_l)} | {best_w_zh} {fmt(best_w)} | {win} |"
        )

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"\n写成 {TABLE}", flush=True)
    print("\n".join(lines[-20:]))


if __name__ == "__main__":
    main()
