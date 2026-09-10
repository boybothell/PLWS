#!/usr/bin/env python3
"""High-conf + FS + stop_margin vs last-minus-half. Per-set Acc-first among both."""
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

SIGS = (
    ("exit_minus_half", "出口减一半深"),
    ("stop_margin", "收口比 Wait"),
)
TABLE = AE / "tables/conf_fs_stop_margin.md"


def pick_acc_first_both(points: list[dict[str, Any]], base_acc: float, base_tok: float):
    return accfirst.pick_acc_first_both(points, base_acc, base_tok)


def low_xs(pack: dict[str, Any], sig: str) -> list[float]:
    xs = []
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            vals = ev["vals"].get(sig) or []
            if vals and vals[-1] == vals[-1]:
                xs.append(vals[-1])
    return xs


def cov(pack: dict[str, Any], sig: str) -> float:
    n_sig = 0
    n_low = 0
    for q in pack["questions"]:
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            n_low += 1
            vals = ev["vals"].get(sig) or []
            if vals and vals[-1] == vals[-1]:
                n_sig += 1
    return n_sig / max(n_low, 1)


def eval_thr(pack: dict[str, Any], fs_rows: list[dict[str, Any]], fs_sum: dict[str, Any], sig: str, thr: float):
    ours = layers.run_pack(pack, sig, thr, "all")
    rec = layers.contrast(fs_rows, ours, fs_sum)
    rec["threshold"] = thr
    rec["signal"] = sig
    rec["d_acc_pp_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
    rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
    rec["both_fs"] = rec["acc"] + 1e-12 >= fs_sum["acc"] and rec["tok"] <= fs_sum["tok"] + 1e-6
    rec["both_puma"] = rec["acc"] + 1e-12 >= pack["puma_acc"] and rec["tok"] <= pack["puma_tok"] + 1e-6
    return rec


def sweep_sig(pack: dict[str, Any], fs_rows: list[dict[str, Any]], fs_sum: dict[str, Any], sig: str):
    if cov(pack, sig) < 0.5:
        return None
    xs = low_xs(pack, sig)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = [eval_thr(pack, fs_rows, fs_sum, sig, thr) for thr in thrs]
    return pick_acc_first_both(points, fs_sum["acc"], fs_sum["tok"])


def load_ready(cell: dict[str, Any]) -> dict[str, Any] | None:
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
        return None
    pack = layers.load_pack(cell)
    cons.attach_shared(pack)
    layers.precompute_events(pack, [s for s, _ in SIGS])
    if max(cov(pack, s) for s, _ in SIGS) < 0.5:
        return None
    return pack


def fmt_gate(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "读数不够"
    thr = rec["threshold"]
    thr_s = "不开" if thr == float("inf") else f"{thr:.3f}"
    return f"{rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f}（{thr_s}；放进 {rec['rescue_R']}/{rec['rescue_L']}）"


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    print(
        "高置信 + FS + 滞后。试答即终答。每集自己选门槛："
        "相对高置信+FS，正确率不降且 token 不多时先取正确率最高。",
        flush=True,
    )

    singles: list[tuple[str, dict[str, Any]]] = []
    for cell in list(rg.CELLS) + [more.EXTRA]:
        pack = load_ready(cell)
        if pack is None:
            print(f"skip {cell['name']}", flush=True)
            continue
        print(
            f"load {cell['name']} half={cov(pack, 'exit_minus_half'):.2f} "
            f"wait={cov(pack, 'stop_margin'):.2f}",
            flush=True,
        )
        singles.append((cell["name"], pack))

    aime_merged: list[tuple[str, dict[str, Any]]] = []
    for ds, zh in (("aime24", "7B AIME24"), ("aime25", "7B AIME25")):
        parts = []
        for seed in accfirst.SEEDS:
            pack = load_ready(accfirst.aime_cell(ds, seed))
            if pack is None:
                print(f"skip {ds}-s{seed}", flush=True)
                continue
            print(f"load {ds}-s{seed} wait={cov(pack, 'stop_margin'):.2f}", flush=True)
            pack["_cov"] = cov(pack, "stop_margin")
            parts.append(pack)
        if len(parts) == 4:
            aime_merged.append((zh, accfirst.merge_packs(parts, zh)))

    header = (
        "| 集 | PUMA | 高置信+FS | 出口减一半深（本集最好） | 收口比 Wait（本集最好） | Wait 相对 PUMA |"
    )
    sep = "|---|---|---|---|---|---|"
    lines = [
        "# 高置信 + FS：出口减一半深 vs 收口比 Wait",
        "",
        "交卷：试答即终答。低置信连答上 4 步都过门槛才停。",
        "选门槛：相对高置信+FS，正确率不降且 token 不多时先取正确率最高，再少 token。",
        "括号里是门槛；放进 = 对/错低置信锁。不开 = 本集最好是滞后一扇都不放。",
        "",
        header,
        sep,
    ]
    print("\n" + header)
    print(sep)
    for name, pack in singles + aime_merged:
        print(f"sweep {name}", flush=True)
        fs_rows = layers.run_pack(pack, None, float("inf"), "all")
        fs_sum = rg.summarize(fs_rows)
        half = sweep_sig(pack, fs_rows, fs_sum, "exit_minus_half")
        wait = sweep_sig(pack, fs_rows, fs_sum, "stop_margin")
        wait_vs = (
            f"{rg.fmt_pp(wait['d_acc_pp_puma'])} / {rg.fmt_tok(wait['d_tok_puma'])}"
            if wait
            else "—"
        )
        row = (
            f"| {name} | {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f} "
            f"| {rg.fmt_pct(fs_sum['acc'])} / {fs_sum['tok']:.0f} "
            f"| {fmt_gate(half)} | {fmt_gate(wait)} | {wait_vs} |"
        )
        print(row, flush=True)
        lines.append(row)

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"\n写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
