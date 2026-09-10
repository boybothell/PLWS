#!/usr/bin/env python3
"""0.995+FS+half / Wait / both, vs original 0.98+FS and official PUMA."""
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
import report_brightest_stop_auroc as jobs
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp
import report_half_wait_combo as combo

TABLE = AE / "tables/conf_fs_tau995_fiveway.md"
SIGS = [s for s, _ in cmp.SIGS]
OLD_TAU = 0.98
NEW_TAU = 0.995


def load_any(cell: dict[str, Any]) -> dict[str, Any] | None:
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    pack = layers.load_pack(cell)
    ids = more.discover_layers(pack["scores"])
    if len(ids) >= 2:
        cons.attach_shared(pack)
    pack["_n"] = len(pack["questions"])
    return pack


def bind(pack: dict[str, Any], tau: float) -> None:
    rg.TAU = tau
    layers.precompute_events(pack, SIGS)
    pack["_half_cov"] = cmp.cov(pack, "exit_minus_half")
    pack["_wait_cov"] = cmp.cov(pack, "stop_margin")
    pack["_cov"] = max(pack["_half_cov"], pack["_wait_cov"])


def fs_of(pack: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = layers.run_pack(pack, None, float("inf"), "all")
    return rows, rg.summarize(rows)


def fmt_acc(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "读数不够"
    return f"{rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f}"


def combo_of(pack, fs_rows, fs_sum, half, wait):
    if half is None or wait is None:
        return None
    ta = half["threshold"]
    tb = wait["threshold"]
    pts = []
    base = combo.eval_rows(pack, fs_rows, fs_sum, fs_rows)
    base["threshold"] = float("inf")
    pts.append(base)
    for mode, a, b in (("and", ta, tb), ("or", ta, tb)):
        if mode == "and" and (a == float("inf") or b == float("inf")):
            continue
        if mode == "or" and a == float("inf") and b == float("inf"):
            continue
        rec = combo.eval_rows(pack, fs_rows, fs_sum, combo.run_combo_once(pack, mode, a, b))
        rec["threshold"] = a
        pts.append(rec)
    return accfirst.pick_acc_first_both(pts, fs_sum["acc"], fs_sum["tok"])


def ready_lag(pack: dict[str, Any]) -> bool:
    return pack.get("_half_cov", 0) >= 0.5 and pack.get("_wait_cov", 0) >= 0.5


def main() -> None:
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    print(
        "五列：0.995+强停+一半深 / Wait / 两扇叠加，对照原来 0.98+强停 和官方 PUMA。"
        "滞后门槛相对 0.995+强停，正确率不降且字数不升时先取正确率最高。"
        "叠加=用两扇各自最好门槛，两扇都过或一扇过，再按同一尺子选。",
        flush=True,
    )
    header = (
        "| 集 | 0.995+强停+一半深 | 0.995+强停+Wait | 0.995+强停+Wait+一半深 | 原来 0.98+强停 | PUMA |"
    )
    sep = "|---|---|---|---|---|---|"
    lines = [
        "# 0.995 + 强停 + 滞后 vs 原来的置信度门 vs PUMA",
        "",
        "交卷：试答即终答。强停=后路 FS。原来的置信度门=第一次≥0.98、后面≥第一次−0.03、前 10 步不停。",
        "0.995 门同规则，只把第一次改成≥0.995。滞后门槛相对 0.995+强停自选。",
        "Wait+一半深：两扇用各自最好门槛，两扇都过或一扇过，再按正确率优先选。读数不够=中间层未齐。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, cells, is_aime in jobs.all_jobs():
        parts = []
        for cell in cells:
            pack = load_any(cell)
            if pack is not None:
                parts.append(pack)
        if is_aime and len(parts) != 4:
            print(f"| {name} | 缺 seed |", flush=True)
            continue
        if not parts:
            print(f"| {name} | 缺文件 |", flush=True)
            continue

        for p in parts:
            bind(p, OLD_TAU)
        old_pack = accfirst.merge_packs(parts, name) if is_aime else parts[0]
        _, old_sum = fs_of(old_pack)

        for p in parts:
            bind(p, NEW_TAU)
        pack = accfirst.merge_packs(parts, name) if is_aime else parts[0]
        pack["_half_cov"] = min(p["_half_cov"] for p in parts)
        pack["_wait_cov"] = min(p["_wait_cov"] for p in parts)
        fs_rows, fs_sum = fs_of(pack)
        if ready_lag(pack):
            half = cmp.sweep_sig(pack, fs_rows, fs_sum, "exit_minus_half")
            wait = cmp.sweep_sig(pack, fs_rows, fs_sum, "stop_margin")
            both = combo_of(pack, fs_rows, fs_sum, half, wait)
        else:
            half = wait = both = None
        row = (
            f"| {name} | {fmt_acc(half)} | {fmt_acc(wait)} | {fmt_acc(both)} "
            f"| {fmt_acc(old_sum)} | {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f} |"
        )
        print(
            f"done {name} n={len(pack['questions'])} cov={pack.get('_half_cov', 0):.2f}",
            flush=True,
        )
        print(row, flush=True)
        lines.append(row)

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"\n写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
