#!/usr/bin/env python3
"""TAU=0.995 high-conf + FS + lag, vs official PUMA. Acc-first lag pick."""
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

TABLE = AE / "tables/conf_fs_tau995.md"
NEW_TAU = 0.995


def load_any(cell: dict[str, Any]) -> dict[str, Any] | None:
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    pack = layers.load_pack(cell)
    ids = more.discover_layers(pack["scores"])
    if len(ids) >= 2:
        cons.attach_shared(pack)
    layers.precompute_events(pack, [s for s, _ in cmp.SIGS])
    pack["_half_cov"] = cmp.cov(pack, "exit_minus_half")
    pack["_wait_cov"] = cmp.cov(pack, "stop_margin")
    pack["_cov"] = max(pack["_half_cov"], pack["_wait_cov"])
    pack["_n"] = len(pack["questions"])
    return pack


def sweep_row(name: str, pack: dict[str, Any]) -> str:
    fs_rows = layers.run_pack(pack, None, float("inf"), "all")
    fs_sum = rg.summarize(fs_rows)
    half = cmp.sweep_sig(pack, fs_rows, fs_sum, "exit_minus_half")
    wait = cmp.sweep_sig(pack, fs_rows, fs_sum, "stop_margin")
    half_vs = (
        f"{rg.fmt_pp(half['d_acc_pp_puma'])} / {rg.fmt_tok(half['d_tok_puma'])}" if half else "—"
    )
    wait_vs = (
        f"{rg.fmt_pp(wait['d_acc_pp_puma'])} / {rg.fmt_tok(wait['d_tok_puma'])}" if wait else "—"
    )
    fs_vs = f"{rg.fmt_pp(100*(fs_sum['acc']-pack['puma_acc']))} / {rg.fmt_tok(fs_sum['tok']-pack['puma_tok'])}"
    return (
        f"| {name} | {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f} "
        f"| {rg.fmt_pct(fs_sum['acc'])} / {fs_sum['tok']:.0f}（{fs_vs}） "
        f"| {cmp.fmt_gate(half)} | {half_vs} "
        f"| {cmp.fmt_gate(wait)} | {wait_vs} |"
    )


def main() -> None:
    rg.TAU = NEW_TAU
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    print(
        f"高置信门第一次≥{rg.TAU}，后面≥第一次−{rg.EPS}，前{rg.MSS}步不停。"
        "叠 FS + 滞后。试答即终答。滞后门槛相对 0.995+FS："
        "正确率不降且 token 不多时先取正确率最高。",
        flush=True,
    )
    header = (
        "| 集 | PUMA | 0.995+FS（相对 PUMA） | 一半深（本集最好） | 一半深相对 PUMA | Wait（本集最好） | Wait 相对 PUMA |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines = [
        "# 高置信 0.995 + FS + 滞后 vs 官方 PUMA",
        "",
        "交卷：试答即终答。高置信门第一次≥0.995，后面≥第一次−0.03，前 10 步不停。",
        "滞后：低置信连答 4 步都过门槛才停。门槛相对 0.995+FS，正确率不降且 token 不多时先取正确率最高，再少 token。",
        "括号里是门槛；放进 = 对/错低置信锁。不开 = 本集最好是滞后一扇都不放。读数不够 = 中间层/Wait 未齐，只报 0.995+FS。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, cells, is_aime in jobs.all_jobs():
        packs = []
        missing = []
        for cell in cells:
            pack = load_any(cell)
            if pack is None:
                missing.append(cell["name"])
                continue
            packs.append(pack)
        if is_aime:
            if len(packs) != 4:
                print(f"| {name} | 缺 seed：{', '.join(missing) or '文件不全'} |", flush=True)
                continue
            pack = accfirst.merge_packs(packs, name)
            pack["_half_cov"] = min(p["_half_cov"] for p in packs)
            pack["_wait_cov"] = min(p["_wait_cov"] for p in packs)
        else:
            if not packs:
                print(f"| {name} | 缺文件 |", flush=True)
                continue
            pack = packs[0]
        print(
            f"sweep {name} n={pack.get('_n', len(pack['questions']))} "
            f"half={pack.get('_half_cov', 0):.2f} wait={pack.get('_wait_cov', 0):.2f}",
            flush=True,
        )
        row = sweep_row(name, pack)
        print(row, flush=True)
        lines.append(row)

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"\n写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
