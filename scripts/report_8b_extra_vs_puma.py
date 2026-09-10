#!/usr/bin/env python3
"""8B olympiad / GPQA: high-conf+FS+lag vs official PUMA. Acc-first among both."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_conf_fs_stop_margin as cmp

CELLS = (
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
    },
    {
        "name": "8B-奥赛",
        "dataset": "olympiadbench",
        "trial": AE / "results/dense_G_nemotron_8b/olympiadbench/dense_puma/trial_answers.json",
        "stat": AE / "results/puma_offline_nemotron_8b/olympiadbench/statistics.json",
        "gpath": AE / "results/dense_G_nemotron_8b/olympiadbench/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/nemotron_8b/olympiadbench",
            AE / "results/confcal_judge/v2/dense_lens/nemotron_8b/olympiadbench",
        ),
    },
    {
        "name": "8B-GPQA",
        "dataset": "gpqa-diamond",
        "trial": AE / "results/dense_G_nemotron_8b/gpqa-diamond/dense_puma/trial_answers.json",
        "stat": AE / "results/puma_offline_nemotron_8b/gpqa-diamond/statistics.json",
        "gpath": AE / "results/dense_G_nemotron_8b/gpqa-diamond/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/nemotron_8b/gpqa-diamond",
            AE / "results/confcal_judge/v2/dense_lens/nemotron_8b/gpqa-diamond",
        ),
    },
)


def row_of(name: str, pack: dict[str, Any], half, wait) -> str:
    wait_vs = (
        f"{rg.fmt_pp(wait['d_acc_pp_puma'])} / {rg.fmt_tok(wait['d_tok_puma'])}"
        if wait
        else "—"
    )
    fs_rows = layers.run_pack(pack, None, float("inf"), "all")
    fs_sum = rg.summarize(fs_rows)
    return (
        f"| {name} | {rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f} "
        f"| {rg.fmt_pct(fs_sum['acc'])} / {fs_sum['tok']:.0f} "
        f"| {cmp.fmt_gate(half)} | {cmp.fmt_gate(wait)} | {wait_vs} |"
    )


def main() -> None:
    want = set(sys.argv[1:] or ["8B-GPQA"])
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    header = (
        "| 集 | PUMA | 高置信+FS | 出口减一半深（本集最好） | 收口比 Wait（本集最好） | Wait 相对 PUMA |"
    )
    print(
        "高置信 + FS + 滞后。试答即终答。本集选门槛："
        "相对高置信+FS，正确率不降且 token 不多时先取正确率最高。",
        flush=True,
    )
    print(header)
    print("|---|---|---|---|---|---|")
    table = AE / "tables/conf_fs_stop_margin.md"
    old = table.read_text() if table.is_file() else ""
    for cell in CELLS:
        if cell["name"] not in want and want != {"all"}:
            continue
        pack = cmp.load_ready(cell)
        if pack is None:
            print(f"| {cell['name']} | — | — | 读数不够 | 读数不够 | — |", flush=True)
            continue
        print(
            f"load {cell['name']} half_layer={pack.get('_half_layer')} "
            f"half={cmp.cov(pack, 'exit_minus_half'):.2f} "
            f"wait={cmp.cov(pack, 'stop_margin'):.2f}",
            flush=True,
        )
        fs_rows = layers.run_pack(pack, None, float("inf"), "all")
        fs_sum = rg.summarize(fs_rows)
        half = cmp.sweep_sig(pack, fs_rows, fs_sum, "exit_minus_half")
        wait = cmp.sweep_sig(pack, fs_rows, fs_sum, "stop_margin")
        line = row_of(cell["name"], pack, half, wait)
        print(line, flush=True)
        if f"| {cell['name']} |" in old:
            import re

            old = re.sub(rf"\| {cell['name']} \|.*", line, old)
        else:
            if not old.endswith("\n"):
                old += "\n"
            old += line + "\n"
        if half:
            print(
                f"一半深 vs PUMA  {rg.fmt_pp(half['d_acc_pp_puma'])} / {rg.fmt_tok(half['d_tok_puma'])}",
                flush=True,
            )
    if old:
        table.write_text(old)
        print(f"写成 {table}", flush=True)


if __name__ == "__main__":
    main()
