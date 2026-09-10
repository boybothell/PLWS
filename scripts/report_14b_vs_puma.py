#!/usr/bin/env python3
"""14B MATH (and later oly/GPQA): high-conf+FS+lag vs official PUMA."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_8b_extra_vs_puma as extra8
import report_conf_fs_stop_margin as cmp

CELLS = (
    {
        "name": "14B-MATH",
        "dataset": "math-500",
        "trial": AE / "results/dense_G_r1_14b/math-500/dense_puma/trial_answers.json",
        "stat": AE / "results/puma_offline_r1_14b/math-500/statistics.json",
        "gpath": AE / "results/dense_G_r1_14b/math-500/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/r1_14b/math-500",
            AE / "results/confcal_judge/v2/dense_lens/r1_14b/math-500",
        ),
    },
    {
        "name": "14B-奥赛",
        "dataset": "olympiadbench",
        "trial": AE / "results/dense_G_r1_14b/olympiadbench/dense_puma/trial_answers.json",
        "stat": AE / "results/puma_offline_r1_14b/olympiadbench/statistics.json",
        "gpath": AE / "results/dense_G_r1_14b/olympiadbench/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/r1_14b/olympiadbench",
            AE / "results/confcal_judge/v2/dense_lens/r1_14b/olympiadbench",
        ),
    },
    {
        "name": "14B-GPQA",
        "dataset": "gpqa-diamond",
        "trial": AE / "results/dense_G_r1_14b/gpqa-diamond/dense_puma/trial_answers.json",
        "stat": AE / "results/puma_offline_r1_14b/gpqa-diamond/statistics.json",
        "gpath": AE / "results/dense_G_r1_14b/gpqa-diamond/per_sample.json",
        "scores": (
            AE / "results/confcal_judge/v2/dense_internal/r1_14b/gpqa-diamond",
            AE / "results/confcal_judge/v2/dense_lens/r1_14b/gpqa-diamond",
        ),
    },
)


def mean(xs: list[float]) -> str:
    good = [x for x in xs if x == x]
    if not good:
        return "—"
    return f"{statistics.mean(good):.2f}（{len(good)}）"


def diagnose(pack: dict[str, Any]) -> None:
    fs_rows = layers.run_pack(pack, None, float("inf"), "all")
    n = len(fs_rows)
    high = sum(1 for r in fs_rows if r.get("branch") == "conf")
    fs = sum(1 for r in fs_rows if r.get("branch") == "fs")
    full = sum(1 for r in fs_rows if r.get("branch") == "full")
    half_r, half_l, wait_r, wait_l = [], [], [], []
    reach = 0
    for q in pack["questions"]:
        hits_lag = False
        for ev in q["events"]:
            if ev["high"] or ev["mixed"]:
                continue
            hits_lag = True
            tag = ev.get("tag")
            hv = ev["vals"].get("exit_minus_half") or []
            wv = ev["vals"].get("stop_margin") or []
            h = min(hv) if hv and all(x == x for x in hv) else float("nan")
            w = min(wv) if wv and all(x == x for x in wv) else float("nan")
            if tag == "R":
                half_r.append(h)
                wait_r.append(w)
            elif tag == "L":
                half_l.append(h)
                wait_l.append(w)
        reach += int(hits_lag)
    print(
        f"停法 高置信 {high}/{n}  后路强停 {fs}/{n}  写完 {full}/{n}  "
        f"还会走到滞后门 {reach}/{n}",
        flush=True,
    )
    print(
        f"低置信窗 4 步最差：一半深 停对 {mean(half_r)} / 停错 {mean(half_l)}；"
        f"Wait 停对 {mean(wait_r)} / 停错 {mean(wait_l)}",
        flush=True,
    )


def main() -> None:
    want = set(sys.argv[1:] or ["14B-MATH"])
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
        diagnose(pack)
        fs_rows = layers.run_pack(pack, None, float("inf"), "all")
        fs_sum = rg.summarize(fs_rows)
        half = cmp.sweep_sig(pack, fs_rows, fs_sum, "exit_minus_half")
        wait = cmp.sweep_sig(pack, fs_rows, fs_sum, "stop_margin")
        line = extra8.row_of(cell["name"], pack, half, wait)
        print(line, flush=True)
        if half:
            print(
                f"一半深 vs PUMA  {rg.fmt_pp(half['d_acc_pp_puma'])} / {rg.fmt_tok(half['d_tok_puma'])}",
                flush=True,
            )
        if f"| {cell['name']} |" in old:
            import re

            old = re.sub(rf"\| {cell['name']} \|.*", line, old)
        else:
            if not old.endswith("\n"):
                old += "\n"
            old += line + "\n"
    if old:
        table.write_text(old)
        print(f"写成 {table}", flush=True)


if __name__ == "__main__":
    main()
