#!/usr/bin/env python3
"""Conf-only (new high door) vs official PUMA on every model/set that has trials."""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import replay_rescue_R_gate as rg

MODELS = (
    ("7B", "dense_G_r1_7b", "puma_offline_r1_7b"),
    ("8B", "dense_G_nemotron_8b", "puma_offline_nemotron_8b"),
    ("14B", "dense_G_r1_14b", "puma_offline_r1_14b"),
    ("32B", "dense_G_r1_32b", "puma_offline_r1_32b"),
)
SETS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
)


def cell(model_dir: str, puma_dir: str, name: str, dataset: str) -> dict:
    g = AE / "results" / model_dir / dataset
    if dataset == "math-500" and model_dir == "dense_G_r1_7b":
        stat = AE / "results/math500_official/puma_ds7b/statistics.json"
    else:
        stat = AE / "results" / puma_dir / dataset / "statistics.json"
    return {
        "name": name,
        "dataset": dataset,
        "trial": g / "dense_puma/trial_answers.json",
        "stat": stat,
        "gpath": g / "per_sample.json",
        "scores": (),
        "preferred": "",
    }


def main() -> None:
    print(
        f"高置信度门：k={rg.K} 第一次≥{rg.TAU} 后面≥第一次−{rg.EPS} 前{rg.MSS}步不许停。试答即终答。",
        flush=True,
    )
    print(f"{'模型':<6} {'集':<6} {'题':>4}  {'PUMA':>16}  {'只看置信度':>16}  {'vs PUMA':>18}  {'写完':>10}")
    for tag, md, pd in MODELS:
        for zh, ds in SETS:
            c = cell(md, pd, f"{tag}-{zh}", ds)
            if not c["trial"].is_file() or not c["stat"].is_file() or not c["gpath"].is_file():
                print(f"{tag:<6} {zh:<6}  缺文件", flush=True)
                continue
            print(f"load {c['name']}", flush=True)
            pack = layers.load_pack(c)
            layers.precompute_events(pack, [])
            conf = rg.summarize(layers.run_pack(pack, None, float("inf"), "all"))
            n = len(pack["questions"])
            full = sum(int(q["orig_ok"]) for q in pack["questions"]) / max(n, 1)
            print(
                f"{tag:<6} {zh:<6} {n:4d}  "
                f"{rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f}  "
                f"{rg.fmt_pct(conf['acc'])} / {conf['tok']:.0f}  "
                f"{rg.fmt_pp(100*(conf['acc']-pack['puma_acc']))} / {rg.fmt_tok(conf['tok']-pack['puma_tok'])}  "
                f"{rg.fmt_pct(full)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
