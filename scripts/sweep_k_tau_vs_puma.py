#!/usr/bin/env python3
"""Sweep k and TAU of the high-conf door vs official PUMA. EPS=0.03 MSS=10 fixed."""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import replay_rescue_R_gate as rg
from report_conf_vs_puma_all import MODELS, SETS, cell

KS = (2, 3, 4, 5, 6)
TAUS = (0.90, 0.94, 0.96, 0.97, 0.98, 0.99)


def eval_pack(pack, k: int, tau: float) -> dict:
    rg.K = k
    rg.TAU = tau
    layers.precompute_events(pack, [])
    s = rg.summarize(layers.run_pack(pack, None, float("inf"), "all"))
    return {
        "acc": s["acc"],
        "tok": s["tok"],
        "d_acc": 100.0 * (s["acc"] - pack["puma_acc"]),
        "d_tok": s["tok"] - pack["puma_tok"],
    }


def main() -> None:
    packs = []
    for tag, md, pd in MODELS:
        for zh, ds in SETS:
            c = cell(md, pd, f"{tag}-{zh}", ds)
            if not (c["trial"].is_file() and c["stat"].is_file() and c["gpath"].is_file()):
                continue
            print(f"load {c['name']}", flush=True)
            pack = layers.load_pack(c)
            packs.append((c["name"], pack))

    print(
        f"\n固定：后面几次 ≥ 第一次−{rg.EPS}，前{rg.MSS}步不许停。试答即终答。",
        flush=True,
    )
    grid = []
    for k in KS:
        for tau in TAUS:
            rows = []
            for name, pack in packs:
                rec = eval_pack(pack, k, tau)
                rec["name"] = name
                rows.append(rec)
            n = len(rows)
            mean_acc = sum(r["d_acc"] for r in rows) / n
            mean_tok = sum(r["d_tok"] for r in rows) / n
            n_acc = sum(int(r["d_acc"] >= -1e-9) for r in rows)
            n_both = sum(int(r["d_acc"] >= -1e-9 and r["d_tok"] <= 1e-9) for r in rows)
            grid.append(
                {
                    "k": k,
                    "tau": tau,
                    "mean_acc": mean_acc,
                    "mean_tok": mean_tok,
                    "n_acc": n_acc,
                    "n_both": n_both,
                    "n": n,
                    "rows": rows,
                }
            )
            print(
                f"k={k} τ={tau:.2f}  平均vs PUMA {mean_acc:+.2f}pp / {mean_tok:+.0f}  "
                f"正确率≥PUMA {n_acc}/{n}  正确率不低且token不多 {n_both}/{n}",
                flush=True,
            )

    # overall: Acc first (mean ΔAcc), then tok, then more cells Acc>=PUMA
    best = sorted(grid, key=lambda g: (-g["mean_acc"], g["mean_tok"], -g["n_acc"]))[0]
    most = sorted(grid, key=lambda g: (-g["n_acc"], -g["mean_acc"], g["mean_tok"]))[0]
    both = sorted(grid, key=lambda g: (-g["n_both"], -g["mean_acc"], g["mean_tok"]))[0]
    print("\n按平均正确率优先：k={k} τ={tau:.2f}  {mean_acc:+.2f}pp / {mean_tok:+.0f}  ≥PUMA {n_acc}/{n}".format(**best))
    print("按覆盖格数优先：k={k} τ={tau:.2f}  {mean_acc:+.2f}pp / {mean_tok:+.0f}  ≥PUMA {n_acc}/{n}".format(**most))
    print("按「正确率不低且更省」格数：k={k} τ={tau:.2f}  {mean_acc:+.2f}pp / {mean_tok:+.0f}  双赢 {n_both}/{n}".format(**both))

    show = []
    for g in (best, most, both):
        key = (g["k"], g["tau"])
        if key not in show:
            show.append(key)
    print("\n各格明细（只列上面这几档）")
    for k, tau in show:
        g = next(x for x in grid if x["k"] == k and abs(x["tau"] - tau) < 1e-9)
        print(f"\n--- k={k} τ={tau:.2f} ---")
        for r in g["rows"]:
            print(
                f"  {r['name']:<10} {rg.fmt_pct(r['acc'])} / {r['tok']:.0f}  "
                f"vs PUMA {rg.fmt_pp(r['d_acc'])} / {rg.fmt_tok(r['d_tok'])}"
            )


if __name__ == "__main__":
    main()
