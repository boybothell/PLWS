#!/usr/bin/env python3
"""Sweep high-conf TAU (k=4, no FS) vs official PUMA. Acc first."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import replay_rescue_R_gate as rg
import report_brightest_stop_auroc as jobs
import report_conf_fs_lag_accfirst as accfirst

TABLE = AE / "tables/conf_tau_vs_puma.md"
TAUS = (0.99, 0.992, 0.995, 0.997, 0.998, 0.999, 0.9995, 1.0)
MODELS = jobs.MODELS + (
    ("Qwen3-4B", "qwen3_4b", "dense_G_qwen3_4b", "puma_offline_qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b", "dense_G_qwen3_8b", "puma_offline_qwen3_8b"),
)


def all_jobs() -> list[tuple[str, list[dict[str, Any]]]]:
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for zh, tag, gdir, puma in MODELS:
        for ds_zh, dataset in jobs.BIG:
            if tag == "qwen3_30b_a3b":
                continue
            out.append((f"{zh} {ds_zh}", [jobs.make_cell(zh, tag, gdir, puma, ds_zh, dataset)]))
        for ds_zh, dataset in jobs.AIME:
            cells = [jobs.make_cell(zh, tag, gdir, puma, ds_zh, dataset, seed) for seed in jobs.SEEDS]
            out.append((f"{zh} {ds_zh}", cells))
    return out


def load_jobs() -> list[tuple[str, dict[str, Any]]]:
    loaded: list[tuple[str, dict[str, Any]]] = []
    for name, cells in all_jobs():
        packs: list[dict[str, Any]] = []
        for cell in cells:
            cell = dict(cell)
            cell["scores"] = ()
            if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
                continue
            print(f"load {cell['name']}", flush=True)
            pack = layers.load_pack(cell)
            pack["_cov"] = 1.0
            packs.append(pack)
        if not packs:
            continue
        if len(cells) > 1 and len(packs) != len(cells):
            print(f"skip {name} seeds={len(packs)}/{len(cells)}", flush=True)
            continue
        pack = accfirst.merge_packs(packs, name) if len(packs) > 1 else packs[0]
        loaded.append((name, pack))
    return loaded


def eval_tau(pack: dict[str, Any], tau: float) -> dict[str, Any]:
    rg.TAU = tau
    pack.setdefault("scores", {})
    layers.precompute_events(pack, [])
    rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=False)
    s = rg.summarize(rows)
    n = max(s["n"], 1)
    n_wrong_conf = sum(int(r["branch"] == "conf" and not r["ok"]) for r in rows)
    n_hurt_puma = sum(int(q["puma_ok"] and not r["ok"]) for q, r in zip(pack["questions"], rows))
    n_gain_puma = sum(int((not q["puma_ok"]) and r["ok"]) for q, r in zip(pack["questions"], rows))
    return {
        "acc": s["acc"],
        "tok": s["tok"],
        "n": s["n"],
        "n_conf": s["n_conf"],
        "frac_conf": s["frac_conf"],
        "n_wrong_conf": n_wrong_conf,
        "n_hurt_puma": n_hurt_puma,
        "n_gain_puma": n_gain_puma,
        "d_acc": 100.0 * (s["acc"] - pack["puma_acc"]),
        "d_tok": s["tok"] - pack["puma_tok"],
        "puma_acc": pack["puma_acc"],
        "puma_tok": pack["puma_tok"],
        "ok_vs_puma": s["acc"] + 1e-12 >= pack["puma_acc"],
        "q_hurt": n * max(0.0, pack["puma_acc"] - s["acc"]),
    }


def fmt_pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    packs = load_jobs()
    grid: dict[float, list[dict[str, Any]]] = {}
    for tau in TAUS:
        print(f"\n=== τ={tau} ===", flush=True)
        rows = []
        for name, pack in packs:
            rec = eval_tau(pack, tau)
            rec["name"] = name
            rows.append(rec)
            print(
                f"  {name:<16} {fmt_pair(rec['acc'], rec['tok'])}  "
                f"vs PUMA {rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])}  "
                f"高置信停{rec['n_conf']}/{rec['n']} 停错{rec['n_wrong_conf']} "
                f"伤PUMA{rec['n_hurt_puma']} 救回{rec['n_gain_puma']}",
                flush=True,
            )
        grid[tau] = rows

    def stats(tau: float) -> dict[str, Any]:
        rows = grid[tau]
        return {
            "tau": tau,
            "n_ok": sum(int(r["ok_vs_puma"]) for r in rows),
            "n": len(rows),
            "worst": min(r["d_acc"] for r in rows),
            "mean_acc": sum(r["d_acc"] for r in rows) / len(rows),
            "mean_tok": sum(r["d_tok"] for r in rows) / len(rows),
            "q_hurt": sum(r["q_hurt"] for r in rows),
        }

    ranked = [stats(tau) for tau in TAUS]
    safe = [s for s in ranked if s["n_ok"] == s["n"]]
    pick = sorted(safe, key=lambda s: (s["mean_tok"], -s["mean_acc"]))[0] if safe else None
    if pick is None:
        pick = sorted(ranked, key=lambda s: (-s["n_ok"], -s["worst"], s["q_hurt"], s["mean_tok"]))[0]

    lines = [
        "# 高置信门槛 vs PUMA（只开连答门，k=4，不开强停）",
        "",
        "试答即终答。第一次 ≥ τ，后面 ≥ 第一次 − 0.03，前 10 步不停。",
        "同集现成轨迹回放，不是冻住的方法。4B 不当评判器，只当一格。",
        f"先取所有格正确率都不低于 PUMA 的 τ，再取平均 token 最少的： **τ={pick['tau']}**"
        + ("（有格仍低于 PUMA，这是最少伤的一档）" if pick["n_ok"] < pick["n"] else "")
        + "。",
        "",
        "| τ | 格≥PUMA | 最差格 | 平均 vs PUMA |",
        "|---:|---:|---:|---|",
    ]
    for s in ranked:
        mark = " ←" if s["tau"] == pick["tau"] else ""
        lines.append(
            f"| {s['tau']} | {s['n_ok']}/{s['n']} | {rg.fmt_pp(s['worst'])} | "
            f"{rg.fmt_pp(s['mean_acc'])} / {rg.fmt_tok(s['mean_tok'])}{mark} |"
        )
    lines += [
        "",
        f"| 集 | PUMA | 0.995 | τ={pick['tau']} |",
        "|---|---|---|---|",
    ]
    for name, _pack in packs:
        a = next(r for r in grid[0.995] if r["name"] == name)
        b = next(r for r in grid[pick["tau"]] if r["name"] == name)
        lines.append(
            f"| {name} | {fmt_pair(a['puma_acc'], a['puma_tok'])} | "
            f"{fmt_pair(a['acc'], a['tok'])} | {fmt_pair(b['acc'], b['tok'])} |"
        )
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"\npick τ={pick['tau']} ≥PUMA {pick['n_ok']}/{pick['n']} worst={pick['worst']:+.2f}pp", flush=True)
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
