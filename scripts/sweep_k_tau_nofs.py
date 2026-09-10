#!/usr/bin/env python3
"""k=2/3/4 × τ vs official PUMA. Trial-as-final, no FS."""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/conf_k_tau_vs_puma.md"
KS = (2, 3, 4)
TAUS = (0.98, 0.995, 0.999, 0.9995)


def main() -> None:
    packs = sw.load_jobs()
    grid: dict[tuple[int, float], list[dict]] = {}
    for k in KS:
        for tau in TAUS:
            print(f"\n=== k={k} τ={tau} ===", flush=True)
            rows = []
            rg.K = k
            for name, pack in packs:
                rec = sw.eval_tau(pack, tau)
                rec["name"] = name
                rows.append(rec)
                print(
                    f"  {name:<16} {sw.fmt_pair(rec['acc'], rec['tok'])}  "
                    f"vs PUMA {rg.fmt_pp(rec['d_acc'])} / {rg.fmt_tok(rec['d_tok'])}  "
                    f"≥{int(rec['ok_vs_puma'])} 停错{rec['n_wrong_conf']}",
                    flush=True,
                )
            grid[(k, tau)] = rows

    lines = [
        "# k × τ vs PUMA（试答即终答，不开强停）",
        "",
        "官方 PUMA 是 k=2、τ=0.98，但截断后再写终答。这里只交试答。",
        "同集回放，不是冻住的方法。",
        "",
        "| k | τ | 格≥PUMA | 最差格 | 平均 vs PUMA |",
        "|---:|---:|---:|---:|---|",
    ]
    ranked = []
    for k in KS:
        for tau in TAUS:
            rows = grid[(k, tau)]
            n_ok = sum(int(r["ok_vs_puma"]) for r in rows)
            worst = min(r["d_acc"] for r in rows)
            mean_acc = sum(r["d_acc"] for r in rows) / len(rows)
            mean_tok = sum(r["d_tok"] for r in rows) / len(rows)
            ranked.append((k, tau, n_ok, len(rows), worst, mean_acc, mean_tok))
            lines.append(
                f"| {k} | {tau} | {n_ok}/{len(rows)} | {rg.fmt_pp(worst)} | "
                f"{rg.fmt_pp(mean_acc)} / {rg.fmt_tok(mean_tok)} |"
            )

    show = ((2, 0.98), (2, 0.995), (3, 0.995), (4, 0.995), (4, 0.9995))
    names = [name for name, _ in packs]
    header = "| 集 | PUMA | " + " | ".join(f"k={k} τ={tau}" for k, tau in show) + " |"
    lines += ["", header, "|---|" + "---|" * (1 + len(show))]
    for name in names:
        puma = next(r for r in grid[(4, 0.995)] if r["name"] == name)
        cells = [sw.fmt_pair(puma["puma_acc"], puma["puma_tok"])]
        for key in show:
            rec = next(r for r in grid[key] if r["name"] == name)
            cells.append(sw.fmt_pair(rec["acc"], rec["tok"]))
        lines.append("| " + name + " | " + " | ".join(cells) + " |")

    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)
    print("汇总：")
    for row in ranked:
        print(f"  k={row[0]} τ={row[1]}  ≥PUMA {row[2]}/{row[3]}  最差{row[4]:+.2f}  平均{row[5]:+.2f}pp / {row[6]:+.0f}")


if __name__ == "__main__":
    main()
