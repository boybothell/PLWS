#!/usr/bin/env python3
"""Official PUMA k/τ on dense trials, gold Acc, trial-as-final, no FS."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/puma_k_dense_gold.md"


def eval_gate(pack: dict[str, Any], k: int, tau: float) -> dict[str, Any]:
    rg.K = k
    rec = sw.eval_tau(pack, tau)
    rec["k"] = k
    rec["tau"] = tau
    return rec


def main() -> None:
    packs = sw.load_jobs()
    header = (
        "| 集 | 官方 PUMA | 密探 · 官方门 k=2 τ=0.98 | 密探 · k=4 τ=0.995 |"
    )
    sep = "|---|---|---|---|"
    lines = [
        "# 官方门搬到密探上（正确率只对金标）",
        "",
        "官方 PUMA：连续 2 次同一答案、第一次把握 ≥ 0.98、后面 ≥ 第一次 − 0.03、前 10 步不停；",
        "稀探；截断后再重写终答。",
        "右边两列：同一条 CoT 上每步都试答，闸门说停就交当时 boxed 试答，不再重写，不开后路强停。",
        "正确率只对标准答案。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, pack in packs:
        a = eval_gate(pack, 2, 0.98)
        b = eval_gate(pack, 4, 0.995)
        row = (
            f"| {name} | {sw.fmt_pair(pack['puma_acc'], pack['puma_tok'])} "
            f"| {sw.fmt_pair(a['acc'], a['tok'])}（{rg.fmt_pp(a['d_acc'])} / {rg.fmt_tok(a['d_tok'])}；"
            f"高置信停{a['n_conf']}/{a['n']} 停错{a['n_wrong_conf']}） "
            f"| {sw.fmt_pair(b['acc'], b['tok'])}（{rg.fmt_pp(b['d_acc'])} / {rg.fmt_tok(b['d_tok'])}） |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
