#!/usr/bin/env python3
"""PUMA-base: last-minus-half with each 4-step aggregation, 7B and 8B."""
from __future__ import annotations

import sys
from pathlib import Path

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import screen_puma_layers as sc
import sweep_tau_vs_puma as sw

TABLE = AE / "tables/screen_puma_layers.md"
SIG = "exit_minus_half"


def main() -> None:
    packs = sc.load_ready()
    rows = []
    print("| 集 | 4步都过 | 只看最后一步 | 4步平均 | 4步里3步过 | 最后减第一步 |", flush=True)
    for name, pack in packs:
        cells = []
        for agg in sc.AGGS:
            rec = sc.sweep_cfg(pack, SIG, agg, False)
            if rec is None:
                rec = sc.sweep_cfg(pack, SIG, agg, True)
            cells.append(rec)
        text = []
        for rec in cells:
            if rec is None or rec["threshold"] == float("inf"):
                text.append("不开")
            else:
                flip = "取反 " if rec["flip"] else ""
                text.append(f"{flip}{sw.fmt_pair(rec['acc'], rec['tok'])}（{rg.fmt_pp(rec['d_acc'])}）")
        row = f"| {name} | " + " | ".join(text) + " |"
        print(row, flush=True)
        rows.append(row)

    old = TABLE.read_text() if TABLE.is_file() else ""
    extra = [
        "",
        "## 一半深只换 4 步怎么收（7B / 8B）",
        "",
        "分数仍是末层减一半深，不换层。门槛本集自选。",
        "",
        "| 集 | 4步都过 | 只看最后一步 | 4步平均 | 4步里3步过 | 最后减第一步 |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    if "## 一半深只换 4 步怎么收" in old:
        head = old.split("## 一半深只换 4 步怎么收")[0].rstrip()
        TABLE.write_text(head + "\n" + "\n".join(extra) + "\n")
    else:
        TABLE.write_text(old.rstrip() + "\n" + "\n".join(extra) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
