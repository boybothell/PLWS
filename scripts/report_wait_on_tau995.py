#!/usr/bin/env python3
"""Wait stacked on 0.995+FS (trial-as-final), not official PUMA."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import replay_rescue_R_gate as rg
import report_conf_fs_stop_margin as cmp
import report_prev_sameset_tok as prev

TABLE = AE / "tables/wait_on_tau995.md"
SIG = "stop_margin"
NEED = "last"


def pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def sweep_wait(pack: dict[str, Any], host_rows: list[dict[str, Any]], host: dict[str, Any]) -> dict[str, Any]:
    xs = cmp.low_xs(pack, SIG)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = []
    for thr in thrs:
        ours = layers.run_pack(pack, SIG, thr, NEED, use_fs=True)
        rec = layers.contrast(host_rows, ours, host)
        rec["threshold"] = thr
        rec["d_acc"] = 100.0 * (rec["acc"] - host["acc"])
        rec["d_tok"] = rec["tok"] - host["tok"]
        rec["d_puma_acc"] = 100.0 * (rec["acc"] - pack["puma_acc"])
        rec["d_puma_tok"] = rec["tok"] - pack["puma_tok"]
        points.append(rec)
    return prev.pick_tok_first(points, host["acc"])


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    rg.USE_FS = True
    header = (
        "| 集 | 官方 PUMA | 0.995+后路 Acc | 0.995+后路 token | "
        "+Wait Acc | +Wait token | 比 0.995套 | 比 PUMA |"
    )
    sep = "|---|---|---:|---:|---:|---:|---|---|"
    lines = [
        "# 收口比 Wait 叠在 0.995+后路强停上",
        "",
        "底是我们的 0.995 那一套：高置信连答 k=4、第一次≥0.995，后面≥第一次−0.03，加上后路强停，交试答。",
        "Wait 只加在低置信连答窗最后一步，必须发生在 0.995 / 后路停点之前。",
        "同集偷看：相对 0.995+后路，Acc 不降，再取 token 最少。换集会塌。",
        "AIME 四个 seed。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, pack in prev.load_78():
        if cmp.cov(pack, SIG) < 0.85:
            print(f"| {name} | 未齐 |", flush=True)
            lines.append(f"| {name} | 未齐 |")
            continue
        host_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=True)
        host = rg.summarize(host_rows)
        pick = sweep_wait(pack, host_rows, host)
        if pick["threshold"] == float("inf"):
            wait_note = "不开"
        else:
            wait_note = f"门 {pick['threshold']:.3f}；{pick['n_rescue']}火"
        row = (
            f"| {name} | {pair(pack['puma_acc'], pack['puma_tok'])} "
            f"| {rg.fmt_pct(host['acc'])} | {host['tok']:.0f} "
            f"| {rg.fmt_pct(pick['acc'])} | {pick['tok']:.0f} "
            f"| {rg.fmt_pp(pick['d_acc'])} / {rg.fmt_tok(pick['d_tok'])}（{wait_note}） "
            f"| {rg.fmt_pp(pick['d_puma_acc'])} / {rg.fmt_tok(pick['d_puma_tok'])} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
