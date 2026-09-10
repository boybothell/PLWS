#!/usr/bin/env python3
"""Current Wait τ, plus re-pick: Acc not down vs 0.995, tokens must be < PUMA."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp
import report_tau995_wait_threeway as tw

TABLE = AE / "tables/conf_fs_tau995_wait_tokpuma.md"
WAIT = "stop_margin"


def sweep_points(pack, base_rows, base_sum, use_fs: bool) -> list[dict[str, Any]]:
    xs = cmp.low_xs(pack, WAIT)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = []
    for thr in thrs:
        ours = layers.run_pack(pack, WAIT, thr, "all", use_fs=use_fs)
        rec = layers.contrast(base_rows, ours, base_sum)
        rec["threshold"] = thr
        rec["signal"] = WAIT
        rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
        rec["d_acc_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
        points.append(rec)
    return points


def pick_old(points, base_acc, base_tok):
    return accfirst.pick_acc_first_both(points, base_acc, base_tok)


def pick_tok_puma(points, base_acc, puma_acc, puma_tok):
    under = [p for p in points if p["tok"] < puma_tok - 1e-6]
    keep_base = [p for p in under if p["acc"] + 1e-12 >= base_acc]
    if keep_base:
        return sorted(keep_base, key=lambda p: (-p["acc"], p["tok"]))[0]
    keep_puma = [p for p in under if p["acc"] + 1e-12 >= puma_acc]
    if keep_puma:
        return sorted(keep_puma, key=lambda p: (-p["acc"], p["tok"]))[0]
    return None


def thr_s(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "—"
    thr = rec["threshold"]
    if thr == float("inf"):
        return "不开"
    return f"{thr:.3f}"


def pair(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "不开"
    return f"{rg.fmt_pct(rec['acc'])} / {rec['tok']:.0f}"


def apply_thr(pack, thr: float, use_fs: bool, base_rows, base_sum):
    ours = layers.run_pack(pack, WAIT, thr, "all", use_fs=use_fs)
    rec = layers.contrast(base_rows, ours, base_sum)
    rec["threshold"] = thr
    rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
    rec["d_acc_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
    return rec


def main() -> None:
    rg.TAU = 0.995
    lines = [
        "# Wait 门槛：现行 Acc 优先 vs token 必须少于 PUMA",
        "",
        "交卷：试答即终答。Wait 只读 `dense_puma_wait/`，4 步都过。",
        "现行：相对只开 0.995，正确率不降且 token 不多时先取正确率最高；强停+Wait 用同一 τ。",
        "新选：先要求 Wait 列 token < PUMA。其中正确率不低于 0.995 的优先，再取正确率最高、再少 token。",
        "没有这样的点：token < PUMA 且正确率不低于 PUMA。再没有就不开 Wait。",
        "",
        "| 集 | 现行 τ | 新 τ | PUMA | 0.995 | 现行 Wait | 新 Wait | 现行 +强停 | 新 +强停 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    print(lines[-2], flush=True)
    print(lines[-1], flush=True)
    for name, cells in tw.all_jobs():
        loaded = []
        for cell in cells:
            pack = tw.load_one(cell)
            if pack is None:
                continue
            print(f"load {cell['name']} n={len(pack['questions'])} wait={pack['_wait_cov']:.2f}", flush=True)
            loaded.append(pack)
        if not loaded or len(loaded) != len(cells):
            continue
        pack = accfirst.merge_packs(loaded, name) if len(loaded) > 1 else loaded[0]
        pack["_wait_cov"] = min(p["_wait_cov"] for p in loaded)
        if pack["_wait_cov"] < 0.5:
            continue
        fs_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=True)
        fs_sum = rg.summarize(fs_rows)
        nofs_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=False)
        nofs_sum = rg.summarize(nofs_rows)
        points = sweep_points(pack, nofs_rows, nofs_sum, use_fs=False)
        old = pick_old(points, nofs_sum["acc"], nofs_sum["tok"])
        new = pick_tok_puma(points, nofs_sum["acc"], pack["puma_acc"], pack["puma_tok"])
        old_fs = apply_thr(pack, old["threshold"], True, fs_rows, fs_sum) if old else None
        new_fs = apply_thr(pack, new["threshold"], True, fs_rows, fs_sum) if new else None
        row = (
            f"| {name} | {thr_s(old)} | {thr_s(new)} | "
            f"{rg.fmt_pct(pack['puma_acc'])} / {pack['puma_tok']:.0f} | "
            f"{rg.fmt_pct(nofs_sum['acc'])} / {nofs_sum['tok']:.0f} | "
            f"{pair(old)} | {pair(new)} | {pair(old_fs)} | {pair(new_fs)} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
