#!/usr/bin/env python3
"""Gold-only Acc: PUMA vs 0.995 vs same-set half, with/without FS."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_constructed as cons
import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_brightest_stop_auroc as jobs
import report_conf_fs_lag_accfirst as accfirst
import report_conf_fs_stop_margin as cmp

TABLE = AE / "tables/conf_fs_tau995_gold.md"
SIG = "exit_minus_half"
MODELS = jobs.MODELS + (
    ("Qwen3-4B", "qwen3_4b", "dense_G_qwen3_4b", "puma_offline_qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b", "dense_G_qwen3_8b", "puma_offline_qwen3_8b"),
)


def all_jobs() -> list[tuple[str, list[dict[str, Any]], bool]]:
    out: list[tuple[str, list[dict[str, Any]], bool]] = []
    for zh, tag, gdir, puma in MODELS:
        for ds_zh, dataset in jobs.BIG:
            if tag == "qwen3_30b_a3b":
                continue
            out.append((f"{zh} {ds_zh}", [jobs.make_cell(zh, tag, gdir, puma, ds_zh, dataset)], False))
        for ds_zh, dataset in jobs.AIME:
            cells = [jobs.make_cell(zh, tag, gdir, puma, ds_zh, dataset, seed) for seed in jobs.SEEDS]
            out.append((f"{zh} {ds_zh}", cells, True))
    return out


def load_one(cell: dict[str, Any]) -> dict[str, Any] | None:
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    pack = layers.load_pack(cell)
    ids = more.discover_layers(pack["scores"])
    if len(ids) >= 2:
        cons.attach_shared(pack)
    layers.precompute_events(pack, [SIG])
    pack["_half_cov"] = cmp.cov(pack, SIG)
    pack["_cov"] = pack["_half_cov"]
    pack["_n"] = len(pack["questions"])
    return pack


def sweep_half(
    pack: dict[str, Any],
    base_rows: list[dict[str, Any]],
    base_sum: dict[str, Any],
    use_fs: bool,
) -> dict[str, Any] | None:
    if pack.get("_half_cov", 0) < 0.5:
        return None
    xs = cmp.low_xs(pack, SIG)
    thrs = [float("inf")]
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = []
    for thr in thrs:
        ours = layers.run_pack(pack, SIG, thr, "all", use_fs=use_fs)
        rec = layers.contrast(base_rows, ours, base_sum)
        rec["threshold"] = thr
        rec["d_acc_pp_puma"] = 100.0 * (rec["acc"] - pack["puma_acc"])
        rec["d_tok_puma"] = rec["tok"] - pack["puma_tok"]
        points.append(rec)
    return accfirst.pick_acc_first_both(points, base_sum["acc"], base_sum["tok"])


def fmt_pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def fmt_half(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "未齐"
    core = fmt_pair(rec["acc"], rec["tok"])
    if rec["threshold"] == float("inf"):
        return f"{core}（不开）"
    return f"{core}（门 {rec['threshold']:.3f}）"


def job_name(name: str, n_loaded: int, n_cells: int) -> str:
    if n_loaded == n_cells:
        return name
    return f"{name}（{n_loaded} seed）"


def main() -> None:
    rg.TAU = 0.995
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    header = (
        "| 集 | 官方 PUMA | 只开 0.995 | 0.995 + 一半深（同集） | 0.995 + 后路强停 + 一半深（同集） |"
    )
    sep = "|---|---|---|---|---|"
    lines = [
        "# 试答即终答：0.995 / 同集一半深（正确率只对金标）",
        "",
        "交卷是当时那步 boxed 试答，不再重写。正确率只对标准答案。",
        "试答只跟写完终答一样、标准答案不对，算错。",
        "高置信：同一答案连续 4 次，第一次 ≥ 0.995，后面 ≥ 第一次 − 0.03，前 10 步不停。",
        "一半深 = 末层 boxed 平均 logp − 总层数一半那一层（7B 第 14、8B 第 16、14B 第 24、Qwen3 第 18）。",
        "同集门槛：相对左边那列（只开 0.995，或 0.995+后路强停），正确率不降且 token 不多时先取正确率最高。",
        "不开 = 本集最好是滞后一扇都不放。未齐 = 中间层覆盖不到一半。AIME 要四 seed，缺太多的不报。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, cells, is_aime in all_jobs():
        loaded: list[dict[str, Any]] = []
        for cell in cells:
            pack = load_one(cell)
            if pack is None:
                continue
            print(
                f"load {cell['name']} n={pack['_n']} half={pack['_half_cov']:.2f}",
                flush=True,
            )
            loaded.append(pack)
        if not loaded:
            continue
        if is_aime and len(loaded) < 3:
            print(f"skip {name} seeds={len(loaded)}/{len(cells)}", flush=True)
            continue
        shown = job_name(name, len(loaded), len(cells))
        pack = accfirst.merge_packs(loaded, shown) if len(loaded) > 1 else loaded[0]
        pack["_half_cov"] = min(p["_half_cov"] for p in loaded)
        nofs_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=False)
        nofs_sum = rg.summarize(nofs_rows)
        fs_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=True)
        fs_sum = rg.summarize(fs_rows)
        half_nofs = sweep_half(pack, nofs_rows, nofs_sum, use_fs=False)
        half_fs = sweep_half(pack, fs_rows, fs_sum, use_fs=True)
        row = (
            f"| {shown} | {fmt_pair(pack['puma_acc'], pack['puma_tok'])} "
            f"| {fmt_pair(nofs_sum['acc'], nofs_sum['tok'])} "
            f"| {fmt_half(half_nofs)} | {fmt_half(half_fs)} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
