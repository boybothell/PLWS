#!/usr/bin/env python3
"""Same-set best rel-rise door on dense+regen host. Gold Acc."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import analyze_lag_layers as layers
import analyze_lag_more_layers as more
import replay_rescue_R_gate as rg
import report_brightest_stop_auroc as jobs
import report_conf_fs_lag_accfirst as accfirst
import report_dense_regen_default_door as door
import report_puma_plus_r_oracle as base

TABLE = AE / "tables/dense_regen_sameset_door.md"


def ratio_xs(pack: dict[str, Any]) -> list[float]:
    xs = []
    for q in pack["questions"]:
        host_step = int(q.get("host_step") or 10**9)
        for ev in q["events"]:
            if ev.get("high") or ev.get("mixed"):
                continue
            step = int(q["rows"][ev["end"]]["stopped_len"])
            if step < rg.MSS or step >= host_step:
                continue
            rise = base.last_val(ev["vals"].get(door.SIG))
            exit_s = base.last_val(ev["vals"].get("last_mean_logp"))
            if rise != rise or exit_s != exit_s:
                continue
            xs.append(rise / (abs(exit_s) + 1e-6))
    return xs


def sweep(pack: dict[str, Any]) -> dict[str, Any] | None:
    frozen = door.eval_pack(pack, rg.REL_RISE_THR)
    if frozen is None:
        return None
    thrs = [float("inf")]
    xs = ratio_xs(pack)
    if xs:
        thrs.extend(rg.quantiles(xs, n=41))
    points = []
    for thr in thrs:
        rec = door.eval_pack(pack, thr)
        if rec is None:
            continue
        points.append(rec)
    pick = accfirst.pick_acc_first_both(points, frozen["host_acc"], frozen["host_tok"])
    if pick is None:
        pick = next(p for p in points if p["threshold"] == float("inf"))
    pick["frozen"] = frozen
    return pick


def fmt_pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def fmt_pick(rec: dict[str, Any]) -> str:
    core = fmt_pair(rec["acc"], rec["tok"])
    if rec["threshold"] == float("inf"):
        return f"{core}（不开）"
    return f"{core}（门 {rec['threshold']:.3f}）"


def load_named(zh: str, tag: str, gdir: str, puma: str, ds_zh: str, dataset: str, is_aime: bool):
    if not is_aime:
        pack = door.load_one(zh, tag, gdir, puma, ds_zh, dataset, None)
        if pack is None:
            return None
        print(f"load {zh} {ds_zh} n={len(pack['questions'])} half={pack['_half_cov']:.2f}", flush=True)
        return pack
    packs = []
    for seed in jobs.SEEDS:
        pack = door.load_one(zh, tag, gdir, puma, ds_zh, dataset, seed)
        if pack is None:
            continue
        print(
            f"load {zh} {ds_zh}-s{seed} n={len(pack['questions'])} half={pack['_half_cov']:.2f}",
            flush=True,
        )
        packs.append(pack)
    if len(packs) != 4:
        print(f"skip {zh} {ds_zh} seeds={len(packs)}/4", flush=True)
        return None
    pack = accfirst.merge_packs(packs, f"{zh} {ds_zh}")
    pack["_half_cov"] = min(p["_half_cov"] for p in packs)
    pack["_cov"] = pack["_half_cov"]
    return pack


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    orig = layers.enrich
    layers.enrich = lambda scores: more.enrich_all(orig(scores))
    header = (
        "| 集 | 官方 PUMA | 密探+重写 | 固定 2.85 | 同集最好 | 同集比重写 | 开火 / 救回 / 伤 |"
    )
    sep = "|---|---|---|---|---|---|---:|"
    lines = [
        "# 密探+重写 + 对比率门同集最好（正确率只对金标）",
        "",
        "宿主是已存密探双闸门：k=4、0.995、后路强停，截断后再重写终答。",
        "附加门仍是低置信连答窗最后一步的对比率，必须在密探宿主停点之前，交试答。",
        "同集门槛：相对「只做密探+重写」，正确率不降且 token 不多时先取正确率最高。偷看本集对错，换集会塌。",
        "不开 = 本集最好是一扇都不放。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for zh, tag, gdir, puma in door.MODELS:
        for ds_zh, dataset, is_aime in door.DS:
            pack = load_named(zh, tag, gdir, puma, ds_zh, dataset, is_aime)
            if pack is None:
                continue
            rec = sweep(pack)
            if rec is None:
                print(f"| {zh} {ds_zh} | 未齐 |", flush=True)
                continue
            frozen = rec["frozen"]
            row = (
                f"| {zh} {ds_zh} | {fmt_pair(frozen['puma_acc'], frozen['puma_tok'])} "
                f"| {fmt_pair(frozen['host_acc'], frozen['host_tok'])} "
                f"| {fmt_pair(frozen['acc'], frozen['tok'])} "
                f"| {fmt_pick(rec)} "
                f"| {rg.fmt_pp(rec['d_host_acc'])} / {rg.fmt_tok(rec['d_host_tok'])} "
                f"| {rec['n_fire']} / {rec['n_gain']} / {rec['n_hurt']} |"
            )
            print(row, flush=True)
            lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
