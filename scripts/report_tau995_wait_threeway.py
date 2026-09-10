#!/usr/bin/env python3
"""0.995+FS / 0.995+Wait / 0.995+FS+Wait. Wait is PUMA-header only."""
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
import report_conf_fs_stop_margin as cmp

TABLE = AE / "tables/conf_fs_tau995_wait.md"
WAIT = "stop_margin"
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


def load_one(cell: dict[str, Any]) -> dict[str, Any] | None:
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    pack = layers.load_pack(cell)
    for row in pack["scores"].values():
        row.pop("stop_margin", None)
        row.pop("stop_margin_alt", None)
        row.pop("wait_logp", None)
        row.pop("stop_logp", None)
    puma_dirs = tuple(path for path in cell["scores"] if "dense_puma_wait" in str(path))
    if puma_dirs:
        for key, row in rg.load_scores(puma_dirs).items():
            if "stop_margin" not in row:
                continue
            pack["scores"].setdefault(key, {}).update(
                {name: row[name] for name in row if name.startswith("stop") or name.startswith("wait")}
            )
            pack["scores"][key]["stop_margin"] = row["stop_margin"]
    layers.precompute_events(pack, [WAIT])
    pack["_wait_cov"] = cmp.cov(pack, WAIT)
    pack["_cov"] = pack["_wait_cov"]
    return pack


def sweep_wait(pack: dict[str, Any], base_rows: list[dict[str, Any]], base_sum: dict[str, Any], use_fs: bool):
    if pack.get("_wait_cov", 0) < 0.5:
        return None
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
        points.append(rec)
    return accfirst.pick_acc_first_both(points, base_sum["acc"], base_sum["tok"])


def fmt_pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def fmt_rec(rec: dict[str, Any] | None) -> str:
    if rec is None:
        return "读数不够"
    return fmt_pair(rec["acc"], rec["tok"])


def job_name(name: str, loaded: list[dict[str, Any]], cells: list[dict[str, Any]]) -> str:
    if len(cells) == 1 or len(loaded) == len(cells):
        return name
    have = {p["cell"]["name"] for p in loaded}
    miss = []
    for cell in cells:
        if cell["name"] in have:
            continue
        if "-s" in cell["name"]:
            miss.append(cell["name"].rsplit("-s", 1)[-1])
        else:
            miss.append(cell["name"])
    return f"{name}（缺 s{','.join(miss)}）" if miss else name


def main() -> None:
    rg.TAU = 0.995
    header = "| 集 | PUMA | 0.995+强停 | 0.995+Wait | 0.995+强停+Wait |"
    sep = "|---|---|---|---|---|"
    lines = [
        "# PUMA / 0.995+强停 / 0.995+Wait / 0.995+强停+Wait",
        "",
        "交卷：试答即终答。Wait 只读 `dense_puma_wait/`。",
        "四列齐了才进表：AIME 要四 seed，Wait 覆盖不到一半的不放。",
        "Wait 门槛只选一次：相对只开 0.995，正确率不降且 token 不多时先取正确率最高。",
        "0.995+强停+Wait 用同一扇 Wait 门，只是再加上后路强停。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, cells in all_jobs():
        loaded: list[dict[str, Any]] = []
        for cell in cells:
            pack = load_one(cell)
            if pack is None:
                continue
            print(f"load {cell['name']} n={len(pack['questions'])} wait={pack['_wait_cov']:.2f}", flush=True)
            loaded.append(pack)
        if not loaded or len(loaded) != len(cells):
            print(f"skip {name} incomplete seeds={len(loaded)}/{len(cells)}", flush=True)
            continue
        pack = accfirst.merge_packs(loaded, name) if len(loaded) > 1 else loaded[0]
        pack["_wait_cov"] = min(p["_wait_cov"] for p in loaded)
        if pack["_wait_cov"] < 0.5:
            print(f"skip {name} wait={pack['_wait_cov']:.2f}", flush=True)
            continue
        fs_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=True)
        fs_sum = rg.summarize(fs_rows)
        nofs_rows = layers.run_pack(pack, None, float("inf"), "all", use_fs=False)
        nofs_sum = rg.summarize(nofs_rows)
        wait_only = sweep_wait(pack, nofs_rows, nofs_sum, use_fs=False)
        if wait_only is None:
            print(f"skip {name} wait sweep empty", flush=True)
            continue
        wait_fs_rows = layers.run_pack(pack, WAIT, wait_only["threshold"], "all", use_fs=True)
        wait_fs = rg.summarize(wait_fs_rows)
        print(f"same Wait τ={wait_only['threshold']}", flush=True)
        row = (
            f"| {name} | {fmt_pair(pack['puma_acc'], pack['puma_tok'])} "
            f"| {fmt_pair(fs_sum['acc'], fs_sum['tok'])} "
            f"| {fmt_rec(wait_only)} | {fmt_pair(wait_fs['acc'], wait_fs['tok'])} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
