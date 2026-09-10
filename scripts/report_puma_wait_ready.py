#!/usr/bin/env python3
"""0.995+FS vs 0.995+FS+Wait on sets that already have PUMA-header Wait."""
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

TABLE = AE / "tables/conf_fs_puma_wait_ready.md"
WAIT = "stop_margin"


def ready_jobs() -> list[tuple[str, list[dict[str, Any]]]]:
    out: list[tuple[str, list[dict[str, Any]]]] = []
    out.append(
        (
            "7B MATH",
            [jobs.make_cell("7B", "r1_7b", "dense_G_r1_7b", "puma_offline_r1_7b", "MATH", "math-500")],
        )
    )
    for ds_zh, dataset in (("奥赛", "olympiadbench"), ("GPQA", "gpqa-diamond")):
        out.append(
            (
                f"Qwen3-4B {ds_zh}",
                [jobs.make_cell("4B", "qwen3_4b", "dense_G_qwen3_4b", "puma_offline_qwen3_4b", ds_zh, dataset)],
            )
        )
    out.append(
        (
            "Qwen3-4B AIME24（缺 s123）",
            [
                jobs.make_cell("4B", "qwen3_4b", "dense_G_qwen3_4b", "puma_offline_qwen3_4b", "AIME24", "aime24", seed)
                for seed in (42, 0, 1)
            ],
        )
    )
    out.append(
        (
            "Qwen3-8B GPQA",
            [jobs.make_cell("8B", "qwen3_8b", "dense_G_qwen3_8b", "puma_offline_qwen3_8b", "GPQA", "gpqa-diamond")],
        )
    )
    out.append(
        (
            "Qwen3-8B AIME24",
            [
                jobs.make_cell("8B", "qwen3_8b", "dense_G_qwen3_8b", "puma_offline_qwen3_8b", "AIME24", "aime24", seed)
                for seed in (42, 0, 1, 123)
            ],
        )
    )
    out.append(
        (
            "Qwen3-8B AIME25（缺 s42）",
            [
                jobs.make_cell("8B", "qwen3_8b", "dense_G_qwen3_8b", "puma_offline_qwen3_8b", "AIME25", "aime25", seed)
                for seed in (0, 1, 123)
            ],
        )
    )
    return out


def load_pack(cell: dict[str, Any]) -> dict[str, Any] | None:
    if not cell["trial"].is_file() or not cell["stat"].is_file() or not cell["gpath"].is_file():
        return None
    if not any(Path(p).is_dir() for p in cell.get("scores") or ()):
        return None
    pack = layers.load_pack(cell)
    layers.precompute_events(pack, [WAIT])
    pack["_wait_cov"] = cmp.cov(pack, WAIT)
    return pack


def fmt_pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    rg.TAU = 0.995
    header = "| 集 | PUMA | 0.995+强停 | 0.995+强停+Wait | Wait 相对只开置信度 | Wait 相对 PUMA |"
    sep = "|---|---|---|---|---|---|"
    lines = [
        "# 0.995+强停 vs 0.995+强停+Wait（只含已齐的 PUMA 题头 Wait）",
        "",
        "交卷：试答即终答。Wait 只读 `dense_puma_wait/`。",
        "滞后门槛相对 0.995+强停自选：正确率不降且 token 不多时，先取正确率最高，再少 token。",
        "AIME 缺 seed 的不四 seed 加权，表里已标明。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for name, cells in ready_jobs():
        parts = []
        for cell in cells:
            pack = load_pack(cell)
            if pack is None:
                print(f"skip {cell['name']} missing", flush=True)
                continue
            print(
                f"load {cell['name']} n={len(pack['questions'])} wait={pack['_wait_cov']:.2f}",
                flush=True,
            )
            pack["_cov"] = pack["_wait_cov"]
            parts.append(pack)
        if not parts or len(parts) != len(cells):
            print(f"| {name} | 缺文件 |", flush=True)
            continue
        pack = accfirst.merge_packs(parts, name) if len(parts) > 1 else parts[0]
        pack["_wait_cov"] = min(p["_wait_cov"] for p in parts)
        fs_rows = layers.run_pack(pack, None, float("inf"), "all")
        fs_sum = rg.summarize(fs_rows)
        wait = cmp.sweep_sig(pack, fs_rows, fs_sum, WAIT)
        if wait is None:
            wait_s = "读数不够"
            vs_fs = "—"
            vs_puma = "—"
        else:
            wait_s = fmt_pair(wait["acc"], wait["tok"])
            vs_fs = f"{rg.fmt_pp(100.0 * (wait['acc'] - fs_sum['acc']))} / {rg.fmt_tok(wait['tok'] - fs_sum['tok'])}"
            vs_puma = f"{rg.fmt_pp(wait['d_acc_pp_puma'])} / {rg.fmt_tok(wait['d_tok_puma'])}"
        row = (
            f"| {name} | {fmt_pair(pack['puma_acc'], pack['puma_tok'])} "
            f"| {fmt_pair(fs_sum['acc'], fs_sum['tok'])} "
            f"| {wait_s} | {vs_fs} | {vs_puma} |"
        )
        print(row, flush=True)
        lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
