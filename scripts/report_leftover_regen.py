#!/usr/bin/env python3
"""剩窗早切 + 重写 vs 密探k4 重写。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_conf_fs_lag_accfirst as accfirst

TABLE = AE / "tables/leftover_regen.md"
ROOT = AE / "results/leftover_regen"
MODELS = (("7B", "r1_7b"), ("8B", "nemotron_8b"))
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def pair(rows: list[dict[str, Any]]) -> tuple[float, float]:
    n = max(len(rows), 1)
    acc = sum(int(r.get("compressed_correct")) for r in rows) / n
    tok = sum(
        int(r.get("compressed_tokens") or 0) + int(r.get("tokens_trial_answers") or 0)
        for r in rows
    ) / n
    return acc, tok


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    left_p = ROOT / model / dataset / f"s{seed}" / "statistics.json"
    k4_p = dd.regen_stat_path(model, dataset, seed)
    meta_p = ROOT / model / dataset / f"s{seed}" / "candidates_meta.json"
    if not left_p.is_file() or not k4_p.is_file() or not meta_p.is_file():
        return None
    return {
        "left": dd.load_json(left_p),
        "k4": dd.load_json(k4_p),
        "meta": json.loads(meta_p.read_text()),
    }


def main() -> None:
    lines = [
        "# 剩窗早切 + 重写（相对密探k4）",
        "",
        "第一扇非高把握同答窗若早于密探k4 停点，就从这步截断再写终答。其余题沿用密探k4 的重写。",
        "7B / 8B。Acc 对金标。",
        "",
        "| 集 | 密探k4 | 剩窗早切+重写 | Δ | 开火 | 混合/低把握 |",
        "|---|---|---|---|---:|---|",
    ]
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            lefts: list[dict[str, Any]] = []
            k4s: list[dict[str, Any]] = []
            fire = mix = low = 0
            missing = False
            for seed in seeds:
                cell = load_cell(model, dataset, seed)
                if cell is None:
                    missing = True
                    break
                lefts.extend(cell["left"])
                k4s.extend(cell["k4"])
                fire += int(cell["meta"]["n_fire"])
                mix += int(cell["meta"]["n_mix"])
                low += int(cell["meta"]["n_low"])
            name = f"{zh} {ds_zh}"
            if missing:
                lines.append(f"| {name} | — | — | 未齐 | — | — |")
                continue
            k4 = pair(k4s)
            left = pair(lefts)
            lines.append(
                f"| {name} | {accfirst.fmt_pair(*k4)} | {accfirst.fmt_pair(*left)} | "
                f"{rg.fmt_pp(100.0 * (left[0] - k4[0]))} / {rg.fmt_tok(left[1] - k4[1])} | "
                f"{fire} | {mix}/{low} |"
            )
    lines += [
        "",
        "读法：Δ 是 Acc / token，相对密探k4。开火=真正提前重写的题。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
