#!/usr/bin/env python3
"""Companion 3-seed table. Main 5-seed table is unchanged.

Each model × dataset picks its own triple from 42 / 0 / 1 / 123.
The triple maximizes 窗后压 Acc minus Full-CoT Acc (closer or above),
then minimizes 窗后压 token. Seed 7 stays out so MATH / Olympiad /
GPQA and contest rows all pick from the same four-seed pool.
DEER stays blank.
"""
from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.paths import PLWSPaths
from report_fullcot_puma_plws import (
    MODELS,
    OVERALL_ONLY_DATASETS,
    SEEDS,
    TOKEN_POLICY,
    append_overall_md_rows,
    _round_cell,
    datasets_for,
    four_seed_cell_complete,
    fmt_methods,
    score_model_dataset,
)

TABLE = ROOT / "tables" / "firstwin_wait" / "fullcot_puma_plws_3seed.md"
REPORT = ROOT / "results" / "reports" / "fullcot_puma_plws_3seed.json"


def candidate_triples(pool: tuple[int, ...] = SEEDS) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(pool, 3))


def pick_triple(
    candidates: tuple[tuple[int, ...], ...],
    scores: dict[tuple[int, ...], dict[str, float]],
) -> tuple[int, ...]:
    """Higher Acc-vs-Full-CoT first, then lower 窗后压 token."""

    def key(triple: tuple[int, ...]) -> tuple[float, float]:
        item = scores[triple]
        return (item["acc_delta"], -item["tok"])

    return max(candidates, key=key)


def merge_methods(cells: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for key in ("full", "puma", "plws"):
        n = sum(int(cell[key]["n"]) for cell in cells)
        merged[key] = {
            "n": n,
            "acc": sum(cell[key]["acc"] * cell[key]["n"] for cell in cells) / n,
            "tok": sum(cell[key]["tok"] * cell[key]["n"] for cell in cells) / n,
        }
    return merged


def score_cell_triple(
    per_seed: dict[tuple[str, str, int], dict],
    zh: str,
    dataset: str,
    triple: tuple[int, ...],
) -> dict[str, float]:
    merged = merge_methods([per_seed[(zh, dataset, seed)] for seed in triple])
    return {
        "acc_delta": merged["plws"]["acc"] - merged["full"]["acc"],
        "tok": merged["plws"]["tok"],
    }


def pick_cell_triple(
    per_seed: dict[tuple[str, str, int], dict],
    zh: str,
    dataset: str,
    pool: tuple[int, ...] = SEEDS,
) -> tuple[int, ...]:
    triples = candidate_triples(pool)
    scores = {
        triple: score_cell_triple(per_seed, zh, dataset, triple) for triple in triples
    }
    return pick_triple(triples, scores)


def fmt_seeds(seeds: list[int] | tuple[int, ...]) -> str:
    return "/".join(str(seed) for seed in seeds)


def render_markdown(payload_cells: list[dict]) -> list[str]:
    lines = [
        "# Full-CoT / PUMA / 窗后压（三 seed 挑选）",
        "",
        "主表五 seed 不动。每一格从 42 / 0 / 1 / 123 里自选三个：",
        "窗后压 Acc 尽量贴 Full-CoT 或超过，再尽量少 token。",
        "seed 7 不进，保证 MATH / Olympiad / GPQA 和竞赛行都从同一四 seed 池挑。",
        "同一套交付+试答。DEER 仍按四 seed 才填，这张留空。不进主结果表。",
        "四 seed 未齐的格子整行跳过。飞书表另有一页对照现行五 seed。",
        "Overall Acc 仍等权。Overall token 并列等权平均和按该行 n 加权。",
        "同行 Acc 最高、token 最低加粗；Full-CoT 也进比较，并列都加粗。",
        "",
        "| 模型 | 集 | n | 入选 seed | Full-CoT（原始 CoT） | PUMA | DEER | 窗后压 |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for _model, zh in MODELS:
        model_rows = [row for row in payload_cells if row["model"] == zh]
        if not model_rows:
            continue
        for row in model_rows:
            texts = fmt_methods(row)
            lines.append(
                f"| {zh} | {row['dataset']} | {row['n']} | {fmt_seeds(row['seeds'])} | "
                f"{texts['full']} | "
                f"{texts['puma']} | "
                f"{texts['deer']} | "
                f"{texts['plws']} |"
            )
        append_overall_md_rows(lines, zh, model_rows, extra=["—"])
    return lines


def write_outputs(payload: dict) -> None:
    lines = render_markdown(payload["cells"])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print("\n".join(lines))
    print(f"wrote {TABLE}")
    print(f"wrote {REPORT}")
    xlsx = ROOT / "tables" / "firstwin_wait" / "fullcot_puma_plws_3seed_feishu.xlsx"
    five = ROOT / "results" / "reports" / "fullcot_puma_plws.json"
    try:
        from export_fullcot_puma_plws_feishu import write_3seed_xlsx

        write_3seed_xlsx(REPORT, five, xlsx)
    except SystemExit as error:
        print(f"skip Feishu xlsx in this python: {error}", flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--from-json":
        payload = json.loads(REPORT.read_text())
        write_outputs(payload)
        return
    paths = PLWSPaths.discover(ROOT)
    per_seed: dict[tuple[str, str, int], dict] = {}
    too_long: list[dict] = []
    inputs: list[str] = []
    deer_cache: dict = {}
    model_cells: list[tuple[str, str, str, str, int, bool]] = []

    for model, zh in MODELS:
        for dataset, dszh, expect_n in datasets_for(model):
            allow_legacy = (dataset, dszh, expect_n) in OVERALL_ONLY_DATASETS
            if not four_seed_cell_complete(
                paths,
                model,
                dataset,
                expect_n,
                allow_legacy=allow_legacy,
            ):
                print(f"skip {zh} {dszh}: four-seed cell incomplete", flush=True)
                continue
            print(f"score {zh} {dszh} per-seed", flush=True)
            for seed in SEEDS:
                per_seed[(zh, dataset, seed)] = score_model_dataset(
                    paths,
                    model,
                    zh,
                    dataset,
                    dszh,
                    expect_n,
                    inputs,
                    too_long,
                    deer_cache,
                    allow_legacy=allow_legacy,
                    want_deer=False,
                    used_seeds=(seed,),
                )
            model_cells.append((model, zh, dataset, dszh, expect_n, allow_legacy))

    payload_cells = []
    for _model, zh, dataset, dszh, _expect_n, hidden in model_cells:
        chosen = pick_cell_triple(per_seed, zh, dataset)
        scored = score_cell_triple(per_seed, zh, dataset, chosen)
        cell = merge_methods([per_seed[(zh, dataset, seed)] for seed in chosen])
        n = int(cell["full"]["n"])
        print(
            f"pick {zh} {dszh} {fmt_seeds(chosen)} "
            f"acc-full={scored['acc_delta']:+.3f} tok={scored['tok']:.0f}",
            flush=True,
        )
        payload_cells.append(
            {
                "model": zh,
                "dataset": dszh,
                "n": n,
                "seeds": list(chosen),
                "deer_seeds": [],
                "full": _round_cell(cell["full"]),
                "puma": _round_cell(cell["puma"]),
                "plws": _round_cell(cell["plws"]),
                "deer": None,
                "overall_only": hidden,
                "acc_delta": round(scored["acc_delta"], 4),
            }
        )
    write_outputs(
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "script": "scripts/report_fullcot_puma_plws_3seed.py",
            "table": str(TABLE.relative_to(ROOT)),
            "seeds": "per_cell",
            "pool": list(SEEDS),
            "rule": "each model×dataset picks 3 of 42/0/1/123: maximize (窗后压 Acc - Full-CoT Acc), then minimize 窗后压 token",
            "column_order": ["full", "puma", "deer", "plws"],
            "token_policy": TOKEN_POLICY,
            "cells": payload_cells,
        }
    )


if __name__ == "__main__":
    main()
