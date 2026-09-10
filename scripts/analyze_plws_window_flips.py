#!/usr/bin/env python3
"""Window rate and answer-flip stats for canonical PLWS cells."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/d/lsj/visual-latent-tts/repos/plws")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.paths import PLWSPaths
from report_fullcot_puma_plws import (
    DATASETS,
    KINDS,
    MODELS,
    SEEDS,
    load_json,
    load_scores,
    puma_delivery_path,
    rec_key,
)

OUT = ROOT / "results/reports/plws_window_flips.json"


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    cells: list[dict] = []
    for model, zh in MODELS:
        for dataset, dszh, expect_n in DATASETS:
            n = 0
            windowed = 0
            unwindowed = 0
            too_long = 0
            flips_up = 0
            flips_down = 0
            keep_wrong = 0
            keep_right = 0
            win_ok = 0
            win_full_ok = 0
            win_puma_ok = 0
            win_tok = 0.0
            win_full_tok = 0.0
            win_puma_tok = 0.0
            un_ok = 0
            un_tok = 0.0
            wait_helps = 0
            will_change = 0
            for seed in SEEDS:
                official_path = paths.puma_statistics_path(model, dataset, seed)
                official = {
                    int(row["question_idx"]): row for row in load_json(official_path)
                }
                delivery = load_json(puma_delivery_path(paths, model, dataset, seed))
                delivery_rows = (
                    delivery if isinstance(delivery, list) else delivery.get("rows", [])
                )
                puma = {int(row["question_idx"]): row for row in delivery_rows}
                jobs: set[int] = set()
                scores: dict[tuple[str, int], dict] = {}
                for kind in KINDS:
                    jobs_path = paths.resolve_jobs_path(
                        model, dataset, seed, kind, k=4, lexicon="core"
                    )
                    if jobs_path.is_file():
                        for line in jobs_path.read_text().splitlines():
                            if not line.strip():
                                continue
                            job = json.loads(line)
                            if job.get("dataset") in (None, dataset):
                                jobs.add(int(job["question_idx"]))
                    scores.update(
                        load_scores(
                            paths.score_dir(
                                model, dataset, seed, kind, k=4, lexicon="core"
                            )
                        )
                    )
                for qid, info in official.items():
                    n += 1
                    rec = scores.get((dataset, int(qid)))
                    full_ok = bool(info.get("original_correct"))
                    full_tok = float(info.get("original_tokens") or 0)
                    prow = puma.get(int(qid), {})
                    if rec is not None:
                        windowed += 1
                        plws_ok = bool(rec.get("new_gold_ok"))
                        plws_tok = float(rec.get("n_think_tok") or 0) + float(
                            rec.get("n_ans_tok") or 0
                        )
                        if rec.get("status") == "too_long":
                            too_long += 1
                        if rec.get("wait_helps"):
                            wait_helps += 1
                        if rec.get("will_change"):
                            will_change += 1
                        if plws_ok and not full_ok:
                            flips_up += 1
                        elif full_ok and not plws_ok:
                            flips_down += 1
                        elif plws_ok:
                            keep_right += 1
                        else:
                            keep_wrong += 1
                        win_ok += int(plws_ok)
                        win_full_ok += int(full_ok)
                        win_puma_ok += int(bool(prow.get("compressed_correct")))
                        win_tok += plws_tok
                        win_full_tok += full_tok
                        win_puma_tok += float(prow.get("compressed_tokens") or 0)
                    elif int(qid) not in jobs:
                        unwindowed += 1
                        un_ok += int(full_ok)
                        un_tok += full_tok
                    else:
                        raise RuntimeError(
                            f"missing PLWS score {model} {dataset} seed={seed} q={qid}"
                        )
            if n != expect_n:
                raise RuntimeError(f"{zh} {dszh} expected {expect_n}, got {n}")
            cells.append(
                {
                    "model": zh,
                    "dataset": dszh,
                    "n": n,
                    "windowed": windowed,
                    "window_rate": 100.0 * windowed / n,
                    "unwindowed": unwindowed,
                    "too_long": too_long,
                    "flips_up": flips_up,
                    "flips_down": flips_down,
                    "net_flips": flips_up - flips_down,
                    "keep_right": keep_right,
                    "keep_wrong": keep_wrong,
                    "wait_helps": wait_helps,
                    "will_change": will_change,
                    "windowed_plws_acc": (100.0 * win_ok / windowed) if windowed else None,
                    "windowed_full_acc": (100.0 * win_full_ok / windowed) if windowed else None,
                    "windowed_puma_acc": (100.0 * win_puma_ok / windowed) if windowed else None,
                    "windowed_plws_tok": (win_tok / windowed) if windowed else None,
                    "windowed_full_tok": (win_full_tok / windowed) if windowed else None,
                    "windowed_puma_tok": (win_puma_tok / windowed) if windowed else None,
                    "unwindowed_acc": (100.0 * un_ok / unwindowed) if unwindowed else None,
                    "unwindowed_tok": (un_tok / unwindowed) if unwindowed else None,
                }
            )
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "script": "scripts/analyze_plws_window_flips.py",
                "cells": cells,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT}")
    for cell in cells:
        print(
            f"{cell['model']:4} {cell['dataset']:14} win={cell['window_rate']:5.1f}% "
            f"up={cell['flips_up']:3} down={cell['flips_down']:3} net={cell['net_flips']:+3} "
            f"win_acc {cell['windowed_full_acc']:.1f}->{cell['windowed_plws_acc']:.1f} "
            f"tok {cell['windowed_full_tok']:.0f}->{cell['windowed_plws_tok']:.0f}"
        )


if __name__ == "__main__":
    main()
