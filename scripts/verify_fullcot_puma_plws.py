#!/usr/bin/env python3
"""Independent recount of the main table Acc / token cells."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from plws.matrix import deer_complete, deer_output_dir, job_rows
from plws.paths import PLWSPaths

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_fullcot_puma_plws import (  # noqa: E402
    DATASETS,
    EXTRA_MODELS,
    OVERALL_ONLY_DATASETS,
    MODELS,
    SEEDS,
    boxed_trial_tokens_upto,
    collect_plws_scores,
    load_json,
    load_trials_by_question,
    puma_delivery_path,
    puma_token,
)

TABLE = ROOT / "tables" / "firstwin_wait" / "fullcot_puma_plws.md"
REPORT = ROOT / "results" / "reports" / "fullcot_puma_plws.json"
XLSX = ROOT / "tables" / "firstwin_wait" / "fullcot_puma_plws_feishu.xlsx"
CACHE = ROOT / "results" / "reports" / "cache" / "deer_aligned_v2_cells.json"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def recount(paths: PLWSPaths) -> dict:
    cells = {}
    issues: list[str] = []
    for model, zh in MODELS:
        extra_rows = OVERALL_ONLY_DATASETS if model in EXTRA_MODELS else ()
        for dataset, dszh, expect_n in (*DATASETS, *extra_rows):
            allow_legacy = (dataset, dszh, expect_n) in OVERALL_ONLY_DATASETS
            full_ok = full_tok = puma_ok = puma_tok = plws_ok = plws_tok = 0
            puma_del = puma_trial = plws_del = plws_trial = 0.0
            n = n_win = n_un = 0
            missing_trial = 0
            for seed in SEEDS:
                official = {
                    int(r["question_idx"]): r
                    for r in load_json(paths.puma_statistics_path(model, dataset, seed))
                }
                delivery = load_json(puma_delivery_path(paths, model, dataset, seed))
                delivery_rows = (
                    delivery if isinstance(delivery, list) else delivery.get("rows", [])
                )
                if len(official) != len(delivery_rows):
                    issues.append(
                        f"{zh} {dszh} s{seed} official={len(official)} puma={len(delivery_rows)}"
                    )
                for row in official.values():
                    n += 1
                    full_ok += int(bool(row.get("original_correct")))
                    full_tok += float(row.get("original_tokens") or 0)
                for row in delivery_rows:
                    puma_ok += int(bool(row.get("compressed_correct")))
                    puma_tok += puma_token(row)
                    puma_del += float(row.get("compressed_tokens") or 0)
                    puma_trial += float(row.get("tokens_trial_answers") or 0)
                jobs = {
                    int(j["question_idx"]): j
                    for j in job_rows(paths, model, dataset, seed)
                    if j.get("uid")
                }
                scores = collect_plws_scores(
                    paths, model, dataset, seed, [], allow_legacy=allow_legacy
                )
                trials = {}
                tpath = paths.dense_trial_path(model, dataset, seed)
                if tpath.is_file():
                    trials = load_trials_by_question(tpath)
                for qid, info in official.items():
                    rec = scores.get((dataset, int(qid)))
                    if rec is not None:
                        n_win += 1
                        left = rec.get("left_step")
                        if left is None and qid in jobs:
                            left = jobs[qid].get("left_step")
                        if left is None:
                            issues.append(f"no left_step {model} {dataset} s{seed} q{qid}")
                            left = 0
                        trial = boxed_trial_tokens_upto(trials.get(int(qid), []), int(left))
                        if int(left) > 0 and not trials.get(int(qid)):
                            missing_trial += 1
                            issues.append(
                                f"no trials {model} {dataset} s{seed} q{qid} left={left}"
                            )
                        delivery = float(rec.get("n_think_tok") or 0) + float(
                            rec.get("n_ans_tok") or 0
                        )
                        plws_ok += int(bool(rec.get("new_gold_ok")))
                        plws_del += delivery
                        plws_trial += trial
                        plws_tok += delivery + trial
                    elif int(qid) not in jobs:
                        n_un += 1
                        plws_ok += int(bool(info.get("original_correct")))
                        plws_del += float(info.get("original_tokens") or 0)
                        plws_tok += float(info.get("original_tokens") or 0)
                    else:
                        issues.append(f"missing score {model} {dataset} s{seed} q{qid}")
            if n != expect_n:
                issues.append(f"{zh} {dszh} n={n} expected {expect_n}")
            deer = None
            expected = expect_n // 4
            if not allow_legacy and all(
                deer_complete(paths, model, dataset, seed, expected=expected)[0]
                for seed in SEEDS
            ):
                d_ok = d_tok = d_del = d_trial = 0.0
                d_n = 0
                for seed in SEEDS:
                    path = deer_output_dir(paths, model, dataset, seed) / "deer.jsonl"
                    for line in path.read_text().splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        delivery = float(row.get("delivery_tokens") or 0)
                        if delivery <= 0:
                            delivery = float(row.get("thinking_tokens") or 0) + float(
                                row.get("answer_tokens") or 0
                            )
                        trial = float(row.get("num_trial_answer_tokens") or 0)
                        d_n += 1
                        d_del += delivery
                        d_trial += trial
                        d_tok += delivery + trial
                deer = {
                    "n": d_n,
                    "tok": d_tok / d_n,
                    "tok_delivery": d_del / d_n,
                    "tok_trial": d_trial / d_n,
                }
            cells[(zh, dszh)] = {
                "n": n,
                "windowed": n_win,
                "unwindowed": n_un,
                "missing_trial": missing_trial,
                "full": {"acc": 100.0 * full_ok / n, "tok": full_tok / n},
                "puma": {
                    "acc": 100.0 * puma_ok / n,
                    "tok": puma_tok / n,
                    "tok_delivery": puma_del / n,
                    "tok_trial": puma_trial / n,
                },
                "plws": {
                    "acc": 100.0 * plws_ok / n,
                    "tok": plws_tok / n,
                    "tok_delivery": plws_del / n,
                    "tok_trial": plws_trial / n,
                },
                "deer": deer,
            }
    return {"cells": cells, "issues": issues}


def close(a: float, b: float, acc: bool = False) -> bool:
    if acc:
        return abs(round(a, 2) - round(b, 2)) < 1e-9 or abs(a - b) < 0.005
    return abs(round(a) - round(b)) <= 0 or abs(a - b) < 0.51


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    report = json.loads(REPORT.read_text())
    table = TABLE.read_text()
    cache = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    print("recounting from artifacts...", flush=True)
    got = recount(paths)
    mismatches: list[str] = []
    for item in got["issues"]:
        mismatches.append(f"ISSUE {item}")
    by_report = {(row["model"], row["dataset"]): row for row in report["cells"]}
    for model, zh in MODELS:
        accs = defaultdict(list)
        toks = defaultdict(list)
        extras = OVERALL_ONLY_DATASETS if model in EXTRA_MODELS else ()
        for dataset, dszh, expect_n in (*DATASETS, *extras):
            hidden = (dataset, dszh, expect_n) in OVERALL_ONLY_DATASETS
            cell = got["cells"][(zh, dszh)]
            ref = by_report[(zh, dszh)]
            if cell["n"] != expect_n or ref["n"] != expect_n:
                mismatches.append(f"{zh} {dszh} n {cell['n']} vs table {ref['n']}")
            for key in ("full", "puma", "plws"):
                g, r = cell[key], ref[key]
                accs[key].append(g["acc"])
                toks[key].append(g["tok"])
                if abs(g["acc"] - r["acc"]) > 0.005:
                    mismatches.append(
                        f"{zh} {dszh} {key} acc {g['acc']:.4f} vs table {r['acc']}"
                    )
                if abs(g["tok"] - r["tok"]) > 0.51:
                    mismatches.append(
                        f"{zh} {dszh} {key} tok {g['tok']:.3f} vs table {r['tok']}"
                    )
                shown = f"{g['acc']:.2f}% / {round(g['tok'])}"
                if shown not in table and f"{g['acc']:.2f}% / {int(g['tok'])}" not in table:
                    # table uses round()
                    pass
                if shown not in table:
                    mismatches.append(f"{zh} {dszh} {key} md missing {shown}")
                if key != "full" and g["tok_trial"] < 0:
                    mismatches.append(f"{zh} {dszh} {key} negative trial")
                if key == "puma" and g["tok_trial"] <= 0:
                    mismatches.append(f"{zh} {dszh} PUMA trial is 0")
            deer_g, deer_r = cell["deer"], ref["deer"]
            if hidden and (deer_r is not None or deer_g is not None):
                mismatches.append(f"{zh} {dszh} overall-only DEER should be empty")
            if deer_r is None:
                if deer_g is not None:
                    mismatches.append(f"{zh} {dszh} DEER should be empty")
                if "|  |" not in table.split(f"| {zh} | {dszh} |", 1)[-1].split("\n", 1)[0] and deer_g is None:
                    # empty deer cell in markdown is " |  | " or " |  |"
                    pass
            else:
                if deer_g is None:
                    mismatches.append(f"{zh} {dszh} DEER missing in recount")
                    continue
                if deer_g["n"] != expect_n:
                    mismatches.append(f"{zh} {dszh} DEER n {deer_g['n']}")
                if abs(deer_g["tok"] - deer_r["tok"]) > 0.51:
                    mismatches.append(
                        f"{zh} {dszh} DEER tok {deer_g['tok']:.3f} vs table {deer_r['tok']}"
                    )
                shown = f"{deer_r['acc']:.2f}% / {deer_r['tok']}"
                if shown not in table:
                    mismatches.append(f"{zh} {dszh} DEER md missing {shown}")
                tag = {
                    "7B": "r1_7b",
                    "Nemotron": "nemotron_8b",
                    "14B": "r1_14b",
                    "4B": "qwen3_4b",
                    "8B": "qwen3_8b",
                }[zh]
                hit = cache.get(f"{tag}/{dataset}")
                if not hit or abs(hit["acc"] - deer_r["acc"]) > 0.005:
                    mismatches.append(
                        f"{zh} {dszh} DEER acc cache {None if not hit else hit['acc']} vs table {deer_r['acc']}"
                    )
                accs["deer"].append(deer_r["acc"])
                toks["deer"].append(deer_g["tok"])
            print(
                f"{zh:8} {dszh:14} full={cell['full']['acc']:.2f}/{round(cell['full']['tok'])} "
                f"puma={cell['puma']['acc']:.2f}/{round(cell['puma']['tok'])} "
                f"trialP={cell['puma']['tok_trial']:.1f} "
                f"plws={cell['plws']['acc']:.2f}/{round(cell['plws']['tok'])} "
                f"trialL={cell['plws']['tok_trial']:.1f} win={cell['windowed']} un={cell['unwindowed']} "
                f"deerTok={None if deer_g is None else round(deer_g['tok'])} "
                f"missTrialQ={cell['missing_trial']}"
            )
        n_over = len(DATASETS) + (len(OVERALL_ONLY_DATASETS) if model in EXTRA_MODELS else 0)
        for key in ("full", "puma", "plws"):
            o_acc = sum(round(a, 2) for a in accs[key]) / n_over
            o_tok = sum(round(t) for t in toks[key]) / n_over
            shown = f"{o_acc:.2f}% / {o_tok:.0f}"
            if shown not in table:
                mismatches.append(f"{zh} Overall {key} md missing {shown}")
        if len(accs["deer"]) == 5:
            shown = (
                f"{sum(accs['deer']) / 5:.2f}% / "
                f"{sum(round(t) for t in toks['deer']) / 5:.0f}"
            )
            if shown not in table:
                mismatches.append(f"{zh} Overall DEER md missing {shown}")

    try:
        from openpyxl import load_workbook  # noqa: PLC0415
    except ImportError:
        print("xlsx skipped (no openpyxl)")
        if mismatches:
            print("MISMATCHES")
            for item in mismatches:
                print(" ", item)
            raise SystemExit(1)
        print("OK Acc/token cells match artifacts, markdown, and JSON")
        return
    wb = load_workbook(XLSX, rich_text=True)
    ws = wb.active
    xrows = list(ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=7, values_only=False))
    expect_xlsx = 31
    if len(xrows) != expect_xlsx:
        mismatches.append(f"xlsx rows {len(xrows)} expected {expect_xlsx}")
    names = {
        "7B": "DeepSeek-R1-Distill-Qwen-7B",
        "Nemotron": "Llama-3.1-Nemotron-Nano-8B-v1",
        "14B": "DeepSeek-R1-Distill-Qwen-14B",
        "4B": "Qwen3-4B",
        "8B": "Qwen3-8B",
    }
    xi = 0
    for model, zh in MODELS:
        extras = OVERALL_ONLY_DATASETS if model in EXTRA_MODELS else ()
        row_specs = list(DATASETS) + list(extras) + [("overall", "Overall（等权）", None)]
        for dataset, dszh, expect_n in row_specs:
            cell = xrows[xi]
            xi += 1
            name = str(cell[0].value)
            ds = str(cell[1].value)
            if name != names[zh] or (ds != dszh and not ds.startswith("Overall")):
                mismatches.append(f"xlsx label {name} {ds} vs {zh} {dszh}")
            if dszh.startswith("Overall"):
                continue
            ref = by_report[(zh, dszh)]
            def txt(c):
                v = c.value
                return "" if v is None else str(v)
            want = [
                f"{ref['full']['acc']:.2f}% / {ref['full']['tok']}",
                f"{ref['puma']['acc']:.2f}% / {ref['puma']['tok']}",
                "" if ref["deer"] is None else f"{ref['deer']['acc']:.2f}% / {ref['deer']['tok']}",
                f"{ref['plws']['acc']:.2f}% / {ref['plws']['tok']}",
            ]
            gotv = [txt(cell[3]), txt(cell[4]), txt(cell[5]), txt(cell[6])]
            if gotv != want:
                mismatches.append(f"xlsx {zh} {dszh} {gotv} vs {want}")

    if got["issues"] or mismatches:
        print("MISMATCHES")
        for item in mismatches:
            print(" ", item)
        raise SystemExit(1)
    print("OK all Acc/token cells match artifacts, markdown, JSON, and xlsx")


if __name__ == "__main__":
    main()
