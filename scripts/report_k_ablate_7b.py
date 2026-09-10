#!/usr/bin/env python3
"""7B first-window k ablation: Full-CoT vs leftover suppress at k=2/3/4/5/6."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plws.artifacts import load_jsonl  # noqa: E402
from plws.matrix import FIRSTWIN, TIER_KINDS  # noqa: E402
from plws.paths import PLWSPaths  # noqa: E402
from report_fullcot_puma_plws import (  # noqa: E402
    SHARD_RE,
    boxed_trial_tokens_upto,
    load_json,
    load_trials_by_question,
    rec_key,
)

TABLE = ROOT / "tables" / "firstwin_wait" / "k_ablate_7b.md"
REPORT = ROOT / "results" / "reports" / "k_ablate_7b.json"
MODEL = "r1_7b"
KS = (2, 3, 4, 5, 6)
SEEDS = (42, 0, 1, 123)
DATASETS = (
    ("math-500", "MATH", 2000),
    ("olympiadbench", "OlympiadBench", 2700),
    ("gpqa-diamond", "GPQA-Diamond", 792),
    ("aime24", "AIME24", 120),
    ("aime25", "AIME25", 120),
)
K_KEYS = tuple(f"k{k}" for k in KS)
TOKEN_POLICY = "delivery_plus_boxed_trial_v1"
EXPECTED_WINDOWED = {2: 5286, 3: 5259, 4: 5178, 5: 5010, 6: 4802}
MAIN_TABLE_K4 = {
    "MATH": {"acc": 92.00, "tok": 3228},
    "OlympiadBench": {"acc": 56.78, "tok": 7175},
    "GPQA-Diamond": {"acc": 47.98, "tok": 5682},
    "AIME24": {"acc": 51.67, "tok": 11185},
    "AIME25": {"acc": 40.83, "tok": 12269},
}


def kinds_for(k: int) -> tuple[str, ...]:
    return (FIRSTWIN,) if k == 4 else TIER_KINDS


def load_k_scores(folder: Path, *, require_v2: bool) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.iterdir()):
        if not SHARD_RE.match(path.name):
            continue
        for rec in load_jsonl(path):
            if rec.get("status") not in {"ok", "too_long"} or not rec.get("uid"):
                continue
            if require_v2 and rec.get("protocol_id") != "puma-fullcot-32k-v2":
                continue
            key = rec_key(rec)
            if key is not None:
                out[key] = rec
    return out


def load_k_jobs(
    paths: PLWSPaths, dataset: str, seed: int, k: int
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for kind in kinds_for(k):
        path = paths.jobs_path(MODEL, dataset, seed, kind, k=k, lexicon="core")
        if not path.is_file():
            continue
        for job in load_jsonl(path):
            if not job.get("uid"):
                continue
            qid = int(job["question_idx"])
            if qid in out:
                raise RuntimeError(
                    f"duplicate job k={k} {dataset} seed={seed} q={qid}"
                )
            out[qid] = job
    return out


def load_k_cell_scores(
    paths: PLWSPaths, dataset: str, seed: int, k: int
) -> dict[tuple[str, int], dict]:
    scores: dict[tuple[str, int], dict] = {}
    for kind in kinds_for(k):
        loaded = load_k_scores(
            paths.score_dir(MODEL, dataset, seed, kind, k=k, lexicon="core"),
            require_v2=(k == 4),
        )
        overlap = set(scores) & set(loaded)
        if overlap:
            raise RuntimeError(
                f"duplicate score k={k} {dataset} seed={seed} n={len(overlap)}"
            )
        scores.update(loaded)
    return scores


def fmt_cell(acc: float, tok: float) -> str:
    return f"{acc:.2f}% / {tok:.0f}"


def md_cell(acc: float, tok: float, acc_win: bool, tok_win: bool) -> str:
    acc_s = f"{acc:.2f}%"
    tok_s = f"{tok:.0f}"
    if acc_win:
        acc_s = f"**{acc_s}**"
    if tok_win:
        tok_s = f"**{tok_s}**"
    return f"{acc_s} / {tok_s}"


def winners(cells: dict) -> tuple[set[str], set[str]]:
    accs = {key: cells[key]["acc"] for key in K_KEYS}
    toks = {key: cells[key]["tok"] for key in K_KEYS}
    max_acc = max(accs.values())
    min_tok = min(toks.values())
    return (
        {key for key, value in accs.items() if value == max_acc},
        {key for key, value in toks.items() if value == min_tok},
    )


def round_cell(cell: dict) -> dict:
    return {"acc": round(cell["acc"], 2), "tok": round(cell["tok"])}


def main() -> None:
    paths = PLWSPaths.discover(ROOT)
    cells: list[dict] = []

    for dataset, dszh, expect_n in DATASETS:
        full_ok = full_tok = 0.0
        n = 0
        by_k: dict[int, dict[str, float]] = {
            k: {"ok": 0.0, "tok": 0.0, "win": 0, "un": 0} for k in KS
        }
        for seed in SEEDS:
            official_path = paths.puma_statistics_path(MODEL, dataset, seed)
            if not official_path.is_file():
                raise FileNotFoundError(official_path)
            official = {
                int(row["question_idx"]): row for row in load_json(official_path)
            }
            trials_path = paths.dense_trial_path(MODEL, dataset, seed)
            if not trials_path.is_file():
                raise FileNotFoundError(trials_path)
            trials = load_trials_by_question(trials_path)
            jobs_by_k = {
                k: load_k_jobs(paths, dataset, seed, k) for k in KS
            }
            scores_by_k = {
                k: load_k_cell_scores(paths, dataset, seed, k) for k in KS
            }
            for qid, info in official.items():
                n += 1
                full_ok += int(bool(info.get("original_correct")))
                full_tok += float(info.get("original_tokens") or 0)
                for k in KS:
                    rec = scores_by_k[k].get((dataset, int(qid)))
                    job = jobs_by_k[k].get(int(qid))
                    if rec is not None:
                        left = rec.get("left_step")
                        if left is None and job is not None:
                            left = job.get("left_step")
                        if left is None:
                            raise RuntimeError(
                                f"missing left_step k={k} {dataset} "
                                f"seed={seed} q={qid}"
                            )
                        by_k[k]["win"] += 1
                        by_k[k]["ok"] += int(bool(rec.get("new_gold_ok")))
                        by_k[k]["tok"] += (
                            float(rec.get("n_think_tok") or 0)
                            + float(rec.get("n_ans_tok") or 0)
                            + boxed_trial_tokens_upto(
                                trials.get(int(qid), []), int(left)
                            )
                        )
                    elif job is None:
                        by_k[k]["un"] += 1
                        by_k[k]["ok"] += int(bool(info.get("original_correct")))
                        by_k[k]["tok"] += float(info.get("original_tokens") or 0)
                    else:
                        raise RuntimeError(
                            f"missing score k={k} {dataset} seed={seed} q={qid}"
                        )
        if n != expect_n:
            raise RuntimeError(f"{dszh} n={n} expected {expect_n}")
        row = {
            "model": "7B",
            "dataset": dszh,
            "n": n,
            "full": {"acc": 100.0 * full_ok / n, "tok": full_tok / n},
        }
        for k in KS:
            row[f"k{k}"] = {
                "acc": 100.0 * by_k[k]["ok"] / n,
                "tok": by_k[k]["tok"] / n,
                "windowed": int(by_k[k]["win"]),
                "unwindowed": int(by_k[k]["un"]),
            }
        cells.append(row)

    for k in KS:
        windowed = sum(row[f"k{k}"]["windowed"] for row in cells)
        expect = EXPECTED_WINDOWED[k]
        if windowed != expect:
            raise RuntimeError(f"k={k} windowed={windowed} expected {expect}")

    shown = {key: [] for key in ("full", *K_KEYS)}
    display_cells = []
    for row in cells:
        shown_row = {
            "model": row["model"],
            "dataset": row["dataset"],
            "n": row["n"],
            "full": round_cell(row["full"]),
        }
        for key in K_KEYS:
            shown_row[key] = {
                **round_cell(row[key]),
                "windowed": row[key]["windowed"],
                "unwindowed": row[key]["unwindowed"],
            }
            shown[key].append((shown_row[key]["acc"], shown_row[key]["tok"]))
        shown["full"].append((shown_row["full"]["acc"], shown_row["full"]["tok"]))
        expect_k4 = MAIN_TABLE_K4[shown_row["dataset"]]
        if shown_row["k4"]["acc"] != expect_k4["acc"] or shown_row["k4"]["tok"] != expect_k4["tok"]:
            raise RuntimeError(
                f"k=4 {shown_row['dataset']} "
                f"{shown_row['k4']} != main table {expect_k4}"
            )
        display_cells.append(shown_row)

    overall = {"model": "7B", "dataset": "Overall（等权）", "n": None}
    for key in shown:
        accs, toks = zip(*shown[key])
        overall[key] = {
            "acc": sum(accs) / 5,
            "tok": sum(toks) / 5,
        }
    display_overall = {
        "model": "7B",
        "dataset": "Overall（等权）",
        "n": None,
        "full": round_cell(overall["full"]),
        **{key: round_cell(overall[key]) for key in K_KEYS},
    }

    method_rows = [
        ("Full-CoT", "full"),
        ("k=2", "k2"),
        ("k=3", "k3"),
        ("k=4", "k4"),
        ("k=5", "k5"),
        ("k=6", "k6"),
    ]
    all_rows = [*display_cells, display_overall]
    acc_w, tok_w = winners(display_overall)
    lines = [
        "# 7B 第一扇窗 k 消融",
        "",
        "DeepSeek-R1-Distill-Qwen-7B。四 seed：0 / 1 / 42 / 123。",
        "官方题数；该 k 没有第一扇窗的题贴 Full-CoT。",
        "每格是 Acc / token。Token = 交付思考+终答 + 窗末及之前 boxed 试答。",
        "k=4 是主表窗后压（`puma-fullcot-32k-v2`）。k=2/3/5/6 是旧 32768 入口的 leftover。",
        "只在五个 k 之间标黑。Full-CoT 只作对照。Overall 是五集先按格四舍五入再等权平均。",
        "这张是消融，可以进论文消融节；不进主结果表。",
        "",
        "- **Acc**：五个 k 里最高",
        "- **token**：五个 k 里最少",
        "",
        "| Method | MATH | OlympiadBench | GPQA-Diamond | AIME24 | AIME25 | Avg |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, key in method_rows:
        bits = [f"| {name}"]
        for row in all_rows:
            acc_win = key in acc_w and key != "full"
            tok_win = key in tok_w and key != "full"
            # Bold per-dataset using that row's winners, Avg using overall.
            if row is not display_overall:
                row_acc_w, row_tok_w = winners(row)
                acc_win = key in row_acc_w
                tok_win = key in row_tok_w
            if key == "full":
                bits.append(f" | {fmt_cell(row[key]['acc'], row[key]['tok'])}")
            else:
                bits.append(
                    " | "
                    + md_cell(row[key]["acc"], row[key]["tok"], acc_win, tok_win)
                )
        lines.append("".join(bits) + " |")

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "script": "scripts/report_k_ablate_7b.py",
                "table": str(TABLE.relative_to(ROOT)),
                "model": "r1_7b",
                "seeds": list(SEEDS),
                "ks": list(KS),
                "token_policy": TOKEN_POLICY,
                "cells": display_cells + [display_overall],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {TABLE}")
    print(f"wrote {REPORT}")
    for row in display_cells + [display_overall]:
        n_s = "—" if row["n"] is None else row["n"]
        print(
            f"{row['dataset']:16} n={n_s}  "
            + "  ".join(
                f"{key}={fmt_cell(row[key]['acc'], row[key]['tok'])}"
                for key in ("full", *K_KEYS)
            )
        )


if __name__ == "__main__":
    main()
