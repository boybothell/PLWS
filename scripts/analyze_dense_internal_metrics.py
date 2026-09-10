#!/usr/bin/env python3
"""Dense A_final / G / first-Af slices for training-free internal metrics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq, finite, load_refs  # noqa: E402

SIGNALS = (
    "geo_conf",
    "dola_score",
    "dola_jsd_max",
    "dola_mean_rise",
    "neg_emerge",
    "layer_agree",
    "dola_logp_l8",
    "dola_logp_l14",
    "dola_logp_l20",
    "dola_logp_l27",
    "dola_mean_l27",
    "stop_margin",
    "stop_margin_alt",
    "stop_vs_cont",
    "stop_logp",
    "lookback_ratio",
    "lb_ans",
    "lb_rev",
    "lb_q",
    "lb_recency5",
    "pre_lookback_ratio",
    "neg_eigen_k2",
    "neg_eigen_k4",
    "mid_neg_eigen_k4",
)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) < 8 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr(y: np.ndarray, s: np.ndarray, target: float = 0.05) -> float:
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) < 8 or y.min() == y.max():
        return float("nan")
    fpr, rec, _ = roc_curve(y, s)
    usable = rec[fpr <= target]
    return float(usable[-1]) if len(usable) else 0.0


def load(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        files = [path] if path.is_file() else sorted(path.glob("scores_shard*.jsonl"))
        for file in files:
            for line in file.open():
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        rows.append(row)
    return rows


def report(rows: list[dict[str, Any]], title: str, label: str) -> list[str]:
    lines = [f"## {title}", "", f"n={len(rows)} pos={sum(int(r[label]) for r in rows)}", ""]
    lines.append("| signal | AUROC | 同子集 geo | TPR@5% | geo@5% | pos p50 | neg p50 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    y = np.asarray([int(row[label]) for row in rows])
    g = np.asarray([finite(row.get("geo_conf")) for row in rows])
    for name in SIGNALS:
        s = np.asarray([finite(row.get(name)) for row in rows])
        mask = np.isfinite(s) & np.isfinite(g)
        if mask.sum() < 8 or y[mask].min() == y[mask].max():
            continue
        pos = s[mask & (y == 1)]
        neg = s[mask & (y == 0)]
        lines.append(
            f"| {name} | {auc(y[mask], s[mask]):.3f} | {auc(y[mask], g[mask]):.3f} | "
            f"{tpr(y[mask], s[mask]):.3f} | {tpr(y[mask], g[mask]):.3f} | "
            f"{float(np.median(pos)):.4f} | {float(np.median(neg)):.4f} |"
        )
    lines.append("")
    return lines


def annotate_slices(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["question_idx"])].append(row)
    first_vs_early = []
    first_stay_leave = []
    for seq in grouped.values():
        seq.sort(key=lambda row: int(row["decision_step"]))
        flags = [int(row["is_af"]) for row in seq]
        if not any(flags):
            continue
        first = flags.index(1)
        left = any(flag == 0 for flag in flags[first + 1 :])
        for index, row in enumerate(seq[: first + 1]):
            item = dict(row)
            item["is_first_af"] = int(index == first)
            first_vs_early.append(item)
        stay = dict(seq[first])
        stay["is_stay"] = int(not left)
        first_stay_leave.append(stay)
    return first_vs_early, first_stay_leave


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a_final, gold = load_refs(args.dataset)
    rows = []
    for row in load(args.scores):
        qi = int(row["question_idx"])
        row["is_af"] = int(_fast_eq(row.get("answer"), a_final.get(qi)))
        row["is_gt"] = int(_fast_eq(row.get("answer"), gold.get(qi)))
        if "is_g" not in row:
            row["is_g"] = int(row["is_af"] or row["is_gt"])
        rows.append(row)
    first_vs_early, first_stay_leave = annotate_slices(rows)
    lines = [f"# Dense internal metrics — {args.dataset}", ""]
    lines += report(rows, "全量 dense：试答 ≈ A_final", "is_af")
    lines += report(rows, "全量 dense：G = Af ∨ GT", "is_g")
    if any("is_g_window" in row for row in rows):
        lines += report(rows, "全量 dense：G 窗 k=2", "is_g_window")
    lines += report(first_vs_early, "first Af vs 更早的非 Af", "is_first_af")
    lines += report(first_stay_leave, "first Af：之后不再离开 vs 会离开", "is_stay")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
