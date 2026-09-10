#!/usr/bin/env python3
"""Full-dense A_final discrimination: hidden / logit-lens vs geo."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq, finite, load_refs  # noqa: E402

SIGNALS = (
    "geo_conf",
    "last_mean_logp",
    "last_min_logp",
    "neg_ans_entropy",
    "hidden_norm",
    "prefix_ans_cos",
    "hidden_nn_cos",
    "lens_l8",
    "lens_l14",
    "lens_l20",
    "lens_l27",
    "lens_best",
    "lens_rise",
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
        files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
        for file in files:
            for line in file.open():
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        rows.append(row)
    return rows


def report(rows: list[dict[str, Any]], title: str) -> list[str]:
    lines = [f"## {title}", "", f"n={len(rows)} Af={sum(int(r['is_af']) for r in rows)}", ""]
    lines.append("| signal | AUROC | 同子集 geo | TPR@5% | geo@5% | Af p50 | 非Af p50 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    y = np.asarray([int(row["is_af"]) for row in rows])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a_final, _ = load_refs(args.dataset)
    rows = []
    for row in load(args.scores):
        row["is_af"] = int(_fast_eq(row.get("answer"), a_final.get(int(row["question_idx"]))))
        rows.append(row)
    lines = [f"# Dense A_final — hidden / logit lens — {args.dataset}", ""]
    lines += report(rows, "全量 dense：试答 ≈ A_final")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
