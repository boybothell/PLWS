#!/usr/bin/env python3
"""G / G-window AUROC for dense solver/forecast/think-attn scores vs geo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import auc, finite, tpr_at_fpr  # noqa: E402
from analyze_g_window import label_trials  # noqa: E402

SIGNALS = (
    "geo_conf",
    "current_logp",
    "reasoning_pmi",
    "margin",
    "evidence_gain",
    "hist_forget",
    "increment_nll",
    "neg_increment_nll",
    "wrapup_nll",
    "neg_wrapup_nll",
    "forecast_p_keep",
    "last_neg_entropy",
    "last_neg_norm_entropy",
    "last_top10",
    "neg_challenge_changed",
    "prune_same_k50",
)


def load_jsonl_dir(path: Path) -> list[dict[str, Any]]:
    rows = []
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for file in files:
        with file.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def eval_label(rows: list[dict[str, Any]], signal: str, label: str) -> dict[str, float]:
    y = np.asarray([int(row[label]) for row in rows], dtype=int)
    s = np.asarray([finite(row.get(signal)) for row in rows], dtype=float)
    g = np.asarray([finite(row.get("geo_conf")) for row in rows], dtype=float)
    mask = np.isfinite(s) & np.isfinite(g)
    if mask.sum() < 8 or y[mask].min() == y[mask].max():
        return {}
    pos = s[mask & (y == 1)]
    neg = s[mask & (y == 0)]
    return {
        "n": float(mask.sum()),
        "n_pos": float(y[mask].sum()),
        "auroc": auc(y[mask], s[mask]),
        "geo": auc(y[mask], g[mask]),
        "tpr1": tpr_at_fpr(y[mask], s[mask], 0.01),
        "geo1": tpr_at_fpr(y[mask], g[mask], 0.01),
        "tpr5": tpr_at_fpr(y[mask], s[mask], 0.05),
        "geo5": tpr_at_fpr(y[mask], g[mask], 0.05),
        "pos_p50": float(np.median(pos)) if len(pos) else float("nan"),
        "neg_p50": float(np.median(neg)) if len(neg) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = label_trials(args.dataset)
    rows = []
    for path in args.scores:
        for row in load_jsonl_dir(path):
            if row.get("status") not in {None, "ok"}:
                continue
            lab = labels.get((int(row["question_idx"]), int(row["decision_step"])))
            if lab is None:
                continue
            slim = {key: value for key, value in row.items() if key != "reasoning_prefix"}
            slim["is_g"] = lab["is_g"]
            slim["is_g_window"] = lab["is_g_window"]
            slim["geo_conf"] = finite(row.get("geo_conf", lab["geo_conf_trial"]))
            if "forecast_p_change" in row:
                slim["forecast_p_keep"] = 1.0 - finite(row.get("forecast_p_change"))
            if row.get("challenge_changed") is not None:
                slim["neg_challenge_changed"] = 1.0 - float(row["challenge_changed"])
            rows.append(slim)
    lines = [
        f"# Dense remaining probes — {args.dataset}",
        "",
        f"n={len(rows)} G={sum(r['is_g'] for r in rows)} G窗={sum(r['is_g_window'] for r in rows)}",
        "",
        "| 标签 | signal | n / 正 | AUROC | 同子集 geo | TPR@1% | geo@1% | TPR@5% | geo@5% | 正 p50 | 负 p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, tag in (("is_g", "G"), ("is_g_window", "G窗")):
        for name in SIGNALS:
            metrics = eval_label(rows, name, label)
            if not metrics:
                continue
            lines.append(
                f"| {tag} | {name} | {int(metrics['n'])} / {int(metrics['n_pos'])} | "
                f"{metrics['auroc']:.3f} | {metrics['geo']:.3f} | "
                f"{metrics['tpr1']:.3f} | {metrics['geo1']:.3f} | "
                f"{metrics['tpr5']:.3f} | {metrics['geo5']:.3f} | "
                f"{metrics['pos_p50']:.4f} | {metrics['neg_p50']:.4f} |"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
