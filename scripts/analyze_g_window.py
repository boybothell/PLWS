#!/usr/bin/env python3
"""Dense AUROC for G and G-window (k=2) on already-extracted signals."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq, auc, finite, load_refs, load_v1, tpr_at_fpr  # noqa: E402
from analyze_confcal_v2_available import load_jsonl  # noqa: E402

DATASETS = ("math-500", "olympiadbench", "gpqa-diamond")
SIGNALS = (
    "geo_conf",
    "traj_ema",
    "direct_yes",
    "verbal_h3",
    "h4_neg_entropy",
    "h4_margin",
    "ha_p_small_boxed",
    "hb_base_answer",
    "hc_base",
)
TABLE = AE / "tables/probe_g_window.md"


def load_trials(dataset: str) -> dict[int, list[tuple[int, str, float]]]:
    grouped: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    path = AE / f"results/dense_G_r1_7b/{dataset}/dense_puma/trial_answers.json"
    for row in json.loads(path.read_text()):
        grouped[int(row["question_idx"])].append(
            (int(row["stopped_len"]), str(row.get("final_answer") or ""), finite(row.get("confidence")))
        )
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
    return grouped


def label_trials(dataset: str) -> dict[tuple[int, int], dict[str, Any]]:
    a_final, gold = load_refs(dataset)
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for qi, seq in load_trials(dataset).items():
        flags = [_fast_eq(answer, a_final.get(qi)) or _fast_eq(answer, gold.get(qi)) for _, answer, _ in seq]
        ema = float("nan")
        prev = ""
        for index, ((step, answer, conf), is_g) in enumerate(zip(seq, flags, strict=True)):
            changed = bool(prev and not _fast_eq(answer, prev))
            if not math.isfinite(ema):
                ema = conf
            elif changed:
                ema = 0.25 * ema + 0.75 * conf
            else:
                ema = 0.70 * conf + 0.30 * ema
            neighbor = (index > 0 and flags[index - 1]) or (index + 1 < len(flags) and flags[index + 1])
            out[(qi, step)] = {
                "is_g": int(is_g),
                "is_g_window": int(is_g and neighbor),
                "traj_ema": ema,
                "geo_conf_trial": conf,
                "answer": answer,
            }
            prev = answer
    return out


def attach_scores(dataset: str, labels: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    v1 = load_v1(dataset)
    key = load_jsonl(AE / "results/confcal_judge/v2/keytoken" / f"{dataset}_s42")
    base = load_jsonl(AE / "results/confcal_judge/v2/universal_v2b" / f"{dataset}_s42_base")
    rows: list[dict[str, Any]] = []
    keys = set(v1) | set(key) | set(base)
    for qi, step, answer in keys:
        lab = labels.get((qi, step))
        if lab is None:
            continue
        v = v1.get((qi, step, answer), {})
        k = key.get((qi, step, answer), {})
        b = base.get((qi, step, answer), {})
        h4 = v.get("direct_h4") or {}
        verbal = (v.get("verbal_h3") or {}).get("value")
        summary = k.get("keytoken") or {}
        rows.append(
            {
                "dataset": dataset,
                "question_idx": qi,
                "decision_step": step,
                "is_g": lab["is_g"],
                "is_g_window": lab["is_g_window"],
                "geo_conf": finite(v.get("geo_conf", lab["geo_conf_trial"])),
                "traj_ema": lab["traj_ema"],
                "direct_yes": finite(h4.get("yes")),
                "verbal_h3": finite(verbal) / 100.0 if verbal is not None else float("nan"),
                "h4_neg_entropy": -finite(h4.get("vocab_entropy")),
                "h4_margin": finite(h4.get("margin")),
                "ha_p_small_boxed": finite(summary.get("p_small_boxed")),
                "hb_base_answer": finite((b.get("hb_answer") or {}).get("per_byte", b.get("text_logprob_per_byte"))),
                "hc_base": finite(b.get("candidate_prob")),
            }
        )
    return rows


def eval_label(rows: list[dict[str, Any]], signal: str, label: str) -> dict[str, float]:
    sub = [row for row in rows if math.isfinite(row.get(signal, float("nan")))]
    if len(sub) < 8:
        return {}
    y = np.asarray([int(row[label]) for row in sub], dtype=int)
    s = np.asarray([row[signal] for row in sub], dtype=float)
    g = np.asarray([row["geo_conf"] for row in sub], dtype=float)
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
    lines = [
        "# Dense G vs G-window (k=2)",
        "",
        "同一套已有分数，只换正类。G 窗 = 连续 ≥2 步都是 G。孤立 G 算负类。",
        "对照 `geo_conf` 始终在同一子集上。标签用 fast_eq（与 v1 分析一致）。",
        "",
    ]
    report: dict[str, Any] = {}
    for dataset in DATASETS:
        print(f"{dataset}: labeling", flush=True)
        labels = label_trials(dataset)
        rows = attach_scores(dataset, labels)
        n = len(rows)
        n_g = sum(row["is_g"] for row in rows)
        n_w = sum(row["is_g_window"] for row in rows)
        n_iso = n_g - n_w
        report[dataset] = {"n": n, "n_g": n_g, "n_window": n_w, "n_iso": n_iso, "signals": {}}
        lines += [
            f"## {dataset}（评分行 n={n}，G={n_g}，G窗={n_w}，孤立G={n_iso}）",
            "",
            "| 标签 | signal | n / 正 | AUROC | 同子集 geo | TPR@1% | geo@1% | TPR@5% | geo@5% | 正 p50 | 负 p50 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        print(f"{dataset}: n={n} G={n_g} window={n_w} iso={n_iso}", flush=True)
        for label, tag in (("is_g", "G"), ("is_g_window", "G窗")):
            for name in SIGNALS:
                metrics = eval_label(rows, name, label)
                if not metrics:
                    continue
                report[dataset]["signals"][f"{tag}:{name}"] = metrics
                lines.append(
                    f"| {tag} | {name} | {int(metrics['n'])} / {int(metrics['n_pos'])} | "
                    f"{metrics['auroc']:.3f} | {metrics['geo']:.3f} | "
                    f"{metrics['tpr1']:.3f} | {metrics['geo1']:.3f} | "
                    f"{metrics['tpr5']:.3f} | {metrics['geo5']:.3f} | "
                    f"{metrics['pos_p50']:.4f} | {metrics['neg_p50']:.4f} |"
                )
        lines.append("")
    TABLE.write_text("\n".join(lines) + "\n")
    (AE / "results/confcal_judge/v2/analysis_g_window.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
