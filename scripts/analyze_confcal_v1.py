#!/usr/bin/env python3
"""G vs non-G discrimination for v1 Direct / H3 / H4 on MATH, OLP, GPQA."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE))
sys.path.insert(0, str(AE / "scripts"))

_WS = __import__("re").compile(r"\s+")

DATASETS = ("math-500", "olympiadbench", "gpqa-diamond")
SIGNALS = (
    ("geo_conf", False),
    ("direct_yes", False),
    ("verbal_h3", False),
    ("h4_margin", False),
    ("h4_coverage", False),
    ("h4_neg_entropy", False),
    ("h4_top2", False),
)


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def load_refs(dataset: str) -> tuple[dict[int, str], dict[int, str]]:
    root = AE / f"results/dense_G_r1_7b/{dataset}"
    per_sample = root / "per_sample.json"
    answers = root / "dense_puma" / "answers.json"
    a_final = {int(row["question_idx"]): row.get("A_final") for row in json.loads(per_sample.read_text())}
    gold: dict[int, str] = {}
    for index, row in enumerate(json.loads(answers.read_text())):
        gold[int(row.get("question_idx") or index + 1)] = row.get("ground_truth_answer")
    return a_final, gold


def load_v0(dataset: str) -> dict[tuple[int, int, str], float]:
    out: dict[tuple[int, int, str], float] = {}
    folder = AE / "results/confcal_judge/qwen4b_variants" / f"{dataset}_s42"
    if not folder.exists():
        return out
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                out[(int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))] = finite(
                    (row.get("B_yesno") or {}).get("yes")
                )
    return out


def load_v1(dataset: str) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    folder = AE / "results/confcal_judge/v1" / f"{dataset}_s42"
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                out[(int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))] = row
    return out


def _norm(text: Any) -> str:
    if not text:
        return ""
    return _WS.sub("", str(text).strip().lower().replace("dfrac", "frac").replace("$", ""))


def _fast_eq(left: Any, right: Any) -> bool:
    a, b = _norm(left), _norm(right)
    return bool(a) and a == b


def assemble(dataset: str) -> list[dict[str, Any]]:
    print(f"{dataset}: loading scores", flush=True)
    scores = load_v1(dataset)
    print(f"{dataset}: v1={len(scores)}", flush=True)
    v0 = load_v0(dataset)
    a_final, gold = load_refs(dataset)

    def is_g(answer: str, qi: int) -> bool:
        return _fast_eq(answer, a_final.get(qi)) or _fast_eq(answer, gold.get(qi))

    first_g: dict[int, int] = {}
    for qi, step, answer in sorted(scores):
        if qi not in first_g and is_g(answer, qi):
            first_g[qi] = step

    rows: list[dict[str, Any]] = []
    for (qi, step, answer), score in scores.items():
        h4 = score.get("direct_h4") or {}
        verbal = (score.get("verbal_h3") or {}).get("value")
        g = int(is_g(answer, qi))
        rows.append(
            {
                "dataset": dataset,
                "question_idx": qi,
                "decision_step": step,
                "answer": answer,
                "is_g": g,
                "is_first_g": int(g == 1 and first_g.get(qi) == step),
                "has_g": int(qi in first_g),
                "geo_conf": finite(score.get("geo_conf")),
                "direct_yes": finite(h4.get("yes")),
                "verbal_h3": finite(verbal) / 100.0 if verbal is not None else float("nan"),
                "h4_margin": finite(h4.get("margin")),
                "h4_coverage": finite(h4.get("yes_no_coverage")),
                "h4_neg_entropy": -finite(h4.get("vocab_entropy")),
                "h4_top2": finite(h4.get("top2_logit_margin")),
                "v0_yes": v0.get((qi, step, answer), float("nan")),
            }
        )
    return rows


def auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(y) < 2 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    if len(y) < 2 or y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    usable = tpr[fpr <= target]
    return float(usable[-1]) if len(usable) else 0.0


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "p10": float(np.quantile(arr, 0.10)) if len(arr) else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if len(arr) else float("nan"),
        "p90": float(np.quantile(arr, 0.90)) if len(arr) else float("nan"),
        "mid": float(np.mean((arr > 0.05) & (arr < 0.95))) if len(arr) else float("nan"),
        "low": float(np.mean(arr <= 0.05)) if len(arr) else float("nan"),
        "high": float(np.mean(arr >= 0.95)) if len(arr) else float("nan"),
    }


def eval_signal(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    sub = [row for row in rows if math.isfinite(row[name]) and math.isfinite(row["geo_conf"])]
    if not sub:
        return {"signal": name, "n": 0}
    y = np.asarray([row["is_g"] for row in sub], dtype=int)
    s = np.asarray([row[name] for row in sub], dtype=float)
    g = np.asarray([row["geo_conf"] for row in sub], dtype=float)
    first = [row for row in sub if row["is_first_g"] or row["is_g"] == 0]
    y_first = np.asarray([row["is_first_g"] for row in first], dtype=int)
    s_first = np.asarray([row[name] for row in first], dtype=float)
    return {
        "signal": name,
        "n": len(sub),
        "n_g": int(y.sum()),
        "n_first_g": int(y_first.sum()),
        "n_q": len({row["question_idx"] for row in sub}),
        "auroc": auc(y, s),
        "auroc_geo": auc(y, g),
        "auroc_first_g": auc(y_first, s_first),
        "tpr_fpr1": tpr_at_fpr(y, s, 0.01),
        "tpr_fpr5": tpr_at_fpr(y, s, 0.05),
        "geo_tpr_fpr1": tpr_at_fpr(y, g, 0.01),
        "geo_tpr_fpr5": tpr_at_fpr(y, g, 0.05),
        "all": quantiles(s.tolist()),
        "g": quantiles(s[y == 1].tolist()),
        "non_g": quantiles(s[y == 0].tolist()),
        "first_g": quantiles([row[name] for row in sub if row["is_first_g"] and math.isfinite(row[name])]),
    }


def main() -> None:
    report: dict[str, Any] = {}
    for dataset in DATASETS:
        rows = assemble(dataset)
        print(f"{dataset}: rows={len(rows)} G={sum(r['is_g'] for r in rows)} firstG={sum(r['is_first_g'] for r in rows)}", flush=True)
        signals = [name for name, _ in SIGNALS]
        if any(math.isfinite(row["v0_yes"]) for row in rows):
            for row in rows:
                row["v0_yes"] = row["v0_yes"]
            signals.append("v0_yes")
        report[dataset] = {name: eval_signal(rows, name) for name in signals}
    out = AE / "results/confcal_judge/v1/analysis.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)

    print("\n# AUROC / TPR@FPR\n")
    print("| dataset | signal | n / G | AUROC | geo | first-G AUROC | TPR@1% | TPR@5% | geo@1% | geo@5% |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, block in report.items():
        for name, row in block.items():
            if row.get("n", 0) == 0:
                continue
            print(
                f"| {dataset} | {name} | {row['n']} / {row['n_g']} | "
                f"{row['auroc']:.3f} | {row['auroc_geo']:.3f} | {row['auroc_first_g']:.3f} | "
                f"{row['tpr_fpr1']:.3f} | {row['tpr_fpr5']:.3f} | "
                f"{row['geo_tpr_fpr1']:.3f} | {row['geo_tpr_fpr5']:.3f} |"
            )
    print("\n# Score mass\n")
    print("| dataset | signal | group | n | mean | p10 | p50 | p90 | <=0.05 | mid | >=0.95 |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, block in report.items():
        for name in ("geo_conf", "direct_yes", "verbal_h3", "v0_yes", "h4_coverage"):
            row = block.get(name)
            if not row or row.get("n", 0) == 0:
                continue
            for group in ("all", "g", "non_g", "first_g"):
                pack = row[group]
                print(
                    f"| {dataset} | {name} | {group} | {pack['n']} | {pack['mean']:.3f} | "
                    f"{pack['p10']:.3f} | {pack['p50']:.3f} | {pack['p90']:.3f} | "
                    f"{pack['low']:.3f} | {pack['mid']:.3f} | {pack['high']:.3f} |"
                )


if __name__ == "__main__":
    main()
