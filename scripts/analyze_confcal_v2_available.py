#!/usr/bin/env python3
"""Analyze whatever v2 scores are already on disk, with geo-conf on the same subset."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq, auc, finite, load_refs, load_v1, tpr_at_fpr  # noqa: E402
from score_confcal_keytoken import score_from_window  # noqa: E402


def load_jsonl(folder: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
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
                key = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
                slim = {k: row[k] for k in row if k != "tokens"}
                out[key] = slim
    return out


def evaluate(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    valid = [row for row in rows if math.isfinite(row.get(name, float("nan"))) and math.isfinite(row.get("geo_conf", float("nan")))]
    if len(valid) < 2:
        return {"signal": name, "n": 0}
    y = np.asarray([row["is_g"] for row in valid], dtype=int)
    s = np.asarray([row[name] for row in valid], dtype=float)
    g = np.asarray([row["geo_conf"] for row in valid], dtype=float)
    first = [row for row in valid if row["is_first_g"] or not row["is_g"]]
    return {
        "signal": name,
        "n": len(valid),
        "n_g": int(y.sum()),
        "n_q": len({id(row) for row in valid}),
        "auroc": auc(y, s),
        "auroc_geo": auc(y, g),
        "tpr_fpr1": tpr_at_fpr(y, s, 0.01),
        "tpr_fpr5": tpr_at_fpr(y, s, 0.05),
        "geo_tpr1": tpr_at_fpr(y, g, 0.01),
        "geo_tpr5": tpr_at_fpr(y, g, 0.05),
        "first_g_auroc": auc(
            np.asarray([row["is_first_g"] for row in first], dtype=int),
            np.asarray([row[name] for row in first], dtype=float),
        ),
        "g_mean": float(s[y == 1].mean()) if (y == 1).any() else float("nan"),
        "nong_mean": float(s[y == 0].mean()) if (y == 0).any() else float("nan"),
    }


def assemble(dataset: str) -> list[dict[str, Any]]:
    a_final, gold = load_refs(dataset)
    v1 = load_v1(dataset)
    base = load_jsonl(AE / "results/confcal_judge/v2/universal_v2b" / f"{dataset}_s42_base")
    inst = load_jsonl(AE / "results/confcal_judge/v2/universal_v2b" / f"{dataset}_s42_instruct")
    key = load_jsonl(AE / "results/confcal_judge/v2/keytoken" / f"{dataset}_s42")
    first_g: dict[int, int] = {}
    for qi, step, answer in sorted(set(v1) | set(key) | set(base)):
        if qi not in first_g and (_fast_eq(answer, a_final.get(qi)) or _fast_eq(answer, gold.get(qi))):
            first_g[qi] = step
    rows: list[dict[str, Any]] = []
    for qi, step, answer in sorted(set(v1) | set(key) | set(base) | set(inst)):
        is_g = int(_fast_eq(answer, a_final.get(qi)) or _fast_eq(answer, gold.get(qi)))
        v = v1.get((qi, step, answer), {})
        b = base.get((qi, step, answer), {})
        i = inst.get((qi, step, answer), {})
        k = key.get((qi, step, answer), {})
        h4 = v.get("direct_h4") or {}
        summary = k.get("keytoken") or {}
        row = {
            "question_idx": qi,
            "is_g": is_g,
            "is_first_g": int(is_g and first_g.get(qi) == step),
            "geo_conf": finite((v or b or i or k).get("geo_conf")),
            "direct_yes": finite(h4.get("yes")),
            "hb_base_answer": finite((b.get("hb_answer") or {}).get("per_byte", b.get("text_logprob_per_byte"))),
            "hb_base_step": finite((b.get("hb_step") or {}).get("per_byte")),
            "hb_base_window": finite((b.get("hb_window") or {}).get("per_byte")),
            "hb_base_full": finite((b.get("hb_full") or {}).get("per_byte")),
            "hc_base": finite(b.get("candidate_prob")),
            "ha_big_js0": finite(k.get("score_big_js0")),
            "ha_small_js0": finite(k.get("score_small_js0")),
            "ha_big_js025": finite(k.get("score_big_js025")),
            "ha_small_js025": finite(k.get("score_small_js025")),
            "ha_js_mean": finite(summary.get("js_mean")),
            "ha_neg_js": -finite(summary.get("js_mean")),
            "ha_p_big_boxed": finite(summary.get("p_big_boxed")),
            "ha_p_small_boxed": finite(summary.get("p_small_boxed")),
            "ha_self_certainty": finite(summary.get("self_certainty")),
        }
        if summary.get("window"):
            for theta in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
                row[f"ha_big_js{theta:.2f}"] = score_from_window(summary, theta, "big")
                row[f"ha_small_js{theta:.2f}"] = score_from_window(summary, theta, "small")
        rows.append(row)
    return rows


def main() -> None:
    datasets = [name for name in ("math-500", "olympiadbench", "gpqa-diamond") if (AE / "results/confcal_judge/v1" / f"{name}_s42").exists()]
    report: dict[str, Any] = {}
    print("| dataset | signal | n / G | AUROC | geo | TPR@1% | geo@1% | TPR@5% | geo@5% | first-G | G mean | nonG mean |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        print(f"{dataset}: assembling", flush=True)
        rows = assemble(dataset)
        names = [
            "geo_conf", "direct_yes",
            "hb_base_answer", "hb_base_step", "hb_base_window", "hb_base_full", "hc_base",
            "ha_big_js0", "ha_small_js0", "ha_big_js025", "ha_small_js025",
            "ha_js_mean", "ha_neg_js", "ha_p_big_boxed", "ha_p_small_boxed", "ha_self_certainty",
        ]
        extra = sorted({key for row in rows for key in row if key.startswith("ha_big_js") or key.startswith("ha_small_js")})
        names.extend(name for name in extra if name not in names)
        report[dataset] = {name: evaluate(rows, name) for name in names}
        for name, metrics in report[dataset].items():
            if metrics.get("n", 0) == 0:
                continue
            print(
                f"| {dataset} | {name} | {metrics['n']} / {metrics['n_g']} | "
                f"{metrics['auroc']:.3f} | {metrics['auroc_geo']:.3f} | "
                f"{metrics['tpr_fpr1']:.3f} | {metrics['geo_tpr1']:.3f} | "
                f"{metrics['tpr_fpr5']:.3f} | {metrics['geo_tpr5']:.3f} | "
                f"{metrics['first_g_auroc']:.3f} | {metrics['g_mean']:.3f} | {metrics['nong_mean']:.3f} |"
            )
    out = AE / "results/confcal_judge/v2/analysis_available.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
