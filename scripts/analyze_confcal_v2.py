#!/usr/bin/env python3
"""Unified G/non-G analysis for H-A / H-B / H-C against geo-conf and v1 Direct."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import (  # noqa: E402
    DATASETS,
    _fast_eq,
    auc,
    finite,
    load_refs,
    load_v1,
    tpr_at_fpr,
)


def load_jsonl(folder: Path, pattern: str = "scores_shard*.jsonl") -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    if not folder.exists():
        return out
    for path in sorted(folder.glob(pattern)):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                out[(int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))] = row
    return out


def evaluate(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    valid = [row for row in rows if math.isfinite(row.get(name, float("nan")))]
    if len(valid) < 2:
        return {"signal": name, "n": 0}
    y = np.asarray([row["is_g"] for row in valid], dtype=int)
    s = np.asarray([row[name] for row in valid], dtype=float)
    first = [row for row in valid if row["is_first_g"] or not row["is_g"]]
    return {
        "signal": name,
        "n": len(valid),
        "n_g": int(y.sum()),
        "auroc": auc(y, s),
        "tpr_fpr1": tpr_at_fpr(y, s, 0.01),
        "tpr_fpr5": tpr_at_fpr(y, s, 0.05),
        "first_g_auroc": auc(
            np.asarray([row["is_first_g"] for row in first], dtype=int),
            np.asarray([row[name] for row in first], dtype=float),
        ),
    }


def assemble(dataset: str) -> list[dict[str, Any]]:
    a_final, gold = load_refs(dataset)
    v1 = load_v1(dataset)
    base = load_jsonl(AE / "results/confcal_judge/v2/universal_v2b" / f"{dataset}_s42_base")
    inst = load_jsonl(AE / "results/confcal_judge/v2/universal_v2b" / f"{dataset}_s42_instruct")
    key = load_jsonl(AE / "results/confcal_judge/v2/keytoken" / f"{dataset}_s42")
    keys = set(v1) | set(base) | set(inst) | set(key)
    first_g: dict[int, int] = {}
    labeled: list[dict[str, Any]] = []
    for qi, step, answer in sorted(keys):
        is_g = int(_fast_eq(answer, a_final.get(qi)) or _fast_eq(answer, gold.get(qi)))
        if is_g and qi not in first_g:
            first_g[qi] = step
        h4 = (v1.get((qi, step, answer), {}).get("direct_h4") or {})
        b = base.get((qi, step, answer), {})
        i = inst.get((qi, step, answer), {})
        k = key.get((qi, step, answer), {})
        labeled.append(
            {
                "is_g": is_g,
                "is_first_g": int(is_g and first_g.get(qi) == step),
                "geo_conf": finite((v1.get((qi, step, answer)) or b or i or k).get("geo_conf")),
                "direct_yes": finite(h4.get("yes")),
                "hb_base_answer": finite((b.get("hb_answer") or {}).get("per_byte", b.get("text_logprob_per_byte"))),
                "hb_inst_answer": finite((i.get("hb_answer") or {}).get("per_byte", i.get("text_logprob_per_byte"))),
                "hb_delta": finite((b.get("hb_answer") or {}).get("per_byte", b.get("text_logprob_per_byte")))
                - finite((i.get("hb_answer") or {}).get("per_byte", i.get("text_logprob_per_byte"))),
                "hb_base_full": finite((b.get("hb_full") or {}).get("per_byte")),
                "hb_inst_full": finite((i.get("hb_full") or {}).get("per_byte")),
                "hc_base": finite(b.get("candidate_prob")),
                "hc_inst": finite(i.get("candidate_prob")),
                "ha_big_js0": finite(k.get("score_big_js0")),
                "ha_small_js0": finite(k.get("score_small_js0")),
                "ha_big_js025": finite(k.get("score_big_js025")),
                "ha_small_js025": finite(k.get("score_small_js025")),
                "ha_js_mean": finite((k.get("keytoken") or {}).get("js_mean")),
            }
        )
    return labeled


def main() -> None:
    report: dict[str, Any] = {}
    signals = (
        "geo_conf", "direct_yes",
        "hb_base_answer", "hb_inst_answer", "hb_delta", "hb_base_full", "hb_inst_full",
        "hc_base", "hc_inst",
        "ha_big_js0", "ha_small_js0", "ha_big_js025", "ha_small_js025", "ha_js_mean",
    )
    print("| dataset | signal | n / G | AUROC | TPR@1% | TPR@5% | first-G AUROC |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        rows = assemble(dataset)
        report[dataset] = {name: evaluate(rows, name) for name in signals}
        for name, metrics in report[dataset].items():
            if metrics.get("n", 0) == 0:
                continue
            print(
                f"| {dataset} | {name} | {metrics['n']} / {metrics['n_g']} | "
                f"{metrics['auroc']:.3f} | {metrics['tpr_fpr1']:.3f} | "
                f"{metrics['tpr_fpr5']:.3f} | {metrics['first_g_auroc']:.3f} |"
            )
    out = AE / "results/confcal_judge/v2/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
