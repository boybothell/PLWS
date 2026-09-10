#!/usr/bin/env python3
"""Standalone first-hit replay for 4B open-think stop-margin vs geo."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq, finite, load_refs  # noqa: E402
from analyze_dense_massmean import curve_lines, replay  # noqa: E402

SIGNALS = (
    "geo_conf",
    "stop_margin",
    "stop_vs_cont",
    "stop_margin_alt",
    "stop_logp",
    "stop_margin_nl",
    "stop_vs_cont_nl",
    "stop_logp_nl",
    "closed_yes",
    "closed_margin",
)


def load(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("scores_shard*.jsonl"))
    rows = []
    for file in files:
        for line in file.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def oracle_first_af(seq: list[dict[str, Any]]) -> dict[str, float]:
    i_af = next((i for i, row in enumerate(seq) if row["is_af"]), None)
    if i_af is None:
        last = seq[-1]
        return {
            "keep": 1.0,
            "faith": float(last["is_af"]),
            "gt": float(last["is_gt"]),
            "early": 0.0,
            "on": 0.0,
            "late": 0.0,
            "miss": 1.0,
        }
    fake = [{**row, "oracle": 1.0 if i >= i_af else 0.0} for i, row in enumerate(seq)]
    return replay(fake, "oracle", 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a_final, gold = load_refs(args.dataset)
    rows = []
    for row in load(args.scores):
        qi = int(row["question_idx"])
        row["is_af"] = int(_fast_eq(row.get("answer"), a_final.get(qi)))
        row["is_gt"] = int(_fast_eq(row.get("answer"), gold.get(qi)))
        if "is_g" not in row or row["is_g"] is None:
            row["is_g"] = int(row["is_af"] or row["is_gt"])
        rows.append(row)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["question_idx"])].append(row)
    for seq in grouped.values():
        seq.sort(key=lambda row: int(row["decision_step"]))
    y = np.asarray([int(row["is_af"]) for row in rows])
    lines = [
        f"# Dense 4B open-think stop-margin — {args.dataset}",
        "",
        "开思考模板：7B 推理 + 本步 boxed 塞进 4B 的 `<think>`，4B 不续写。",
        "主表是独立 first-hit 回放，不是 AUROC。标签 = 试答≈A_final；GT 只作旁注。",
        f"n={len(rows)} Af={int(y.sum())} Q={len(grouped)}",
        "",
    ]
    geo = np.asarray([finite(row.get("geo_conf")) for row in rows])
    mask = np.isfinite(geo)
    if mask.sum() >= 8 and y[mask].min() != y[mask].max():
        lines.append(f"全表 AUROC geo={roc_auc_score(y[mask], geo[mask]):.3f}")
    for name in SIGNALS:
        if name == "geo_conf":
            continue
        scores = np.asarray([finite(row.get(name)) for row in rows])
        use = np.isfinite(scores)
        if use.sum() >= 8 and y[use].min() != y[use].max():
            lines.append(f"全表 AUROC {name}={roc_auc_score(y[use], scores[use]):.3f}")
    lines.append("")
    oracles = [oracle_first_af(seq) for seq in grouped.values()]
    o_faith = float(np.mean([row["faith"] for row in oracles]))
    o_gt = float(np.mean([row["gt"] for row in oracles]))
    o_keep = float(np.mean([row["keep"] for row in oracles]))
    o_on = int(sum(row["on"] for row in oracles))
    lines.append(
        f"first-Af oracle：Faith={o_faith:.3f} GT={o_gt:.3f} keep={o_keep:.3f} onAf={o_on}"
    )
    lines.append("")
    for name in SIGNALS:
        if any(np.isfinite(finite(row.get(name))) for row in rows):
            lines += curve_lines(grouped, name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
