#!/usr/bin/env python3
"""Replay 4B DoLA / Lookback / EigenScore as a stop gate."""
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
from analyze_dense_4b_lens import curve_zh  # noqa: E402

SIGNALS = (
    "geo_conf",
    "dola_score",
    "dola_jsd_max",
    "dola_mean_rise",
    "neg_emerge",
    "layer_agree",
    "lookback_ratio",
    "lb_ans",
    "lb_rev",
    "neg_eigen_k2",
    "neg_eigen_k4",
    "mid_neg_eigen_k4",
)
LABELS = {
    "geo_conf": "7B 自己的把握（对照）",
    "dola_score": "DoLA：最后一层比和它差得最多的那一层，更想写试答多少",
    "dola_jsd_max": "DoLA：中间层和最后一层对下一个词的分布差多少",
    "dola_mean_rise": "浅层到深层，试答越来越好写的幅度",
    "neg_emerge": "第几层开始把试答的第一个词排第一（越早越大）",
    "layer_agree": "有几层已经把试答第一个词排第一",
    "lookback_ratio": "Lookback：写完试答时，注意力还在看题面的比例",
    "lb_ans": "注意力落在试答上的比例",
    "lb_rev": "注意力落在「等一下 / 或者」这类词上的比例",
    "neg_eigen_k2": "EigenScore：最近两步内部状态有多散（取负，越稳越大）",
    "neg_eigen_k4": "EigenScore：最近四步内部状态有多散（取负）",
    "mid_neg_eigen_k4": "中间层上的同样散度（取负）",
}


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
        rows.append(row)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["question_idx"])].append(row)
    for seq in grouped.values():
        seq.sort(key=lambda row: int(row["decision_step"]))
    y = np.asarray([int(row["is_af"]) for row in rows])
    lines = [
        f"# 4B DoLA / 回看注意力 / 状态散度 — {args.dataset}",
        "",
        "7B 上用过的 DoLA、Lookback、EigenScore，换到 4B 再算。",
        "每道题从早到晚走，分数第一次高过门槛就交那一步试答。对照是 7B 自己的把握。",
        f"成功读出 {len(rows)} 步，试答已等于 7B 最终答案的有 {int(y.sum())} 步，题数 {len(grouped)}。",
        "",
    ]
    for name in SIGNALS:
        scores = np.asarray([finite(row.get(name)) for row in rows])
        use = np.isfinite(scores)
        if use.sum() >= 8 and y[use].min() != y[use].max():
            lines.append(f"- {LABELS[name]}：分得开的程度 {roc_auc_score(y[use], scores[use]):.3f}")
    lines.append("")
    for name in SIGNALS:
        if any(np.isfinite(finite(row.get(name))) for row in rows):
            lines += curve_zh(grouped, name, LABELS[name])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
