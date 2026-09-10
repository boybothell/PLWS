#!/usr/bin/env python3
"""Replay 4B PMI / margin / evidence-gain as a stop gate."""
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
    "current_logp",
    "question_only_logp",
    "reasoning_pmi",
    "margin",
    "evidence_gain",
    "hist_forget",
    "neg_increment_nll",
    "neg_wrapup_nll",
)
LABELS = {
    "geo_conf": "7B 自己的把握（对照）",
    "current_logp": "4B 看完草稿后，写出这个试答有多顺",
    "question_only_logp": "4B 只看题目、不看草稿时，写出这个试答有多顺",
    "reasoning_pmi": "PMI：看完草稿后比只看题目，试答顺了多少",
    "margin": "当前试答比更早出现过的别的试答，4B 更站当前多少",
    "evidence_gain": "这一步草稿比上一步，试答顺了多少",
    "hist_forget": "整段草稿比只看新写的几句，试答顺了多少",
    "neg_increment_nll": "新写的那几句，4B 读起来有多顺",
    "neg_wrapup_nll": "强迫收口后再写试答，4B 有多顺",
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
        f"# 4B 草稿是否支持试答 — {args.dataset}",
        "",
        "7B 上用过的 PMI / 和历史试答比 / 比上一步涨了多少，换到 4B 再算。",
        "每道题从早到晚走，分数第一次高过门槛就交那一步试答。对照是 7B 自己的把握。",
        f"成功读出 {len(rows)} 步，试答已等于 7B 最终答案的有 {int(y.sum())} 步，题数 {len(grouped)}。",
        "",
        "「分得开」只说明对的步和错的步分数不一样，不能单独当交卷门。",
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
