#!/usr/bin/env python3
"""Replay 4B confidence / logit-lens as a stop gate vs 7B geo-conf."""
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
from analyze_dense_massmean import pairwise, replay  # noqa: E402

SIGNALS = (
    "geo_conf",
    "conf4b",
    "last_mean_logp",
    "last_min_logp",
    "neg_ans_entropy",
    "lens_l8",
    "lens_l17",
    "lens_l26",
    "lens_l35",
    "lens_best",
    "lens_rise",
    "prefix_ans_cos",
    "hidden_nn_cos",
    "hidden_norm",
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


def curve_zh(grouped: dict[int, list[dict[str, Any]]], name: str, title: str) -> list[str]:
    values = np.asarray([finite(row.get(name)) for seq in grouped.values() for row in seq])
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return [f"## {title}", "", "没有可用分数", ""]
    win, cmp = pairwise(grouped, name)
    taus = sorted(set(np.quantile(values, [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98]).tolist()))
    if name == "geo_conf":
        taus = sorted(set([0.90, 0.95, 0.97, 0.98, 0.99, 0.995] + [round(float(t), 6) for t in taus]))
    lines = [
        f"## {title}",
        "",
        f"在「前面写错过、后面才第一次写出 7B 最终答案」的题里，"
        f"第一次写对时这个分数比之前所有错的都高：{win}/{cmp}"
        + (f" = {win / cmp:.3f}" if cmp else ""),
        "",
        "| 分数门槛 | 交上去和 7B 写完后一致 | 对标准答案 | 用掉的思考长度 | 停早了 | 正好停在第一次写对 | 写对了还继续 | 从没停 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    recs = []
    for tau in taus:
        stats = [replay(seq, name, tau) for seq in grouped.values()]
        rec = {key: float(np.mean([row[key] for row in stats])) for key in ("faith", "gt", "keep")}
        rec.update({key: int(sum(row[key] for row in stats)) for key in ("early", "on", "late", "miss")})
        rec["tau"] = tau
        recs.append(rec)
        lines.append(
            f"| {tau:.4f} | {rec['faith']:.3f} | {rec['gt']:.3f} | {rec['keep']:.3f} | "
            f"{rec['early']} | {rec['on']} | {rec['late']} | {rec['miss']} |"
        )
    near = min(recs, key=lambda row: abs(row["keep"] - 0.43))
    bounded = [row for row in recs if row["keep"] <= 0.50]
    lines.append("")
    lines.append(
        f"和 7B 把握到 0.98 时用掉的长度差不多（约 43%）时："
        f"和 7B 写完后一致 {near['faith']:.3f}，对标准答案 {near['gt']:.3f}，"
        f"用掉 {near['keep']:.3f}，门槛 {near['tau']:.4f}"
    )
    if bounded:
        best = max(bounded, key=lambda row: row["faith"])
        lines.append(
            f"用掉不超过一半时最好的一档：和 7B 写完后一致 {best['faith']:.3f}，"
            f"对标准答案 {best['gt']:.3f}，用掉 {best['keep']:.3f}"
        )
    lines.append("")
    return lines


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
        f"# 4B 把握 / 中间层读出 — {args.dataset}",
        "",
        "做法：7B 的草稿塞进 4B 思考框，4B 不续写。看它写出这个试答有多顺（最后一层），",
        "以及中间层用同一套读法是否已经能读出这个试答。",
        "",
        "怎么比：每道题从早到晚走，分数第一次高过门槛就交那一步的试答。",
        "对照是 7B 自己对试答的把握。",
        "",
        f"成功读出 {len(rows)} 步，其中试答已经等于 7B 最终答案的有 {int(y.sum())} 步，题数 {len(grouped)}。",
        "",
        "下面「分得开」只说明对的步和错的步分数不一样，不能单独当交卷门。",
        "",
    ]
    labels = {
        "geo_conf": "7B 自己的把握（对照）",
        "conf4b": "4B 对试答每个词概率的几何平均（4B 版把握）",
        "last_mean_logp": "4B 对试答每个词的平均对数概率",
        "last_min_logp": "4B 对试答里最没把握的那个词",
        "neg_ans_entropy": "4B 写试答时用词有多集中（越大越集中）",
        "lens_l8": "第 8 层读出的试答平均对数概率",
        "lens_l17": "第 17 层读出的试答平均对数概率",
        "lens_l26": "第 26 层读出的试答平均对数概率",
        "lens_l35": "第 35 层读出的试答平均对数概率",
        "lens_best": "各层读出里最高的一档",
        "lens_rise": "最后一层比中间层高出多少",
        "prefix_ans_cos": "写试答前和写完后，最后一层向量有多像",
        "hidden_nn_cos": "这一步和上一步，写试答前的向量有多像",
        "hidden_norm": "写试答前最后一层向量的长度",
    }
    for name in SIGNALS:
        scores = np.asarray([finite(row.get(name)) for row in rows])
        use = np.isfinite(scores)
        if use.sum() >= 8 and y[use].min() != y[use].max():
            lines.append(f"- {labels.get(name, name)}：分得开的程度 {roc_auc_score(y[use], scores[use]):.3f}")
    lines.append("")
    for name in SIGNALS:
        if any(np.isfinite(finite(row.get(name))) for row in rows):
            lines += curve_zh(grouped, name, labels.get(name, name))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
