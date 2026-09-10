#!/usr/bin/env python3
"""对窗 / 错窗：遮一块之后试答 logP、把握掉多少。"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

AE = Path(__file__).resolve().parents[1]
ROOT = AE / "results/samewin_mask"
TABLE = AE / "tables/samewin_mask.md"
MASKS = (
    ("hide_problem", "遮题"),
    ("hide_cot", "遮草稿"),
    ("hide_last", "遮最后一步"),
    ("hide_early", "遮前半草稿"),
    ("hide_rand", "随机同长遮草稿"),
)
KIND_ZH = {"high": "高把握", "mix": "混合", "low": "低把握"}


def finite(x: Any) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def load_rows(dataset: str, seed: int) -> list[dict[str, Any]]:
    folder = ROOT / f"{dataset}_s{seed}"
    rows: list[dict[str, Any]] = []
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob("scores_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def arr(xs: list[dict[str, Any]], key: str, ok: bool | None = None) -> np.ndarray:
    out = []
    for x in xs:
        if ok is True and not x["gold_ok"]:
            continue
        if ok is False and x["gold_ok"]:
            continue
        v = finite(x.get(key))
        if v == v:
            out.append(v)
    return np.asarray(out, dtype=float)


def auroc(xs: list[dict[str, Any]], key: str) -> float:
    y, s = [], []
    for x in xs:
        v = finite(x.get(key))
        if v == v:
            y.append(int(bool(x["gold_ok"])))
            s.append(v)
    if len(y) < 8 or len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def block(title: str, xs: list[dict[str, Any]]) -> list[str]:
    ok_n = sum(1 for x in xs if x["gold_ok"])
    bad_n = len(xs) - ok_n
    lines = [f"### {title}  对/错 {ok_n}/{bad_n}", "", "| 遮法 | 对窗掉(均值/中位) | 错窗掉(均值/中位) | ΔAUROC |", "|---|---|---|---:|"]
    for key, name in MASKS:
        ok = arr(xs, f"dlogp_{key}", True)
        bad = arr(xs, f"dlogp_{key}", False)
        if len(ok) == 0 or len(bad) == 0:
            lines.append(f"| {name} | — | — | — |")
            continue
        lines.append(
            f"| {name} | {ok.mean():.2f} / {np.median(ok):.2f} | {bad.mean():.2f} / {np.median(bad):.2f} | {auroc(xs, f'dlogp_{key}'):.2f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "math-500"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rows = load_rows(dataset, seed)
    lines = [
        "# 四步同答窗：遮一块，对窗/错窗试答掉多少",
        "",
        "7B。每一扇连续 4 步同一 boxed 都进，含高把握 / 混合 / 低把握。",
        "对窗 = 试答对金标。数字是未遮 − 遮后的试答 token 均 logP，越大越敏感。",
        f"已打 {len(rows)} 扇。",
        "",
    ]
    lines += block("全体", rows)
    for kind, zh in KIND_ZH.items():
        part = [x for x in rows if x.get("kind") == kind]
        if part:
            lines += block(zh, part)
    lines += [
        "读法：对窗掉、错窗不掉，才是接地。两边掉一样，和旧 PMI 同类。",
        "AUROC 用掉多少排序，越大越像对窗。0.50 是猜。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
