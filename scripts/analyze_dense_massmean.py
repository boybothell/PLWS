#!/usr/bin/env python3
"""Question-fold mass-mean / logistic probe, standalone first-hit vs geo."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq, finite, load_refs  # noqa: E402

FOLDS = 5


def load_pack(folder: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    last, pre, mid = [], [], []
    for scores in sorted(folder.glob("scores_shard*.jsonl")):
        stem = scores.stem.replace("scores", "")
        h_last = np.load(scores.with_name(f"hidden_last{stem}.npy"))
        h_pre = np.load(scores.with_name(f"hidden_pre{stem}.npy"))
        h_mid = np.load(scores.with_name(f"hidden_mid{stem}.npy"))
        ok = 0
        for line in scores.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            idx = int(row.get("hidden_idx", ok))
            rows.append(row)
            last.append(h_last[idx])
            pre.append(h_pre[idx])
            mid.append(h_mid[idx])
            ok += 1
        if ok != len(h_last):
            raise SystemExit(f"{scores} ok={ok} hidden={len(h_last)}")
    return rows, {
        "last": np.stack(last).astype(np.float32),
        "pre": np.stack(pre).astype(np.float32),
        "mid": np.stack(mid).astype(np.float32),
    }


def l2(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norm, 1e-8, None)


def mass_mean(train_h: np.ndarray, train_y: np.ndarray, test_h: np.ndarray) -> np.ndarray:
    pos = train_h[train_y == 1]
    neg = train_h[train_y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.full(len(test_h), np.nan)
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    scale = np.linalg.norm(direction)
    if scale < 1e-8:
        return np.full(len(test_h), np.nan)
    return test_h @ (direction / scale)


def logreg_score(train_h: np.ndarray, train_y: np.ndarray, test_h: np.ndarray) -> np.ndarray:
    if train_y.min() == train_y.max() or len(train_y) < 16:
        return np.full(len(test_h), np.nan)
    model = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")
    model.fit(train_h, train_y)
    return model.predict_proba(test_h)[:, 1]


def oof_scores(rows: list[dict[str, Any]], hidden: np.ndarray, kind: str) -> np.ndarray:
    qis = np.asarray([int(row["question_idx"]) for row in rows])
    y = np.asarray([int(row["is_af"]) for row in rows])
    folds = np.asarray([int(row["question_idx"]) % FOLDS for row in rows])
    out = np.full(len(rows), np.nan)
    matrix = l2(hidden) if kind.startswith("mm_") else hidden
    for fold in range(FOLDS):
        train = folds != fold
        test = folds == fold
        if test.sum() == 0 or y[train].min() == y[train].max():
            continue
        if kind.startswith("mm_"):
            out[test] = mass_mean(matrix[train], y[train], matrix[test])
        else:
            out[test] = logreg_score(matrix[train], y[train], matrix[test])
    return out


def replay(seq: list[dict[str, Any]], name: str, tau: float) -> dict[str, float]:
    n = len(seq)
    i_af = next((i for i, row in enumerate(seq) if row["is_af"]), None)
    stop = None
    for i, row in enumerate(seq):
        score = finite(row.get(name))
        if np.isfinite(score) and score >= tau:
            stop = i
            break
    if stop is None:
        last = seq[-1]
        return {
            "keep": 1.0,
            "faith": float(last["is_af"]),
            "gt": float(last["is_gt"]),
            "early": 0.0,
            "on": 0.0,
            "late": float(i_af is not None),
            "miss": 1.0,
        }
    row = seq[stop]
    return {
        "keep": (stop + 1) / n,
        "faith": float(row["is_af"]),
        "gt": float(row["is_gt"]),
        "early": float(i_af is not None and stop < i_af),
        "on": float(i_af is not None and stop == i_af),
        "late": float(i_af is not None and stop > i_af),
        "miss": 0.0,
    }


def pairwise(grouped: dict[int, list[dict[str, Any]]], name: str) -> tuple[int, int]:
    win = cmp = 0
    for seq in grouped.values():
        flags = [int(row["is_af"]) for row in seq]
        if not any(flags):
            continue
        first = flags.index(1)
        if first == 0:
            continue
        cmp += 1
        current = finite(seq[first].get(name))
        earlier = max(finite(row.get(name)) for row in seq[:first])
        if np.isfinite(current) and np.isfinite(earlier) and current > earlier:
            win += 1
    return win, cmp


def curve_lines(grouped: dict[int, list[dict[str, Any]]], name: str) -> list[str]:
    values = np.asarray([finite(row.get(name)) for seq in grouped.values() for row in seq])
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return [f"## {name}", "", "no scores", ""]
    win, cmp = pairwise(grouped, name)
    taus = sorted(set(np.quantile(values, [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98]).tolist()))
    if name == "geo_conf":
        taus = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995] + taus
        taus = sorted(set(round(float(t), 6) for t in taus))
    lines = [
        f"## {name}",
        "",
        f"题内成对 first Af > max earlier：{win}/{cmp} = {win / cmp if cmp else float('nan'):.3f}",
        "",
        "| tau | Faith | GT | keep | early | onAf | late | miss |",
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
    lines.append(f"对齐 keep≈0.43：Faith={near['faith']:.3f} GT={near['gt']:.3f} keep={near['keep']:.3f} τ={near['tau']:.4f}")
    if bounded:
        best = max(bounded, key=lambda row: row["faith"])
        lines.append(f"keep≤0.50 最好 Faith：Faith={best['faith']:.3f} GT={best['gt']:.3f} keep={best['keep']:.3f} τ={best['tau']:.4f}")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math-500")
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a_final, gold = load_refs(args.dataset)
    rows, pack = load_pack(args.hidden)
    for row in rows:
        qi = int(row["question_idx"])
        row["is_af"] = int(_fast_eq(row.get("answer"), a_final.get(qi)))
        row["is_gt"] = int(_fast_eq(row.get("answer"), gold.get(qi)))
    y = np.asarray([int(row["is_af"]) for row in rows])
    probes = {
        "mm_last": oof_scores(rows, pack["last"], "mm_last"),
        "mm_pre": oof_scores(rows, pack["pre"], "mm_pre"),
        "mm_mid": oof_scores(rows, pack["mid"], "mm_mid"),
        "logreg_last": oof_scores(rows, pack["last"], "logreg_last"),
        "logreg_pre": oof_scores(rows, pack["pre"], "logreg_pre"),
    }
    for index, row in enumerate(rows):
        for name, scores in probes.items():
            row[name] = float(scores[index])
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["question_idx"])].append(row)
    for seq in grouped.values():
        seq.sort(key=lambda row: int(row["decision_step"]))
    lines = [
        f"# Dense mass-mean probe — {args.dataset}",
        "",
        f"题折 {FOLDS} 折，标签 = 试答≈A_final。分数只在未见到该题的折上估方向。",
        f"n={len(rows)} Af={int(y.sum())} Q={len(grouped)}",
        "",
    ]
    geo = np.asarray([finite(row.get("geo_conf")) for row in rows])
    mask = np.isfinite(geo)
    lines.append(f"全表 AUROC geo={roc_auc_score(y[mask], geo[mask]):.3f}")
    for name, scores in probes.items():
        mask = np.isfinite(scores)
        if mask.sum() >= 8 and y[mask].min() != y[mask].max():
            lines.append(f"全表 AUROC {name}={roc_auc_score(y[mask], scores[mask]):.3f}")
    lines.append("")
    lines.append("主表是独立 first-hit 回放，不是 AUROC。")
    lines.append("")
    for name in ("geo_conf", *probes):
        lines += curve_lines(grouped, name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
