#!/usr/bin/env python3
"""Frozen embedding-step cosine on first geo-stop.

Uses the PUMA redundancy detector (Qwen3-Embedding-0.6B, frozen).
This is the Figure-1 metric: semantic step similarity, not residual cosine.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(AE.parents[0] / "PUMA"))

EMBED = Path("/mnt/d/lsj/models/qwen3-embedding-redundancy-detector-0.6B")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def split_nn(text: str) -> list[str]:
    parts = [chunk.strip() for chunk in (text or "").split("\n\n") if chunk.strip()]
    if len(parts) < 2:
        parts = [chunk.strip() for chunk in (text or "").split("\n") if chunk.strip()]
    return parts


def split_puma(text: str) -> list[str]:
    from puma.separate_steps import separate_steps

    return [chunk.strip() for chunk in separate_steps(text or "", min_step_chars=200, max_step_chars=1000) if chunk.strip()]


def pair_stats(vectors: np.ndarray) -> dict[str, float]:
    if vectors.shape[0] < 2:
        return {
            "adj": float("nan"),
            "max_k3": float("nan"),
            "tail3": float("nan"),
            "rise": float("nan"),
            "flag035": float("nan"),
        }
    sims = np.asarray(
        [float(vectors[i] @ vectors[i - 1]) for i in range(1, len(vectors))],
        dtype=np.float64,
    )
    max_k3 = []
    for i in range(1, len(vectors)):
        start = max(0, i - 3)
        max_k3.append(float(np.max(vectors[i] @ vectors[start:i].T)))
    last = float(sims[-1])
    earlier = sims[:-1]
    return {
        "adj": last,
        "max_k3": float(max_k3[-1]),
        "tail3": float(np.mean(sims[-3:])),
        "rise": last - float(np.mean(earlier)) if len(earlier) else float("nan"),
        "flag035": float(last >= 0.35),
    }


def score_cmd(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    rows = load_jsonl(args.candidates)
    if args.limit:
        rows = rows[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(str(EMBED), device=args.device)
    print(f"embed-sim n={len(rows)} device={args.device} -> {args.out}", flush=True)
    started = time.perf_counter()
    ok = skipped = 0
    with args.out.open("w") as handle:
        for number, row in enumerate(rows, 1):
            text = str(row.get("reasoning_prefix") or "")
            nn = split_nn(text)
            puma = split_puma(text)
            if len(nn) < 2 and len(puma) < 2:
                slim = {key: row[key] for key in row if key != "reasoning_prefix"}
                handle.write(json.dumps({**slim, "status": "few_steps"}) + "\n")
                skipped += 1
                continue
            unique = []
            index = {}
            for step in nn + puma:
                if step not in index:
                    index[step] = len(unique)
                    unique.append(step)
            emb = model.encode(unique, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            emb = np.asarray(emb, dtype=np.float32)
            nn_vecs = np.stack([emb[index[step]] for step in nn], axis=0) if nn else np.zeros((0, 1))
            puma_vecs = np.stack([emb[index[step]] for step in puma], axis=0) if puma else np.zeros((0, 1))
            nn_stats = pair_stats(nn_vecs)
            puma_stats = pair_stats(puma_vecs)
            slim = {key: row[key] for key in row if key != "reasoning_prefix"}
            handle.write(
                json.dumps(
                    {
                        **slim,
                        "status": "ok",
                        "n_nn": len(nn),
                        "n_puma": len(puma),
                        **{f"nn_{key}": value for key, value in nn_stats.items()},
                        **{f"puma_{key}": value for key, value in puma_stats.items()},
                    }
                )
                + "\n"
            )
            ok += 1
            if number % 50 == 0 or number == len(rows):
                print(f"[{number}/{len(rows)}] ok={ok} skip={skipped} {time.perf_counter()-started:.0f}s", flush=True)
    print(f"done ok={ok} skip={skipped} -> {args.out}", flush=True)


def analyze_cmd(args: argparse.Namespace) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    def auc(y, s):
        y, s = np.asarray(y), np.asarray(s)
        if len(y) < 4 or y.min() == y.max():
            return float("nan")
        return float(roc_auc_score(y, s))

    def tpr_at(y, s, target=0.05):
        y, s = np.asarray(y), np.asarray(s)
        if len(y) < 4 or y.min() == y.max():
            return float("nan")
        fpr, tpr, _ = roc_curve(y, s)
        usable = tpr[fpr <= target]
        return float(usable[-1]) if len(usable) else 0.0

    rows = [row for path in args.scores for row in load_jsonl(path) if row.get("status") == "ok"]
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[str(row.get("dataset") or "unknown")].append(row)
    signals = [
        "geo_conf",
        "nn_adj",
        "nn_max_k3",
        "nn_tail3",
        "nn_rise",
        "nn_flag035",
        "puma_adj",
        "puma_max_k3",
        "puma_tail3",
        "puma_rise",
        "puma_flag035",
    ]
    lines = [
        "# Embedding-step cosine on first geo-stop",
        "",
        "Frozen `qwen3-embedding-redundancy-detector-0.6B`. Higher cosine = more G-like.",
        "",
    ]
    for dataset, items in by.items():
        y = np.asarray([int(row.get("is_g") or 0) for row in items], dtype=int)
        raw_geo = np.asarray([finite(row.get("geo_conf")) for row in items])
        raw_adj = np.asarray([finite(row.get("nn_adj")) for row in items])
        lines += [
            f"## {dataset} (n={len(items)}, G={int(y.sum())}, nonG={int((1 - y).sum())})",
            "",
            "| signal | AUROC | TPR@1% | TPR@5% | G p50 | nonG p50 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in signals:
            raw = np.asarray([finite(row.get(name)) for row in items])
            mask = np.isfinite(raw)
            if mask.sum() < 8:
                continue
            g = raw[mask & (y == 1)]
            n = raw[mask & (y == 0)]
            lines.append(
                f"| {name} | {auc(y[mask], raw[mask]):.3f} | "
                f"{tpr_at(y[mask], raw[mask], 0.01):.3f} | {tpr_at(y[mask], raw[mask], 0.05):.3f} | "
                f"{float(np.median(g)) if len(g) else float('nan'):.4f} | "
                f"{float(np.median(n)) if len(n) else float('nan'):.4f} |"
            )
        combo = raw_geo + 0.10 * np.nan_to_num(raw_adj, nan=0.0)
        mask = np.isfinite(raw_geo)
        lines.append(
            f"| geo+0.10*nn_adj | {auc(y[mask], combo[mask]):.3f} | "
            f"{tpr_at(y[mask], combo[mask], 0.01):.3f} | {tpr_at(y[mask], combo[mask], 0.05):.3f} | "
            f"{float(np.median(combo[mask & (y == 1)])):.4f} | "
            f"{float(np.median(combo[mask & (y == 0)])):.4f} |"
        )
        lines.append("")
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    score = sub.add_parser("score")
    score.add_argument("--candidates", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--limit", type=int, default=0)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--scores", type=Path, nargs="+", required=True)
    analyze.add_argument("--out", type=Path, default=AE / "tables/probe_embed_sim.md")
    args = parser.parse_args()
    if args.cmd == "score":
        score_cmd(args)
    else:
        analyze_cmd(args)


if __name__ == "__main__":
    main()
