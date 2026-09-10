#!/usr/bin/env python3
"""Re-evaluate first-geo process signals on every dense trial step.

Sample = dense_puma/trial_answers.json rows. Labels = G and G-window (k=2).
geo_conf is a competing score on the same rows, not a sample filter.

CPU: TF-IDF / Jaccard adjacent-step overlap, length, future-answer stability.
Optional: frozen qwen3-embedding-redundancy-detector-0.6B step cosine.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(AE.parents[0] / "PUMA"))

from analyze_confcal_v1 import _fast_eq, auc, finite, load_refs, tpr_at_fpr  # noqa: E402

DATASETS = ("math-500", "olympiadbench", "gpqa-diamond")
EMBED = Path("/mnt/d/lsj/models/qwen3-embedding-redundancy-detector-0.6B")
TABLE = AE / "tables/probe_dense_process.md"
OUT_DIR = AE / "results/confcal_judge/v2/dense_process"

CPU_SIGNALS = (
    "geo_conf",
    "tfidf_nn_adj",
    "tfidf_puma_adj",
    "jaccard_nn_adj",
    "jaccard_puma_adj",
    "tfidf_nn_tail3",
    "tfidf_puma_tail3",
    "step",
    "neg_step",
    "n_reason_tok",
    "neg_n_reason_tok",
    "ans_len",
    "neg_ans_len",
    "pref_chars",
    "inc_chars",
    "inc_tok",
    "same_run",
    "future_same_h1",
    "future_same_frac8",
    "future_all_same",
)
EMBED_SIGNALS = (
    "nn_adj",
    "nn_max_k3",
    "nn_tail3",
    "nn_rise",
    "puma_adj",
    "puma_max_k3",
    "puma_tail3",
    "puma_rise",
)


def split_nn(text: str) -> list[str]:
    parts = [chunk.strip() for chunk in (text or "").split("\n\n") if chunk.strip()]
    if len(parts) < 2:
        parts = [chunk.strip() for chunk in (text or "").split("\n") if chunk.strip()]
    return parts


def split_puma(text: str) -> list[str]:
    from puma.separate_steps import separate_steps

    return [
        chunk.strip()
        for chunk in separate_steps(text or "", min_step_chars=200, max_step_chars=1000)
        if chunk.strip()
    ]


def jaccard(left: str, right: str) -> float:
    a = set(left.lower().split())
    b = set(right.lower().split())
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def tfidf_pair_stats(steps: list[str]) -> dict[str, float]:
    empty = {"adj": float("nan"), "tail3": float("nan")}
    if len(steps) < 2:
        return empty
    try:
        matrix = TfidfVectorizer(min_df=1).fit_transform(steps)
    except ValueError:
        return empty
    sims = cosine_similarity(matrix[:-1], matrix[1:]).diagonal()
    if sims.size == 0:
        return empty
    return {
        "adj": float(sims[-1]),
        "tail3": float(np.mean(sims[-3:])),
    }


def embed_pair_stats(vectors: np.ndarray) -> dict[str, float]:
    if vectors.shape[0] < 2:
        return {"adj": float("nan"), "max_k3": float("nan"), "tail3": float("nan"), "rise": float("nan")}
    sims = np.asarray([float(vectors[i] @ vectors[i - 1]) for i in range(1, len(vectors))], dtype=np.float64)
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
    }


def load_trials(dataset: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    path = AE / f"results/dense_G_r1_7b/{dataset}/dense_puma/trial_answers.json"
    for row in json.loads(path.read_text()):
        grouped[int(row["question_idx"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["stopped_len"]))
    return grouped


def label_question(seq: list[dict[str, Any]], a_final: str, gold: str) -> list[dict[str, Any]]:
    answers = [str(row.get("final_answer") or "") for row in seq]
    flags = [_fast_eq(answer, a_final) or _fast_eq(answer, gold) for answer in answers]
    out: list[dict[str, Any]] = []
    prev_answer = ""
    prev_chars = 0
    prev_tok = 0
    run = 0
    for index, row in enumerate(seq):
        answer = answers[index]
        is_g = flags[index]
        neighbor = (index > 0 and flags[index - 1]) or (index + 1 < len(flags) and flags[index + 1])
        if prev_answer and _fast_eq(answer, prev_answer):
            run += 1
        else:
            run = 1
        future = answers[index + 1 :]
        same = [_fast_eq(item, answer) for item in future]
        prefix = str(row.get("reasoning_prefix") or "")
        n_tok = int(row.get("count_reasoning_tokens") or 0)
        n_chars = len(prefix)
        horizon = same[:8]
        out.append(
            {
                "question_idx": int(row["question_idx"]),
                "decision_step": int(row["stopped_len"]),
                "answer": answer,
                "is_g": int(is_g),
                "is_g_window": int(is_g and neighbor),
                "geo_conf": finite(row.get("confidence")),
                "step": float(row["stopped_len"]),
                "neg_step": -float(row["stopped_len"]),
                "n_reason_tok": float(n_tok),
                "neg_n_reason_tok": -float(n_tok),
                "ans_len": float(len(answer)),
                "neg_ans_len": -float(len(answer)),
                "pref_chars": float(n_chars),
                "inc_chars": float(n_chars - prev_chars) if index else float("nan"),
                "inc_tok": float(n_tok - prev_tok) if index else float("nan"),
                "same_run": float(run),
                "future_same_h1": float(same[0]) if same else float("nan"),
                "future_same_frac8": float(np.mean(horizon)) if horizon else float("nan"),
                "future_all_same": float(all(same)) if same else float("nan"),
                "reasoning_prefix": prefix,
            }
        )
        prev_answer = answer
        prev_chars = n_chars
        prev_tok = n_tok
    return out


def attach_lexical(row: dict[str, Any]) -> None:
    prefix = row.pop("reasoning_prefix")
    nn = split_nn(prefix)
    puma = split_puma(prefix)
    nn_tfidf = tfidf_pair_stats(nn)
    puma_tfidf = tfidf_pair_stats(puma)
    row["tfidf_nn_adj"] = nn_tfidf["adj"]
    row["tfidf_nn_tail3"] = nn_tfidf["tail3"]
    row["tfidf_puma_adj"] = puma_tfidf["adj"]
    row["tfidf_puma_tail3"] = puma_tfidf["tail3"]
    row["jaccard_nn_adj"] = jaccard(nn[-2], nn[-1]) if len(nn) >= 2 else float("nan")
    row["jaccard_puma_adj"] = jaccard(puma[-2], puma[-1]) if len(puma) >= 2 else float("nan")
    row["nn_steps"] = nn
    row["puma_steps"] = puma


def score_cmd(args: argparse.Namespace) -> None:
    a_final, gold = load_refs(args.dataset)
    grouped = load_trials(args.dataset)
    rows: list[dict[str, Any]] = []
    print(f"{args.dataset}: labeling {sum(len(v) for v in grouped.values())} trials", flush=True)
    for qi, seq in grouped.items():
        rows.extend(label_question(seq, str(a_final.get(qi) or ""), str(gold.get(qi) or "")))
    print(f"{args.dataset}: lexical features", flush=True)
    for index, row in enumerate(rows, 1):
        attach_lexical(row)
        if index % 2000 == 0 or index == len(rows):
            print(f"  lexical {index}/{len(rows)}", flush=True)

    cpu_out = args.out.with_name(args.out.stem + ".cpu.jsonl")
    cpu_out.parent.mkdir(parents=True, exist_ok=True)
    with cpu_out.open("w") as handle:
        for row in rows:
            slim = {
                key: value
                for key, value in row.items()
                if key not in {"nn_steps", "puma_steps", "reasoning_prefix"}
            }
            slim["dataset"] = args.dataset
            handle.write(json.dumps(slim) + "\n")
    print(f"wrote CPU {cpu_out} n={len(rows)}", flush=True)

    if not args.no_embed:
        from sentence_transformers import SentenceTransformer

        unique: list[str] = []
        index_of: dict[str, int] = {}
        for row in rows:
            for step in row["nn_steps"] + row["puma_steps"]:
                clipped = step[:1500]
                if clipped not in index_of:
                    index_of[clipped] = len(unique)
                    unique.append(clipped)
        print(f"{args.dataset}: embed {len(unique)} unique steps on {args.device}", flush=True)
        model = SentenceTransformer(str(EMBED), device=args.device)
        model.max_seq_length = 512
        vectors = np.asarray(
            model.encode(unique, normalize_embeddings=True, batch_size=args.batch_size, show_progress_bar=True),
            dtype=np.float32,
        )
        for row in rows:
            nn = [step[:1500] for step in row["nn_steps"]]
            puma = [step[:1500] for step in row["puma_steps"]]
            nn_vecs = np.stack([vectors[index_of[step]] for step in nn], axis=0) if nn else np.zeros((0, 1))
            puma_vecs = np.stack([vectors[index_of[step]] for step in puma], axis=0) if puma else np.zeros((0, 1))
            for key, value in embed_pair_stats(nn_vecs).items():
                row[f"nn_{key}"] = value
            for key, value in embed_pair_stats(puma_vecs).items():
                row[f"puma_{key}"] = value

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            slim = {key: value for key, value in row.items() if key not in {"nn_steps", "puma_steps", "reasoning_prefix"}}
            slim["dataset"] = args.dataset
            handle.write(json.dumps(slim) + "\n")
    print(f"wrote {args.out} n={len(rows)}", flush=True)


def eval_label(rows: list[dict[str, Any]], signal: str, label: str) -> dict[str, float]:
    y = np.asarray([int(row[label]) for row in rows], dtype=int)
    s = np.asarray([finite(row.get(signal)) for row in rows], dtype=float)
    g = np.asarray([finite(row.get("geo_conf")) for row in rows], dtype=float)
    mask = np.isfinite(s) & np.isfinite(g)
    if mask.sum() < 8 or y[mask].min() == y[mask].max():
        return {}
    pos = s[mask & (y == 1)]
    neg = s[mask & (y == 0)]
    return {
        "n": float(mask.sum()),
        "n_pos": float(y[mask].sum()),
        "auroc": auc(y[mask], s[mask]),
        "geo": auc(y[mask], g[mask]),
        "tpr1": tpr_at_fpr(y[mask], s[mask], 0.01),
        "geo1": tpr_at_fpr(y[mask], g[mask], 0.01),
        "tpr5": tpr_at_fpr(y[mask], s[mask], 0.05),
        "geo5": tpr_at_fpr(y[mask], g[mask], 0.05),
        "pos_p50": float(np.median(pos)) if len(pos) else float("nan"),
        "neg_p50": float(np.median(neg)) if len(neg) else float("nan"),
    }


def analyze_cmd(args: argparse.Namespace) -> None:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.scores:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    by[str(row.get("dataset") or path.stem)].append(row)
    signals = list(CPU_SIGNALS)
    if any(np.isfinite(finite(row.get("nn_adj"))) for rows in by.values() for row in rows):
        signals.extend(EMBED_SIGNALS)
    lines = [
        "# Dense process signals (not first-geo)",
        "",
        "样本 = 每条 dense 试答。标签 G 与 G 窗(k=2)。对照是同子集 `geo_conf`，不是采样器。",
        "TF-IDF / Jaccard / 长度 / 未来试答稳定从轨迹直接算；embedding 为冻结 RD-0.6B。",
        "",
    ]
    report: dict[str, Any] = {}
    for dataset, rows in by.items():
        n = len(rows)
        n_g = sum(int(row["is_g"]) for row in rows)
        n_w = sum(int(row["is_g_window"]) for row in rows)
        report[dataset] = {"n": n, "n_g": n_g, "n_window": n_w, "n_iso": n_g - n_w, "signals": {}}
        lines += [
            f"## {dataset}（n={n}，G={n_g}，G窗={n_w}，孤立G={n_g - n_w}）",
            "",
            "| 标签 | signal | n / 正 | AUROC | 同子集 geo | TPR@1% | geo@1% | TPR@5% | geo@5% | 正 p50 | 负 p50 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, tag in (("is_g", "G"), ("is_g_window", "G窗")):
            for name in signals:
                metrics = eval_label(rows, name, label)
                if not metrics:
                    continue
                report[dataset]["signals"][f"{tag}:{name}"] = metrics
                lines.append(
                    f"| {tag} | {name} | {int(metrics['n'])} / {int(metrics['n_pos'])} | "
                    f"{metrics['auroc']:.3f} | {metrics['geo']:.3f} | "
                    f"{metrics['tpr1']:.3f} | {metrics['geo1']:.3f} | "
                    f"{metrics['tpr5']:.3f} | {metrics['geo5']:.3f} | "
                    f"{metrics['pos_p50']:.4f} | {metrics['neg_p50']:.4f} |"
                )
        lines.append("")
    args.out.write_text("\n".join(lines) + "\n")
    (OUT_DIR / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    score = sub.add_parser("score")
    score.add_argument("--dataset", required=True, choices=DATASETS)
    score.add_argument("--out", type=Path)
    score.add_argument("--device", default="cuda:2")
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument("--no-embed", action="store_true")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--scores", type=Path, nargs="+", required=True)
    analyze.add_argument("--out", type=Path, default=TABLE)
    args = parser.parse_args()
    if args.cmd == "score":
        if args.out is None:
            args.out = OUT_DIR / f"{args.dataset}.jsonl"
        score_cmd(args)
    else:
        analyze_cmd(args)


if __name__ == "__main__":
    main()
