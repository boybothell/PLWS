#!/usr/bin/env python3
"""Layer-curve distributions: rise/fall share and magnitude, not AUROC."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

from analyze_confcal_v1 import _fast_eq, _norm, finite  # noqa: E402

SEG_LABELS = ("浅层前半", "浅层后半", "临输出", "最后一层")


def load_refs(dataset: str, model_tag: str = "r1_7b", seed: int | None = None) -> tuple[dict[int, str], dict[int, str]]:
    root = AE / f"results/dense_G_{model_tag}/{dataset}"
    if seed is not None:
        root = root / f"seed_{seed}"
    per_sample = root / "per_sample.json"
    answers = root / "dense_puma" / "answers.json"
    if not per_sample.exists():
        seeds = sorted(root.glob("seed_*/per_sample.json"))
        if not seeds:
            raise FileNotFoundError(per_sample)
        per_sample = seeds[0]
        answers = per_sample.parent / "dense_puma" / "answers.json"
    a_final = {int(row["question_idx"]): row.get("A_final") for row in json.loads(per_sample.read_text())}
    gold: dict[int, str] = {}
    for index, row in enumerate(json.loads(answers.read_text())):
        gold[int(row.get("question_idx") or index + 1)] = row.get("ground_truth_answer")
    return a_final, gold


def trial_path(model_tag: str, dataset: str, seed: int | None) -> Path | None:
    root = AE / f"results/dense_G_{model_tag}/{dataset}"
    if seed is not None:
        path = root / f"seed_{seed}" / "dense_puma" / "trial_answers.json"
        return path if path.exists() else None
    direct = root / "dense_puma" / "trial_answers.json"
    if direct.exists():
        return direct
    seeds = sorted(root.glob("seed_*/dense_puma/trial_answers.json"))
    return seeds[0] if len(seeds) == 1 else None


def load_scores(folder: Path) -> list[dict[str, Any]]:
    rows = []
    files = sorted(folder.glob("scores_shard*.jsonl")) or sorted(folder.glob("*.jsonl"))
    for path in files:
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def lens_names(rows: list[dict[str, Any]]) -> list[str]:
    names = sorted(
        {key for row in rows for key in row if key.startswith("lens_l") and key[6:].isdigit()},
        key=lambda key: int(key[6:]),
    )
    return names + ["last_mean_logp"]


def build_meta(model_tag: str, dataset: str, seed: int | None) -> dict[tuple[int, int, str], dict[str, Any]]:
    path = trial_path(model_tag, dataset, seed)
    if path is None:
        return {}
    a_final, _ = load_refs(dataset, model_tag=model_tag, seed=seed)
    grouped: dict[int, list] = defaultdict(list)
    for row in json.loads(path.read_text()):
        grouped[int(row["question_idx"])].append(row)
    meta: dict[tuple[int, int, str], dict[str, Any]] = {}
    for qi, seq in grouped.items():
        seq = sorted(seq, key=lambda row: int(row["stopped_len"]))
        answers = [_norm(row.get("final_answer")) for row in seq]
        n = len(seq)
        for i, row in enumerate(seq):
            ans = answers[i]
            consec = (i > 0 and bool(ans) and ans == answers[i - 1]) or (
                i + 1 < n and bool(ans) and ans == answers[i + 1]
            )
            meta[(qi, int(row["stopped_len"]), str(row.get("final_answer") or ""))] = {
                "consec": consec,
                "af": bool(ans) and ans == _norm(a_final.get(qi)),
            }
    return meta


def pct(xs: np.ndarray, pred) -> str:
    if len(xs) == 0:
        return "—"
    return f"{100.0 * float(np.mean(pred(xs))):.0f}%"


def mag_bins(values: np.ndarray, rising: bool) -> list[str]:
    edges = (0.0, 1.0, 2.0, 4.0, 8.0, 20.0)
    labels = ("0–1", "1–2", "2–4", "4–8", "≥8")
    out = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        if rising:
            hit = (values >= lo) & (values < hi if hi < 20 else True)
            if hi >= 20:
                hit = values >= lo
        else:
            if hi >= 20:
                hit = values <= -lo
            else:
                hit = (values <= -lo) & (values > -hi)
        out.append(f"{lab} {100.0 * float(np.mean(hit)):.0f}%")
    return out


def summarize_mask(d: np.ndarray, mask: np.ndarray) -> list[str]:
    if mask.sum() < 20:
        return [f"n={int(mask.sum())} 太少"]
    cur = d[mask]
    lines = []
    for i, name in enumerate(SEG_LABELS[: cur.shape[1]]):
        seg = cur[:, i]
        up = seg[seg > 0]
        dn = seg[seg < 0]
        lines.append(
            f"{name}: 升{np.mean(seg > 0):.0%} 降{np.mean(seg < 0):.0%}"
            + (f"；升的一半超过 {np.median(up):+.2f}，一成超过 {np.quantile(up, 0.9):+.2f}" if len(up) else "；没有升")
            + (f"；降的一半掉 {np.median(-dn):.2f}，一成掉过 {np.quantile(-dn, 0.9):.2f}" if len(dn) else "；没有降")
        )
        if name == "临输出":
            lines.append("  升幅占全部步: " + ", ".join(mag_bins(seg, True)))
            lines.append("  降幅占全部步: " + ", ".join(mag_bins(seg, False)))
    return lines


def report(scores: Path, dataset: str, model_tag: str, seed: int | None, out: Path) -> str:
    rows = load_scores(scores)
    a_final, gold = load_refs(dataset, model_tag=model_tag, seed=seed)
    names = lens_names(rows)
    meta = build_meta(model_tag, dataset, seed)
    kept = []
    for row in rows:
        xs = [finite(row.get(name)) for name in names]
        if not all(math.isfinite(x) for x in xs):
            continue
        qi = int(row["question_idx"])
        key = (qi, int(row["decision_step"]), str(row.get("answer") or ""))
        info = meta.get(key, {})
        is_af = int(info["af"]) if "af" in info else int(_fast_eq(row.get("answer"), a_final.get(qi)))
        kept.append(
            {
                "x": np.array(xs, float),
                "geo": finite(row.get("geo_conf")),
                "af": bool(is_af),
                "gt": bool(_fast_eq(row.get("answer"), gold.get(qi))),
                "consec": bool(info.get("consec", False)),
            }
        )
    x = np.stack([row["x"] for row in kept])
    d = np.diff(x, axis=1)
    geo = np.array([row["geo"] for row in kept])
    af = np.array([row["af"] for row in kept], bool)
    consec = np.array([row["consec"] for row in kept], bool)
    ca = consec & af
    groups = [
        ("试答=终答", af),
        ("试答≠终答", ~af),
        ("连续同答=终答，把握<0.98", ca & np.isfinite(geo) & (geo < 0.98)),
        ("连续同答=终答，把握≥0.98", ca & np.isfinite(geo) & (geo >= 0.98)),
        ("连续同答≠终答，把握<0.98", consec & ~af & np.isfinite(geo) & (geo < 0.98)),
        ("连续同答≠终答，把握≥0.98", consec & ~af & np.isfinite(geo) & (geo >= 0.98)),
    ]
    layer_txt = " → ".join(names)
    lines = [
        f"# 层曲线分布 — {model_tag} / {dataset}",
        "",
        f"步数 {len(kept)}。对的步 = 试答 ≈ 该模型写完后的终答。层读出：`{layer_txt}`。",
        "看每一步自己从浅到深是升是降、幅度有多大，不看中位数当结论。",
        "",
    ]
    for title, mask in groups:
        lines.append(f"## {title}（{int(mask.sum())} 步）")
        lines.extend(f"- {item}" for item in summarize_mask(d, mask))
        lines.append("")
    if (ca & np.isfinite(geo) & (geo < 0.98)).sum() and (ca & np.isfinite(geo) & (geo >= 0.98)).sum():
        low = d[ca & (geo < 0.98) & np.isfinite(geo), -2] if d.shape[1] >= 2 else np.array([])
        # -2 is 临输出 if 4 segments; last is 最后一层, so -2 is 临输出
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-tag", default="r1_7b")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or AE / "tables" / f"probe_layer_curve_{args.model_tag}_{args.dataset}.md"
    print(report(args.scores, args.dataset, args.model_tag, args.seed, out), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
