#!/usr/bin/env python3
"""AUROC of judge-mode calibration vs geo-conf on every held trial step.

G (positive): trial ≈ Full-CoT A_final OR trial ≈ GT, at that step.
Negative: neither. Baseline = that step's geo-conf.
"""
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
sys.path.insert(0, str(AE))

from attn_early_exit.answers import answers_equal  # noqa: E402

ANSWERS = AE / "results/dense_G_r1_7b/math-500/dense_puma/answers.json"
PER_SAMPLE = AE / "results/dense_G_r1_7b/math-500/per_sample.json"
ROOT = AE / "results/confcal_judge"
TABLE = AE / "tables/probe_confcal_judge.md"
BOOTSTRAP = 2000
RNG_SEED = 20260814
HIGH_CONF = 0.995
SIGNALS = (
    ("A_yesno", "A_yesno"),
    ("B_yesno", "B_yesno"),
    ("A_verbal", "A_verbal"),
    ("B_verbal", "B_verbal"),
)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def load_scores(judge: str) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    score_dir = ROOT / judge / "math-500_s42"
    for path in sorted(score_dir.glob("scores_shard*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                key = (int(row["question_idx"]), int(row["decision_step"]), str(row["answer"]))
                out[key] = row
    return out


def yesno_value(row: dict[str, Any], name: str) -> float:
    pack = row.get(name)
    if isinstance(pack, dict):
        return float(pack.get("p_yes_norm", float("nan")))
    return float("nan")


def verbal_value(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def signal_value(row: dict[str, Any], name: str) -> float:
    if name.endswith("yesno"):
        return yesno_value(row, name)
    return verbal_value(row, name)


def load_refs() -> tuple[dict[int, str], dict[int, str]]:
    a_final = {
        int(r["question_idx"]): r.get("A_final")
        for r in json.loads(PER_SAMPLE.read_text())
    }
    gt = {}
    for i, row in enumerate(json.loads(ANSWERS.read_text())):
        qi = int(row.get("question_idx") or i + 1)
        gt[qi] = row.get("ground_truth_answer")
    return a_final, gt


def is_g(answer: str, a_final: str | None, gold: str | None, cache: dict) -> bool:
    def hit(target: str | None, tag: str) -> bool:
        if not target:
            return False
        key = (str(answer), tag, str(target))
        if key not in cache:
            cache[key] = bool(answers_equal(answer, target))
        return cache[key]

    return hit(a_final, "af") or hit(gold, "gt")


def assemble(judge: str) -> list[dict[str, Any]]:
    scores = load_scores(judge)
    a_final, gt = load_refs()
    cache: dict = {}
    rows: list[dict[str, Any]] = []
    for key, score in scores.items():
        g = int(is_g(key[2], a_final.get(key[0]), gt.get(key[0]), cache))
        conf = score.get("geo_conf", score.get("mean_conf_K"))
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = float("nan")
        row = {
            "question_idx": key[0],
            "decision_step": key[1],
            "answer": key[2],
            "is_g": g,
            "mean_conf_K": conf,
            "judge": judge,
        }
        for name, field in SIGNALS:
            row[name] = signal_value(score, field)
        rows.append(row)
    n_g = sum(r["is_g"] for r in rows)
    print(f"{judge}: assembled {len(rows)} rows, G={n_g}", flush=True)
    return rows


def by_question(rows: list[dict[str, Any]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row["question_idx"])].append(i)
    return groups


def bootstrap_auc(rows: list[dict[str, Any]], y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    groups = by_question(rows)
    qids = sorted(groups)
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(BOOTSTRAP):
        sample = rng.choice(qids, size=len(qids), replace=True)
        idx = np.concatenate([groups[int(q)] for q in sample])
        vals.append(auc(y[idx], s[idx]))
    finite = np.asarray(vals, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "ci_lo": float(np.quantile(finite, 0.025)),
        "ci_hi": float(np.quantile(finite, 0.975)),
    }


def bootstrap_paired_diff(
    rows: list[dict[str, Any]], y: np.ndarray, sig: np.ndarray, conf: np.ndarray
) -> dict[str, float]:
    groups = by_question(rows)
    qids = sorted(groups)
    rng = np.random.default_rng(RNG_SEED)
    diffs = []
    for _ in range(BOOTSTRAP):
        sample = rng.choice(qids, size=len(qids), replace=True)
        idx = np.concatenate([groups[int(q)] for q in sample])
        a_sig = auc(y[idx], sig[idx])
        a_conf = auc(y[idx], conf[idx])
        if np.isfinite(a_sig) and np.isfinite(a_conf):
            diffs.append(a_sig - a_conf)
    diffs = np.asarray(diffs, dtype=np.float64)
    if not len(diffs):
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_gt_0": float("nan")}
    return {
        "mean": float(np.mean(diffs)),
        "ci_lo": float(np.quantile(diffs, 0.025)),
        "ci_hi": float(np.quantile(diffs, 0.975)),
        "p_gt_0": float(np.mean(diffs > 0.0)),
    }


def fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}" if math.isfinite(v) else "NA"
    return str(v)


def eval_signal_once(
    rows: list[dict[str, Any]],
    signal: str,
    pop_name: str,
    label: str,
) -> dict[str, Any]:
    sub = [
        row
        for row in rows
        if math.isfinite(row[signal])
        and row.get(label) is not None
        and math.isfinite(row["mean_conf_K"])
    ]
    if not sub:
        return {"population": pop_name, "signal": signal, "label": label, "n": 0}
    y = np.asarray([int(row[label]) for row in sub])
    sig = np.asarray([row[signal] for row in sub], dtype=float)
    conf = np.asarray([row["mean_conf_K"] for row in sub], dtype=float)
    a_sig = auc(y, sig)
    a_conf = auc(y, conf)
    ci_sig = bootstrap_auc(sub, y, sig)
    diff = bootstrap_paired_diff(sub, y, sig, conf)
    return {
        "population": pop_name,
        "signal": signal,
        "label": label,
        "n": len(sub),
        "n_g": int(y.sum()),
        "n_questions": len({row["question_idx"] for row in sub}),
        "auc_signal": a_sig,
        "auc_signal_ci": ci_sig,
        "auc_conf": a_conf,
        "delta": diff["mean"],
        "delta_ci_lo": diff["ci_lo"],
        "delta_ci_hi": diff["ci_hi"],
        "p_gt_0": diff["p_gt_0"],
    }


def verdict(row: dict[str, Any]) -> str:
    if row["n"] == 0:
        return "无数据"
    replace = row["auc_signal"] > row["auc_conf"] and row["delta_ci_lo"] > 0.0
    return "可替代候选" if replace else "不能替代"


def table_block(title: str, results: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| judge | population | signal | n / G / 题 | AUROC(signal) | 95% CI | AUROC(conf) | Δ(signal−conf) | 95% CI | P(Δ>0) | 判定 |",
        "|---|---|---|---|---:|---|---:|---:|---|---:|---|",
    ]
    for row in results:
        if row["n"] == 0:
            lines.append(
                f"| `{row['judge']}` | `{row['population']}` | `{row['signal']}` | 0 | - | - | - | - | - | - | 无数据 |"
            )
            continue
        lines.append(
            f"| `{row['judge']}` | `{row['population']}` | `{row['signal']}` | "
            f"{row['n']} / {row['n_g']} / {row['n_questions']} | "
            f"{fmt(row['auc_signal'])} | "
            f"[{fmt(row['auc_signal_ci']['ci_lo'])}, {fmt(row['auc_signal_ci']['ci_hi'])}] | "
            f"{fmt(row['auc_conf'])} | {fmt(row['delta'])} | "
            f"[{fmt(row['delta_ci_lo'])}, {fmt(row['delta_ci_hi'])}] | "
            f"{fmt(row['p_gt_0'])} | {verdict(row)} |"
        )
    lines.append("")
    return lines


def main() -> None:
    all_results: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for judge in ("self", "qwen4b"):
        rows = assemble(judge)
        if not rows:
            continue
        pops = [
            ("all_held_steps", rows),
            (f"mean_conf_K>={HIGH_CONF}", [row for row in rows if row["mean_conf_K"] >= HIGH_CONF]),
        ]
        for signal, _ in SIGNALS:
            for pop_name, pop_rows in pops:
                rec = eval_signal_once(pop_rows, signal, pop_name, "is_g")
                rec["judge"] = judge
                gate_rows.append(rec)
                all_results.append(rec)

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "auroc_summary.json").write_text(json.dumps(all_results, indent=2) + "\n")

    lines = [
        "# 评判态置信度校准：held AUROC",
        "",
        "协议：mask 原解题前缀，新开 user 轮，把题和当前推理当作材料；`enable_thinking=False`。",
        "A = 无试答（推理是否已足够）；B = 有试答（该答案是否正确）。",
        "主读数 = Yes/No 词表质量；口头 0–100 只作对照。",
        "评判器：`self` = R1-7B；`qwen4b` = Qwen3-4B。基线 = geo-conf `mean_conf_K`。",
        "",
        "试答即终答。正标签 G = 试答 ≈ Full-CoT 终答 **或** 试答 ≈ GT；其余为负。",
        "AUROC 越高越能把可交的试答排在前面。基线 = geo-conf `mean_conf_K`。",
        "总体：held 全部 local-K=4，以及 `mean_conf_K≥0.995`。",
        f"替代判定：AUROC(signal)>AUROC(conf) 且题级 bootstrap Δ 的 95% CI 下界 > 0（{BOOTSTRAP} 次）。",
        "",
    ]
    lines += table_block("G = 试答≈A_final ∨ 试答≈GT", gate_rows)
    lines += [
        "## 预注册判决",
        "",
        "- 这个标签上 B 或 A 稳赢 geo-conf → 再谈当闸门。",
        "- 自身和 4B 都输 → 关线，不改提示词。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
