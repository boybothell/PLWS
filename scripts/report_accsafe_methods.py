#!/usr/bin/env python3
"""只按 Acc 不降、再省步评估现有分数。AUROC 不进表。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import train_afinal_anchor_probe as ap
import train_leftover_commit_probe as tp

TABLE = AE / "tables/accsafe_methods.md"
HARD = ("olympiadbench", "gpqa-diamond", "aime24", "aime25")


def arrays(xs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([int(x["leftover_ok"]) for x in xs], dtype=np.float64),
        np.asarray([int(x["nofs_ok"]) for x in xs], dtype=np.float64),
        np.asarray([int(x["decision_step"]) for x in xs], dtype=np.float64),
        np.asarray([int(x["keep_tok"]) for x in xs], dtype=np.float64),
    )


def mask(xs: list[dict[str, Any]], scores: np.ndarray, mix_only: bool) -> np.ndarray:
    out = np.asarray(scores, dtype=np.float64).copy()
    if mix_only:
        for i, row in enumerate(xs):
            if row.get("kind") != "mix":
                out[i] = -1e9
    return out


def eval_at(xs: list[dict[str, Any]], scores: np.ndarray, tau: float) -> dict[str, float]:
    yf, yk, tf, tk = arrays(xs)
    acc, tok, n_fire = tp.apply_gate(scores, tau, yf, yk, tf, tk)
    host_acc = float(yk.mean()) if xs else 0.0
    host_tok = float(tk.mean()) if xs else 0.0
    fire = scores >= tau
    wait = np.asarray([int(x["wait_helps"]) for x in xs], dtype=np.int32)
    return {
        "d_acc": 100.0 * (acc - host_acc) if xs else 0.0,
        "d_tok": (tok - host_tok) if xs else 0.0,
        "n_fire": float(n_fire),
        "n": float(len(xs)),
        "ok": (not xs) or acc + 1e-12 >= host_acc,
        "wait_fire": float((fire & (wait == 1)).sum()) if xs else 0.0,
    }


def pick_cell_safe(
    xs: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    zero_wait: bool = False,
) -> float:
    """每个训练数据集 Acc 都不低于密探，再尽量少步。zero_wait 再要求训练格误杀为 0。"""
    if not xs:
        return float("inf")
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(xs):
        by_ds[row["dataset"]].append(i)
    finite = scores[scores == scores]
    if len(finite) == 0:
        return float("inf")
    cands = np.unique(np.quantile(finite[finite > -1e8], np.linspace(0.05, 0.99, 30)))
    best: tuple[float, float, float] | None = None
    for tau in cands:
        recs = []
        ok_all = True
        for idxs in by_ds.values():
            rec = eval_at([xs[i] for i in idxs], scores[np.asarray(idxs)], float(tau))
            recs.append(rec)
            if not rec["ok"] or (zero_wait and rec["wait_fire"] > 0):
                ok_all = False
                break
        if not ok_all:
            continue
        n = sum(r["n"] for r in recs)
        d_acc = sum(r["d_acc"] * r["n"] for r in recs) / max(n, 1)
        d_tok = sum(r["d_tok"] * r["n"] for r in recs) / max(n, 1)
        key = (d_acc, -d_tok, -float(tau))
        if best is None or key > (best[1], best[2], 0.0):
            best = (float(tau), d_acc, -d_tok)
    return best[0] if best else float("inf")


def cell(rec: dict[str, float]) -> str:
    if rec["n"] == 0:
        return "—"
    flag = "" if rec["ok"] else " 伤"
    wait = int(rec["wait_fire"])
    return (
        f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f}步"
        f"（{int(rec['n_fire'])}/{int(rec['n'])}，误杀 {wait}）{flag}"
    )


def score_conf(xs: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([tp.rg.finite(x.get("confidence")) for x in xs])


def score_lin(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> np.ndarray:
    return tp.logreg(
        tp.feat_last(train),
        np.asarray([int(x["committed"]) for x in train], dtype=np.int32),
        tp.feat_last(test),
    )


def score_mlp(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> np.ndarray:
    return ap.mlp_scores(train, test, use_anchor=False)


def pool(recs: list[dict[str, float]]) -> str:
    if not recs:
        return "—"
    n = sum(r["n"] for r in recs)
    fire = sum(r["n_fire"] for r in recs)
    wait = sum(r["wait_fire"] for r in recs)
    hurt = sum(1 for r in recs if not r["ok"])
    d_tok = sum(r["d_tok"] * r["n"] for r in recs) / max(n, 1)
    d_acc = sum(r["d_acc"] * r["n"] for r in recs) / max(n, 1)
    return (
        f"{d_acc:+.1f}pp / {d_tok:+.0f}步，伤 {hurt}/{len(recs)} 格，"
        f"开火 {int(fire)}/{int(n)}，误杀 {int(wait)}"
    )


def run_method(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    scorer: Callable[[list[dict[str, Any]], list[dict[str, Any]]], np.ndarray] | None,
    *,
    mix_only: bool,
    oracle: str | None = None,
    zero_wait: bool = False,
) -> dict[str, float]:
    if oracle == "ok":
        te = np.asarray([float(x["leftover_ok"]) for x in test])
        return eval_at(test, mask(test, te, mix_only), 0.5)
    if oracle == "af":
        te = np.asarray([float(x["committed"]) for x in test])
        return eval_at(test, mask(test, te, mix_only), 0.5)
    if scorer is None:
        tr = mask(train, score_conf(train), mix_only)
        te = mask(test, score_conf(test), mix_only)
    else:
        tr = mask(train, scorer(train, train), mix_only)
        te = mask(test, scorer(train, test), mix_only)
    tau = pick_cell_safe(train, tr, zero_wait=zero_wait)
    return eval_at(test, te, tau)


def main() -> None:
    rows = [r for r in tp.load_scored() if r["_has_af"]]
    by_model = tp.group_model(rows)
    methods = (
        ("conf", "窗末把握", None, False, None),
        ("conf_mix", "把握·只批混合窗", None, True, None),
        ("lin", "线性探针", score_lin, False, None),
        ("lin_mix", "线性·只批混合窗", score_lin, True, None),
        ("mlp", "小 MLP", score_mlp, False, None),
        ("ok", "先知：剩窗已对", None, False, "ok"),
        ("af", "先知：等于写完终答", None, False, "af"),
    )
    lines = [
        "# Acc 不降再省步：现有分数怎么当方法",
        "",
        "这张表不报 AUROC。一条分数要先变成门槛，再变成相对密探的 Acc / 步数。",
        "",
        "门槛怎么选：在训练难集上，**每一个数据集**的 Acc 都不低于「剩窗全留密探」，",
        "再尽量少步。选不出这样的门槛就永远不开火。",
        "测试是留一难集。MATH 用全部难集上选出的门槛，不进训练。",
        "",
        "数字是剩窗题上的残差：开火 = 交当前试答，不开 = 继续密探。",
        "误杀 = 交了会错、再等密探是对的题。伤 = 该格 Acc 低于密探。",
        "好方法 = 测试难集每一格都不伤，且步数为负。先知是上限，不是方法。",
        "",
        "## 1. 留一难集",
        "",
        "| 集 | 窗末把握 | 把握·只批混合窗 | 线性探针 | 线性·只批混合窗 | 小 MLP | 先知：剩窗已对 | 先知：等于写完终答 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    summary: dict[str, list[dict[str, float]]] = {k: [] for k, *_ in methods}
    zw_summary: dict[str, list[dict[str, float]]] = {k: [] for k, *_ in methods if k not in ("ok", "af")}
    math_lines: list[str] = []
    cache: dict[tuple[str, str, str], np.ndarray] = {}

    def cached_scorer(name: str, fn, train, test, tag: str):
        key = (name, tag, str(id(train)) + "->" + str(id(test)))
        if key not in cache:
            cache[key] = fn(train, test)
        return cache[key]

    for model, xs in by_model.items():
        hard = [x for x in xs if x["dataset"] in HARD]
        math = [x for x in xs if x["dataset"] == "math-500"]
        by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in hard:
            by_ds[row["dataset"]].append(row)
        for ds, test in by_ds.items():
            train = [x for x in hard if x["dataset"] != ds]
            if not train or not test:
                continue
            recs = {}
            tag = f"{model}|{ds}"
            for key, _name, scorer, mix_only, oracle in methods:
                wrapped = None
                if scorer is score_lin:
                    wrapped = lambda tr, te, _tag=tag: cached_scorer("lin", score_lin, tr, te, _tag + f"|{id(tr)}|{id(te)}")
                elif scorer is score_mlp:
                    wrapped = lambda tr, te, _tag=tag: cached_scorer("mlp", score_mlp, tr, te, _tag + f"|{id(tr)}|{id(te)}")
                recs[key] = run_method(train, test, wrapped, mix_only=mix_only, oracle=oracle)
                summary[key].append(recs[key])
                if key in zw_summary:
                    zw_summary[key].append(
                        run_method(train, test, wrapped, mix_only=mix_only, oracle=oracle, zero_wait=True)
                    )
            lines.append(
                f"| {tp.MODEL_ZH[model]} {tp.DS_ZH[ds]} | "
                + " | ".join(cell(recs[k]) for k, *_ in methods)
                + " |"
            )
        if math and hard:
            recs = {}
            tag = f"{model}|math"
            for key, _name, scorer, mix_only, oracle in methods:
                wrapped = None
                if scorer is score_lin:
                    wrapped = lambda tr, te, _tag=tag: cached_scorer("lin", score_lin, tr, te, _tag + f"|{id(tr)}|{id(te)}")
                elif scorer is score_mlp:
                    wrapped = lambda tr, te, _tag=tag: cached_scorer("mlp", score_mlp, tr, te, _tag + f"|{id(tr)}|{id(te)}")
                recs[key] = run_method(hard, math, wrapped, mix_only=mix_only, oracle=oracle)
            math_lines.append(
                f"| {tp.MODEL_ZH[model]} MATH | "
                + " | ".join(cell(recs[k]) for k, *_ in methods)
                + " |"
            )
    lines += [
        "",
        "## 2. 难集留一合计",
        "",
        "| 方法 | 合计（加权平均 ΔAcc / Δ步；伤几格） |",
        "|---|---|",
    ]
    for key, name, *_ in methods:
        lines.append(f"| {name} | {pool(summary[key])} |")
    lines += [
        "",
        "## 2b. 更严门槛：训练格误杀必须是 0",
        "",
        "Acc 不降若还允许训练上误杀被别的题抬回来，留一测试仍会伤。",
        "这里门槛再加一条：每个训练数据集误杀必须是 0，再尽量少步。",
        "",
        "| 方法 | 合计 |",
        "|---|---|",
    ]
    for key, name, *_ in methods:
        if key in zw_summary:
            lines.append(f"| {name} | {pool(zw_summary[key])} |")
    lines += [
        "",
        "## 3. MATH（难集上选门槛，MATH 不进训练）",
        "",
        "| 集 | 窗末把握 | 把握·只批混合窗 | 线性探针 | 线性·只批混合窗 | 小 MLP | 先知：剩窗已对 | 先知：等于写完终答 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(math_lines)
    lines += [
        "",
        "## 4. 这套评法对训练意味着什么",
        "",
        "- 训练标签仍可以是「试答等于写完终答」。金标只用来报 Acc、选门槛。",
        "- 选头 / 选门槛不要看排序分。看测试难集是否有一条 Acc 不降、步数为负的门。",
        "- 误杀必须接近 0。宁可少开火。混合窗误杀本来就少，只批混合窗是先收口再打分。",
        "- 先知「剩窗已对」是 Acc 不降的上限；先知「等于写完终答」可以伤 Acc，因为写完终答也会错。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
