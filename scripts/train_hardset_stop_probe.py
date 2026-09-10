#!/usr/bin/env python3
"""难集剩窗：去掉 MATH，标签=可停（金标或写完），留一难集。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import train_leftover_commit_probe as tp

TABLE = AE / "tables/hardset_stop_probe.md"
HARD = ("olympiadbench", "gpqa-diamond", "aime24", "aime25")
Y_KEY = "leftover_ok"


def main() -> None:
    rows = [r for r in tp.load_scored() if r["_has_af"] and r["dataset"] in HARD]
    print(f"hard leftover with A_final: {len(rows)}", flush=True)
    by_model = tp.group_model(rows)
    lines = [
        "# 难集剩窗批准头（不含 MATH）",
        "",
        "只留奥赛 / GPQA / AIME。标签 = 试答对金标或写完终答（可停）。",
        "别人难集上训，这集上测。把握只当对照。",
        "门：训练上 Acc 不低于「剩窗全留密探」，再省步数。",
        "",
        "## 1. 同模型五折",
        "",
        "| 模型 | 可停/不可停 | 把握 | 探针 | 门 ΔAcc/步 | 先知 ΔAcc/步 |",
        "|---|---|---|---|---|---|",
    ]
    for model, xs in by_model.items():
        rec = tp.oof_model(xs, tp.feat_last, Y_KEY)
        # oof_model only attaches Acc gate when y_key==committed; compute LOTO-style gate via eval
        rec_c = tp.oof_model(xs, tp.feat_last, "committed")
        y = np.asarray([int(x[Y_KEY]) for x in xs])
        conf = tp.auroc(y, np.asarray([tp.rg.finite(x.get("confidence")) for x in xs]))
        # reuse leftover_ok OOF logreg via 5-fold scores
        buckets: list[list[dict[str, Any]]] = [[] for _ in range(5)]
        for row in xs:
            buckets[int(row["question_idx"]) % 5].append(row)
        ys, ss = [], []
        for i in range(5):
            test = buckets[i]
            train = [x for j, b in enumerate(buckets) if j != i for x in b]
            ev = tp.eval_split(train, test, tp.feat_last, Y_KEY)
            if ev.get("auroc") == ev.get("auroc"):
                y_te = np.asarray([int(x[Y_KEY]) for x in test])
                sc = tp.logreg(
                    tp.feat_last(train),
                    np.asarray([int(x[Y_KEY]) for x in train], dtype=np.int32),
                    tp.feat_last(test),
                )
                ys.append(y_te)
                ss.append(sc)
        auc = tp.auroc(np.concatenate(ys), np.concatenate(ss)) if ys else float("nan")
        lines.append(
            f"| {tp.MODEL_ZH[model]} | {int(y.sum())}/{int(len(y) - y.sum())} | "
            f"{tp.fmt(conf)} | {tp.fmt(auc)} | — | — |"
        )

    lines += [
        "",
        "## 2. 留一难集（boxed 末，标签=可停）",
        "",
        "| 集 | 可停/不可停 | 把握 | 探针 | 探针门 ΔAcc/步 | 先知 ΔAcc/步 |",
        "|---|---|---|---|---|---|",
    ]
    for model, xs in by_model.items():
        recs = tp.loto_dataset(xs, tp.feat_last, "committed")
        # committed gate fields exist; we need leftover_ok LOTO with same gate logic
        recs_ok = {}
        by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in xs:
            by_ds[row["dataset"]].append(row)
        for ds, test in by_ds.items():
            train = [x for x in xs if x["dataset"] != ds]
            recs_ok[ds] = eval_ok(train, test)
        for ds, rec in recs_ok.items():
            name = f"{tp.MODEL_ZH[model]} {tp.DS_ZH[ds]}"
            lines.append(tp.line_loto(name, rec))

    lines += [
        "",
        "读法：去掉 MATH 之后，探针在留一难集上要明显高于把握，且门不伤 Acc，才算这方向能开。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


def eval_ok(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, float]:
    rec = tp.eval_split(train, test, tp.feat_last, "leftover_ok")
    if rec.get("auroc") != rec.get("auroc"):
        return rec
    # eval_split only adds Acc gate for committed; attach it for leftover_ok
    y_test = np.asarray([int(x["leftover_ok"]) for x in test], dtype=np.int32)
    scores = tp.logreg(
        tp.feat_last(train),
        np.asarray([int(x["leftover_ok"]) for x in train], dtype=np.int32),
        tp.feat_last(test),
    )
    rec["auroc"] = tp.auroc(y_test, scores)
    rec["auroc_conf"] = tp.auroc(
        y_test, np.asarray([tp.rg.finite(x.get("confidence")) for x in test])
    )
    rec["n_test"] = float(len(test))
    rec["pos"] = float(y_test.sum())
    y_fire = np.asarray([int(x["leftover_ok"]) for x in test], dtype=np.float64)
    y_keep = np.asarray([int(x["nofs_ok"]) for x in test], dtype=np.float64)
    tok_fire = np.asarray([int(x["decision_step"]) for x in test], dtype=np.float64)
    tok_keep = np.asarray([int(x["keep_tok"]) for x in test], dtype=np.float64)
    rng = np.random.RandomState(0)
    n_tr = len(train)
    perm = rng.permutation(n_tr)
    cut = max(tp.MIN_POS + tp.MIN_NEG, n_tr * 3 // 4)
    fit_idx, val_idx = perm[:cut], perm[cut:]
    if len(val_idx) < 8:
        val_idx = perm
        fit_idx = perm
    fit = [train[i] for i in fit_idx]
    val = [train[i] for i in val_idx]
    val_scores = tp.logreg(
        tp.feat_last(fit),
        np.asarray([int(x["leftover_ok"]) for x in fit], dtype=np.int32),
        tp.feat_last(val),
    )
    tau = tp.pick_tau(
        val_scores,
        np.asarray([int(x["leftover_ok"]) for x in val], dtype=np.float64),
        np.asarray([int(x["nofs_ok"]) for x in val], dtype=np.float64),
        np.asarray([int(x["decision_step"]) for x in val], dtype=np.float64),
        np.asarray([int(x["keep_tok"]) for x in val], dtype=np.float64),
    )
    acc, tok, n_fire = tp.apply_gate(scores, tau, y_fire, y_keep, tok_fire, tok_keep)
    k4_acc = float(y_keep.mean())
    k4_tok = float(tok_keep.mean())
    oracle = y_test.astype(bool)
    o_acc = float(np.where(oracle, y_fire, y_keep).mean())
    o_tok = float(np.where(oracle, tok_fire, tok_keep).mean())
    rec.update(
        {
            "tau": tau,
            "acc": acc,
            "tok": tok,
            "n_fire": float(n_fire),
            "k4_acc": k4_acc,
            "k4_tok": k4_tok,
            "d_acc": 100.0 * (acc - k4_acc),
            "d_tok": tok - k4_tok,
            "o_d_acc": 100.0 * (o_acc - k4_acc),
            "o_d_tok": o_tok - k4_tok,
        }
    )
    return rec


if __name__ == "__main__":
    main()
