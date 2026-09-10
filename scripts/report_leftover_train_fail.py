#!/usr/bin/env python3
"""为什么现有探针训不成：标签、两摊题、把握、省步。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import train_leftover_commit_probe as tp

TABLE = AE / "tables/leftover_train_fail.md"
HARD = ("olympiadbench", "gpqa-diamond", "aime24", "aime25")


def fate(row: dict[str, Any]) -> str:
    if row.get("high_later") and row.get("same_as_high"):
        return "later_same"
    if row.get("high_later") and not row.get("same_as_high"):
        return "later_change"
    return "never_high"


def bucket(row: dict[str, Any]) -> str:
    return f"{row.get('kind')}+{fate(row)}"


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    rows = [r for r in tp.load_scored() if r["_has_af"]]
    hard = [r for r in rows if r["dataset"] in HARD]
    lines = [
        "# 为什么现有探针训不成",
        "",
        "决策集仍是第一扇剩窗。数字只用来解释训练，不当方法。",
        "",
        "## 1. 标签对不齐",
        "",
        "训练正类 = 试答等于写完终答。Acc 要的是试答对金标，且不要误杀（交了错、密探对）。",
        "",
        "| 集 | 剩窗 | 等于写完 | 试答已对 | 两者都是 | 等于写完但试答错 | 其中误杀 | 试答对但后来改口 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hard:
        groups[f"{row['model']}\t{row['dataset']}"].append(row)
    groups["ALL\tALL"] = hard
    for key in sorted(groups):
        xs = groups[key]
        model, ds = key.split("\t")
        name = "难集合计" if model == "ALL" else f"{tp.MODEL_ZH[model]} {tp.DS_ZH[ds]}"
        n = len(xs)
        af = [x for x in xs if x["committed"]]
        ok = [x for x in xs if x["leftover_ok"]]
        both = [x for x in xs if x["committed"] and x["leftover_ok"]]
        af_wrong = [x for x in xs if x["committed"] and not x["leftover_ok"]]
        af_wait = [x for x in af_wrong if x["wait_helps"]]
        ok_change = [x for x in ok if x.get("will_change")]
        lines.append(
            f"| {name} | {n} | {len(af)} | {len(ok)} | {len(both)} | "
            f"{len(af_wrong)} | {len(af_wait)} | {len(ok_change)} |"
        )

    lines += [
        "",
        "## 2. 两摊可停不是同一种窗",
        "",
        "试答已对的剩窗按窗形和后事切开。省步 = 密探步 − 剩窗步。",
        "",
        "| 切片 | 题 | 试答已对 | 误杀率 | 窗末把握 | 已对时省步 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hard:
        by_slice[bucket(row)].append(row)
    for name in (
        "mix+later_same",
        "mix+later_change",
        "mix+never_high",
        "low+later_same",
        "low+later_change",
        "low+never_high",
    ):
        xs = by_slice.get(name) or []
        ok = [x for x in xs if x["leftover_ok"]]
        wait = [x for x in xs if x["wait_helps"]]
        save = [x["keep_tok"] - x["decision_step"] for x in ok]
        conf = [tp.rg.finite(x.get("confidence")) for x in xs]
        label = {
            "mix+later_same": "混合，后面高把握同答",
            "mix+later_change": "混合，后面高把握换答",
            "mix+never_high": "混合，再也没有高把握",
            "low+later_same": "低把握，后面高把握同答",
            "low+later_change": "低把握，后面高把握换答",
            "low+never_high": "低把握，再也没有高把握",
        }[name]
        rate = 100.0 * len(wait) / max(len(xs), 1)
        lines.append(
            f"| {label} | {len(xs)} | {len(ok)} | {rate:.1f}% | "
            f"{mean(conf):.3f} | {mean(save):.1f} |"
        )

    lines += [
        "",
        "## 3. 各数据集的「试答已对」落在哪一摊",
        "",
        "| 集 | 已对·混合马上锁 | 已对·低把握换答前 | 已对·低把握写完 | 误杀·低把握换答 | 误杀·写完才对 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(groups):
        if key == "ALL\tALL":
            continue
        model, ds = key.split("\t")
        xs = groups[key]
        ok_mix = sum(1 for x in xs if x["leftover_ok"] and bucket(x) == "mix+later_same")
        ok_low_ch = sum(1 for x in xs if x["leftover_ok"] and bucket(x) == "low+later_change")
        ok_never = sum(1 for x in xs if x["leftover_ok"] and bucket(x) == "low+never_high")
        w_ch = sum(1 for x in xs if x["wait_helps"] and bucket(x) == "low+later_change")
        w_nv = sum(1 for x in xs if x["wait_helps"] and bucket(x) == "low+never_high")
        lines.append(
            f"| {tp.MODEL_ZH[model]} {tp.DS_ZH[ds]} | {ok_mix} | {ok_low_ch} | "
            f"{ok_never} | {w_ch} | {w_nv} |"
        )

    lines += [
        "",
        "## 4. 线性头是不是在学把握 / 窗形",
        "",
        "同模型五折：探针分数和窗末把握的相关；以及只在混合窗 / 只在低把握窗里，还能不能分「试答已对」。",
        "",
        "| 模型 | 探针↔把握 | 全体·已对 | 混合内·已对 | 低把握内·已对 | 低把握内·误杀 |",
        "|---|---|---|---|---|---|",
    ]
    for model, xs in tp.group_model(hard).items():
        y_ok = np.asarray([int(x["leftover_ok"]) for x in xs])
        y_w = np.asarray([int(x["wait_helps"]) for x in xs])
        conf = np.asarray([tp.rg.finite(x.get("confidence")) for x in xs])
        scores = np.full(len(xs), np.nan)
        # cheap 5-fold on last hidden
        idx = np.arange(len(xs))
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        folds = np.array_split(idx, 5)
        for i, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != i])
            train = [xs[j] for j in tr]
            test = [xs[j] for j in te]
            y_tr = np.asarray([int(x["committed"]) for x in train], dtype=np.int32)
            pred = tp.logreg(tp.feat_last(train), y_tr, tp.feat_last(test))
            scores[te] = pred
        mix = np.asarray([x.get("kind") == "mix" for x in xs])
        low = ~mix
        lines.append(
            f"| {tp.MODEL_ZH[model]} | {corr(scores, conf):.2f} | "
            f"{tp.auroc(y_ok, scores):.2f} | "
            f"{tp.auroc(y_ok[mix], scores[mix]):.2f} | "
            f"{tp.auroc(y_ok[low], scores[low]):.2f} | "
            f"{tp.auroc(y_w[low], scores[low]):.2f} |"
        )

    lines += [
        "",
        "## 5. 同模型里，奥赛正类和 GPQA 正类像不像",
        "",
        "质心：等于写完的剩窗隐状态均值。夹角接近 1 才像同一类。",
        "对照：奥赛「等于写完」对奥赛「后面换答」。",
        "",
        "| 模型 | 奥赛已承诺 vs GPQA已承诺 | 奥赛已承诺 vs 奥赛换答 | GPQA已承诺 把握 | 奥赛已承诺 把握 |",
        "|---|---|---|---|---|",
    ]
    for model, xs in tp.group_model(hard).items():
        oly = [x for x in xs if x["dataset"] == "olympiadbench" and x["committed"]]
        gpq = [x for x in xs if x["dataset"] == "gpqa-diamond" and x["committed"]]
        chg = [x for x in xs if x["dataset"] == "olympiadbench" and x.get("will_change")]
        if len(oly) < 8 or len(gpq) < 8:
            continue
        a = tp.l2(tp.feat_last(oly)).mean(axis=0)
        b = tp.l2(tp.feat_last(gpq)).mean(axis=0)
        c = tp.l2(tp.feat_last(chg)).mean(axis=0) if len(chg) >= 8 else None
        cos_og = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        cos_oc = (
            float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))) if c is not None else float("nan")
        )
        lines.append(
            f"| {tp.MODEL_ZH[model]} | {cos_og:.3f} | {cos_oc:.3f} | "
            f"{mean([tp.rg.finite(x.get('confidence')) for x in gpq]):.3f} | "
            f"{mean([tp.rg.finite(x.get('confidence')) for x in oly]):.3f} |"
        )

    lines += [
        "",
        "## 6. 只看低把握窗：后事三分类在隐状态里有没有",
        "",
        "同模型五折线性头，正类写在列名。若贴 0.5，剩窗当下看不见 19 步后换不换答。",
        "",
        "| 模型 | 低把握窗 | 后面同答 | 后面换答 | 再也没有高把握 | 试答已对 |",
        "|---|---:|---|---|---|---|",
    ]
    for model, xs in tp.group_model(hard).items():
        low = [x for x in xs if x.get("kind") == "low"]
        if len(low) < 40:
            continue
        idx = np.arange(len(low))
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        folds = np.array_split(idx, 5)
        scores = np.full(len(low), np.nan)
        for i, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != i])
            train = [low[j] for j in tr]
            test = [low[j] for j in te]
            y_tr = np.asarray([int(bucket(x) == "low+later_same") for x in train], dtype=np.int32)
            if y_tr.min() == y_tr.max():
                continue
            scores[te] = tp.logreg(tp.feat_last(train), y_tr, tp.feat_last(test))
        y_same = np.asarray([int(bucket(x) == "low+later_same") for x in low])
        y_chg = np.asarray([int(bucket(x) == "low+later_change") for x in low])
        y_nv = np.asarray([int(bucket(x) == "low+never_high") for x in low])
        y_ok = np.asarray([int(x["leftover_ok"]) for x in low])
        # reuse later_same scores to rank other labels only as a cheap check; also fit each
        recs = {}
        for name, y in (("same", y_same), ("chg", y_chg), ("nv", y_nv), ("ok", y_ok)):
            pred = np.full(len(low), np.nan)
            for i, te in enumerate(folds):
                tr = np.concatenate([folds[j] for j in range(5) if j != i])
                train = [low[j] for j in tr]
                test = [low[j] for j in te]
                y_tr = y[tr]
                if y_tr.min() == y_tr.max() or int(y_tr.sum()) < 8 or int((1 - y_tr).sum()) < 8:
                    continue
                pred[te] = tp.logreg(tp.feat_last(train), y_tr.astype(np.int32), tp.feat_last(test))
            recs[name] = tp.auroc(y, pred)
        lines.append(
            f"| {tp.MODEL_ZH[model]} | {len(low)} | {tp.fmt(recs['same'])} | "
            f"{tp.fmt(recs['chg'])} | {tp.fmt(recs['nv'])} | {tp.fmt(recs['ok'])} |"
        )

    lines += [
        "",
        "## 7. 低把握窗留一数据集（防「认出是哪一集」）",
        "",
        "别人集上训、这集上测。若五折有、留一没有，头是在认 GPQA / 奥赛痕迹，不是认后事。",
        "",
        "| 集 | 后面换答 | 再也没有高把握 | 试答已对 |",
        "|---|---|---|---|",
    ]
    for model, xs in tp.group_model(hard).items():
        low = [x for x in xs if x.get("kind") == "low"]
        by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in low:
            by_ds[row["dataset"]].append(row)
        for ds, test in by_ds.items():
            train = [x for x in low if x["dataset"] != ds]
            if len(train) < 30 or len(test) < 15:
                continue
            cells = []
            for pred in (
                lambda x: int(bucket(x) == "low+later_change"),
                lambda x: int(bucket(x) == "low+never_high"),
                lambda x: int(x["leftover_ok"]),
            ):
                y_tr = np.asarray([pred(x) for x in train], dtype=np.int32)
                y_te = np.asarray([pred(x) for x in test], dtype=np.int32)
                if (
                    y_tr.min() == y_tr.max()
                    or y_te.min() == y_te.max()
                    or int(y_tr.sum()) < 8
                    or int((1 - y_tr).sum()) < 8
                ):
                    cells.append("—")
                    continue
                s = tp.logreg(tp.feat_last(train), y_tr, tp.feat_last(test))
                cells.append(tp.fmt(tp.auroc(y_te, s)))
            lines.append(
                f"| {tp.MODEL_ZH[model]} {tp.DS_ZH[ds]} | " + " | ".join(cells) + " |"
            )

    lines += [
        "",
        "读法：奥赛正类更像奥赛换答、而不像 GPQA 正类，一头线性探针留一集必然反。",
        "低把握里后事若五折有、留一没有，就不要再对承诺做 BCE。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
