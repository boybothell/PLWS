#!/usr/bin/env python3
"""终答标签探针：难集上训，MATH 只评估。主损失 BCE，有终答隐状态再加对比。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import train_leftover_commit_probe as tp

TABLE = AE / "tables/afinal_anchor_probe.md"
FINAL = AE / "results/leftover_final_h"
HARD = ("olympiadbench", "gpqa-diamond", "aime24", "aime25")
Y_KEY = "committed"


def load_final_hidden() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not FINAL.is_dir():
        return out
    for path in sorted(FINAL.glob("*_shard*.jsonl")):
        npy = path.with_name(f"hidden_last_{path.stem}.npy")
        if not npy.is_file():
            continue
        last = np.load(npy)
        for row in tp.load_jsonl(path):
            if row.get("status") != "ok":
                continue
            i = int(row.get("hidden_idx", -1))
            if 0 <= i < len(last):
                out[f"{row.get('model')}:{row['uid']}"] = last[i].astype(np.float32)
    return out


def attach_final(rows: list[dict[str, Any]], finals: dict[str, np.ndarray]) -> None:
    for row in rows:
        vec = finals.get(f"{row.get('model')}:{row['uid']}")
        if vec is None:
            continue
        row["h_final"] = vec
        a = row["h_last"]
        b = vec
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        row["cos_final"] = float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")


def mlp_scores(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    use_anchor: bool,
    seed: int = 0,
) -> np.ndarray:
    try:
        import torch
        from torch import nn
    except ImportError:
        return np.full(len(test), np.nan)
    y_train = np.asarray([int(x[Y_KEY]) for x in train], dtype=np.int64)
    if y_train.min() == y_train.max() or int(y_train.sum()) < tp.MIN_POS:
        return np.full(len(test), np.nan)
    x_train = tp.l2(tp.feat_last(train))
    x_test = tp.l2(tp.feat_last(test))
    has_f = use_anchor and all("h_final" in x for x in train)
    f_train = tp.l2(np.stack([x["h_final"] for x in train])) if has_f else None
    device = torch.device("cpu")
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    ft = torch.tensor(f_train, dtype=torch.float32, device=device) if f_train is not None else None
    d = xt.shape[1]
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(y_train.mean())
    weight = torch.where(yt > 0.5, 1.0 / max(pos, 1e-3), 1.0 / max(1.0 - pos, 1e-3))
    net.train()
    for _ in range(80):
        opt.zero_grad()
        logit = net(xt).squeeze(-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt, weight=weight)
        loss = bce
        if ft is not None:
            cos = (xt * ft).sum(dim=-1).clamp(-1, 1)
            aux = (yt * (1.0 - cos) + (1.0 - yt) * torch.relu(cos - 0.1)).mean()
            loss = bce + 0.3 * aux
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.tensor(x_test, dtype=torch.float32))).squeeze(-1).numpy()
    return np.asarray(prob, dtype=np.float64)


def pack_eval(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, float]:
    y = np.asarray([int(x[Y_KEY]) for x in test], dtype=np.int32)
    out = {
        "auroc": tp.auroc(y, scores),
        "n_test": float(len(test)),
        "pos": float(y.sum()),
        "auroc_conf": tp.auroc(y, np.asarray([tp.rg.finite(x.get("confidence")) for x in test])),
    }
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
        np.asarray([int(x[Y_KEY]) for x in fit], dtype=np.int32),
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
    oracle = y.astype(bool)
    o_acc = float(np.where(oracle, y_fire, y_keep).mean())
    o_tok = float(np.where(oracle, tok_fire, tok_keep).mean())
    out.update(
        {
            "d_acc": 100.0 * (acc - k4_acc),
            "d_tok": tok - k4_tok,
            "n_fire": float(n_fire),
            "o_d_acc": 100.0 * (o_acc - k4_acc),
            "o_d_tok": o_tok - k4_tok,
        }
    )
    return out


def line_of(name: str, rec: dict[str, float]) -> str:
    if rec.get("auroc") != rec.get("auroc"):
        return f"| {name} | — | — | — | — | — |"
    gate = "—"
    if rec.get("d_acc") == rec.get("d_acc"):
        gate = f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f}（开火 {int(rec['n_fire'])}/{int(rec['n_test'])}）"
    oracle = "—"
    if rec.get("o_d_acc") == rec.get("o_d_acc"):
        oracle = f"{rec['o_d_acc']:+.1f}pp / {rec['o_d_tok']:+.0f}"
    return (
        f"| {name} | {int(rec['pos'])}/{int(rec['n_test'] - rec['pos'])} | "
        f"{tp.fmt(rec.get('auroc_conf', float('nan')))} | {tp.fmt(rec['auroc'])} | "
        f"{gate} | {oracle} |"
    )


def main() -> None:
    rows = [r for r in tp.load_scored() if r["_has_af"]]
    finals = load_final_hidden()
    attach_final(rows, finals)
    n_final = sum(1 for r in rows if "h_final" in r)
    print(f"leftover={len(rows)} with_final_h={n_final}", flush=True)
    by_model = tp.group_model(rows)
    lines = [
        "# 终答标签探针（难集训，MATH 只评估）",
        "",
        "标签 = 这扇试答是否等于写完终答。不训「会不会换答」。",
        "训练只用奥赛 / GPQA / AIME。MATH 始终不进训练，只当评估。",
        "对照是窗末把握。门：训练上 Acc 不低于剩窗留密探，再省步数。",
        f"写完终答隐状态已对齐 {n_final}/{len(rows)} 扇。",
        "",
        "## 1. 留一难集（线性 BCE）",
        "",
        "| 集 | 终答/非终答 | 把握 | 探针 | 探针门 ΔAcc/步 | 先知 ΔAcc/步 |",
        "|---|---|---|---|---|---|",
    ]
    math_rows: list[tuple[str, dict[str, float]]] = []
    hard_recs: list[tuple[str, dict[str, float]]] = []
    for model, xs in by_model.items():
        hard = [x for x in xs if x["dataset"] in HARD]
        math = [x for x in xs if x["dataset"] == "math-500"]
        by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in hard:
            by_ds[row["dataset"]].append(row)
        for ds, test in by_ds.items():
            train = [x for x in hard if x["dataset"] != ds]
            scores = tp.logreg(
                tp.feat_last(train),
                np.asarray([int(x[Y_KEY]) for x in train], dtype=np.int32),
                tp.feat_last(test),
            )
            rec = pack_eval(train, test, scores)
            name = f"{tp.MODEL_ZH[model]} {tp.DS_ZH[ds]}"
            hard_recs.append((name, rec))
            lines.append(line_of(name, rec))
        if math and hard:
            scores = tp.logreg(
                tp.feat_last(hard),
                np.asarray([int(x[Y_KEY]) for x in hard], dtype=np.int32),
                tp.feat_last(math),
            )
            rec = pack_eval(hard, math, scores)
            math_rows.append((f"{tp.MODEL_ZH[model]} MATH", rec))

    lines += [
        "",
        "## 2. MATH（难集上训，MATH 不进训练）",
        "",
        "| 集 | 终答/非终答 | 把握 | 探针 | 探针门 ΔAcc/步 | 先知 ΔAcc/步 |",
        "|---|---|---|---|---|---|",
    ]
    for name, rec in math_rows:
        lines.append(line_of(name, rec))

    lines += [
        "",
        "## 3. 同模型难集五折（线性 BCE）",
        "",
        "| 模型 | 把握 | 探针 |",
        "|---|---|---|",
    ]
    for model, xs in by_model.items():
        hard = [x for x in xs if x["dataset"] in HARD]
        rec = tp.oof_model(hard, tp.feat_last, Y_KEY)
        lines.append(
            f"| {tp.MODEL_ZH[model]} | {tp.fmt(rec.get('auroc_conf', float('nan')))} | "
            f"{tp.fmt(rec.get('auroc', float('nan')))} |"
        )

    if n_final:
        lines += [
            "",
            "## 4. 偷看终答隐状态（只当诊断，测试用不到）",
            "",
            "分数 = 剩窗 boxed 末 与 写完终答 boxed 末 的夹角。越大越像已经到终答态。",
            "",
            "| 集 | 终答/非终答 | 把握 | 夹角 |",
            "|---|---|---|---|",
        ]
        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if "cos_final" not in row:
                continue
            by_cell[tp.cell_name(row)].append(row)
        for name in sorted(by_cell):
            xs = by_cell[name]
            y = np.asarray([int(x[Y_KEY]) for x in xs])
            lines.append(
                f"| {name} | {int(y.sum())}/{int(len(y) - y.sum())} | "
                f"{tp.fmt(tp.auroc(y, np.asarray([tp.rg.finite(x.get('confidence')) for x in xs])))} | "
                f"{tp.fmt(tp.auroc(y, np.asarray([x['cos_final'] for x in xs])))} |"
            )

        lines += [
            "",
            "## 5. 小 MLP：难集留一，BCE / BCE+终答锚点",
            "",
            "| 集 | 把握 | 线性 | MLP | MLP+锚点 |",
            "|---|---|---|---|---|",
        ]
        for model, xs in by_model.items():
            hard = [x for x in xs if x["dataset"] in HARD]
            by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in hard:
                by_ds[row["dataset"]].append(row)
            for ds, test in by_ds.items():
                train = [x for x in hard if x["dataset"] != ds]
                if not train or not test:
                    continue
                y = np.asarray([int(x[Y_KEY]) for x in test])
                conf = tp.auroc(y, np.asarray([tp.rg.finite(x.get("confidence")) for x in test]))
                lin = tp.logreg(
                    tp.feat_last(train),
                    np.asarray([int(x[Y_KEY]) for x in train], dtype=np.int32),
                    tp.feat_last(test),
                )
                mlp = mlp_scores(train, test, use_anchor=False)
                anc_train = [x for x in train if "h_final" in x]
                anc_test = test
                anc = (
                    mlp_scores(anc_train, anc_test, use_anchor=True)
                    if anc_train and sum("h_final" in x for x in anc_train) >= 16
                    else np.full(len(test), np.nan)
                )
                name = f"{tp.MODEL_ZH[model]} {tp.DS_ZH[ds]}"
                lines.append(
                    f"| {name} | {tp.fmt(conf)} | {tp.fmt(tp.auroc(y, lin))} | "
                    f"{tp.fmt(tp.auroc(y, mlp))} | {tp.fmt(tp.auroc(y, anc))} |"
                )

    lines += [
        "",
        "读法：主结论看第 1 节难集。MATH 是第 2 节，不当主线。",
        "夹角一列若亮，说明终答态锚点有信息，但测试时没有写完隐状态，只能靠 MLP 把这件事学进当前 h。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
