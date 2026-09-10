#!/usr/bin/env python3
"""剩窗批准头：隐状态线性探针，标签 = 试答是否等于写完终答。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low

WAIT = AE / "results/leftover_waithelp"
TABLE = AE / "tables/leftover_commit_probe.md"
MODEL_ZH = {
    "r1_7b": "7B",
    "nemotron_8b": "8B",
    "r1_14b": "14B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}
DS_ZH = {
    "math-500": "MATH",
    "olympiadbench": "奥赛",
    "gpqa-diamond": "GPQA",
    "aime24": "AIME24",
    "aime25": "AIME25",
}
C = 0.1
MIN_POS = 8
MIN_NEG = 8


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell_name(row: dict[str, Any]) -> str:
    return f"{MODEL_ZH[row['model']]} {DS_ZH[row['dataset']]}"


def fmt(v: float) -> str:
    return "—" if v != v else f"{v:.2f}"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos = [float(a) for a, t in zip(s, y) if t == 1 and a == a]
    neg = [float(a) for a, t in zip(s, y) if t == 0 and a == a]
    return rg.auroc(pos, neg)


def g_lookup() -> dict[tuple[str, str, int, int], dict[str, Any]]:
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for model in MODEL_ZH:
        for dataset in DS_ZH:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            for seed in seeds:
                path = low.gpath(model, dataset, seed)
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text())
                if isinstance(payload, dict):
                    payload = payload.get("per_sample") or payload.get("rows") or []
                for row in payload:
                    if isinstance(row, dict) and "question_idx" in row:
                        out[(model, dataset, int(seed), int(row["question_idx"]))] = row
    return out


def load_scored() -> list[dict[str, Any]]:
    golds = g_lookup()
    rows: list[dict[str, Any]] = []
    for path in sorted(WAIT.glob("*_shard*.jsonl")):
        last_p = path.with_name(f"hidden_last_{path.stem}.npy")
        pre_p = path.with_name(f"hidden_pre_{path.stem}.npy")
        if not last_p.is_file() or not pre_p.is_file():
            continue
        last = np.load(last_p)
        pre = np.load(pre_p)
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            i = int(row.get("hidden_idx", -1))
            if i < 0 or i >= len(last) or i >= len(pre):
                continue
            key = (row["model"], row["dataset"], int(row["seed"]), int(row["question_idx"]))
            g = golds.get(key) or {}
            a_final = g.get("A_final")
            committed = bool(a_final) and rg.same(row.get("answer"), a_final)
            will_change = bool(row.get("high_later") and not row.get("same_as_high"))
            n_steps = int(g.get("n_steps") or 0)
            high_step = row.get("high_step")
            keep_tok = int(high_step) if high_step else (n_steps or int(row["decision_step"]))
            rec = dict(row)
            rec["a_final"] = a_final
            rec["committed"] = committed
            rec["will_change"] = will_change
            rec["keep_tok"] = keep_tok
            rec["h_last"] = last[i].astype(np.float32)
            rec["h_pre"] = pre[i].astype(np.float32)
            rec["_has_af"] = bool(a_final)
            rows.append(rec)
    return rows


def feat_last(xs: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([x["h_last"] for x in xs])


def feat_pre(xs: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([x["h_pre"] for x in xs])


def feat_delta(xs: list[dict[str, Any]]) -> np.ndarray:
    return feat_last(xs) - feat_pre(xs)


def feat_concat(xs: list[dict[str, Any]]) -> np.ndarray:
    return np.concatenate([feat_last(xs), feat_pre(xs)], axis=1)


def l2(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norm, 1e-8, None)


def mass_mean(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    pos = train_x[train_y == 1]
    neg = train_x[train_y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.full(len(test_x), np.nan)
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    scale = float(np.linalg.norm(direction))
    if scale < 1e-8:
        return np.full(len(test_x), np.nan)
    return test_x @ (direction / scale)


def logreg(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    if train_y.min() == train_y.max() or int(train_y.sum()) < MIN_POS or int((1 - train_y).sum()) < MIN_NEG:
        return np.full(len(test_x), np.nan)
    scaler = StandardScaler()
    x0 = scaler.fit_transform(l2(train_x))
    x1 = scaler.transform(l2(test_x))
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=400,
        solver="lbfgs",
    )
    model.fit(x0, train_y)
    return model.predict_proba(x1)[:, 1]


def pick_tau(scores: np.ndarray, y_acc_fire: np.ndarray, y_acc_keep: np.ndarray, tok_fire: np.ndarray, tok_keep: np.ndarray) -> float:
    """训练上：Acc 不低于全留密探，再尽量省 token。"""
    keep_acc = float(y_acc_keep.mean()) if len(y_acc_keep) else 0.0
    finite = scores[scores == scores]
    if len(finite) == 0:
        return float("inf")
    cands = np.unique(np.quantile(finite, np.linspace(0.05, 0.95, 19)))
    best = (float("inf"), 0.0, 0.0)  # tau, acc, -tok
    found = False
    for tau in cands:
        fire = scores >= tau
        acc = np.where(fire, y_acc_fire, y_acc_keep).mean()
        tok = np.where(fire, tok_fire, tok_keep).mean()
        if acc + 1e-12 < keep_acc:
            continue
        key = (acc, -tok)
        if not found or key > (best[1], best[2]):
            best = (float(tau), float(acc), float(-tok))
            found = True
    return best[0] if found else float("inf")


def apply_gate(scores: np.ndarray, tau: float, y_fire: np.ndarray, y_keep: np.ndarray, tok_fire: np.ndarray, tok_keep: np.ndarray) -> tuple[float, float, int]:
    fire = scores >= tau
    if len(scores) == 0:
        return float("nan"), float("nan"), 0
    acc = float(np.where(fire, y_fire, y_keep).mean())
    tok = float(np.where(fire, tok_fire, tok_keep).mean())
    return acc, tok, int(fire.sum())


def eval_split(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    feat_fn,
    y_key: str,
) -> dict[str, float]:
    if not train or not test:
        return {"auroc": float("nan")}
    y_train = np.asarray([int(x[y_key]) for x in train], dtype=np.int32)
    y_test = np.asarray([int(x[y_key]) for x in test], dtype=np.int32)
    if y_test.min() == y_test.max() or y_train.min() == y_train.max():
        return {"auroc": float("nan")}
    x_train = feat_fn(train)
    x_test = feat_fn(test)
    scores = logreg(x_train, y_train, x_test)
    mm = mass_mean(l2(x_train), y_train, l2(x_test))
    out = {
        "auroc": auroc(y_test, scores),
        "auroc_mm": auroc(y_test, mm),
        "n_test": float(len(test)),
        "pos": float(y_test.sum()),
    }
    if y_key == "committed":
        conf = np.asarray([rg.finite(x.get("confidence")) for x in test])
        out["auroc_conf"] = auroc(y_test, conf)
        y_fire = np.asarray([int(x["leftover_ok"]) for x in test], dtype=np.float64)
        y_keep = np.asarray([int(x["nofs_ok"]) for x in test], dtype=np.float64)
        tok_fire = np.asarray([int(x["decision_step"]) for x in test], dtype=np.float64)
        tok_keep = np.asarray([int(x["keep_tok"]) for x in test], dtype=np.float64)
        rng = np.random.RandomState(0)
        n_tr = len(train)
        perm = rng.permutation(n_tr)
        cut = max(MIN_POS + MIN_NEG, n_tr * 3 // 4)
        fit_idx, val_idx = perm[:cut], perm[cut:]
        if len(val_idx) < 8:
            val_idx = perm
            fit_idx = perm
        fit = [train[i] for i in fit_idx]
        val = [train[i] for i in val_idx]
        y_fit = np.asarray([int(x[y_key]) for x in fit], dtype=np.int32)
        val_scores = logreg(feat_fn(fit), y_fit, feat_fn(val))
        tau = pick_tau(
            val_scores,
            np.asarray([int(x["leftover_ok"]) for x in val], dtype=np.float64),
            np.asarray([int(x["nofs_ok"]) for x in val], dtype=np.float64),
            np.asarray([int(x["decision_step"]) for x in val], dtype=np.float64),
            np.asarray([int(x["keep_tok"]) for x in val], dtype=np.float64),
        )
        acc, tok, n_fire = apply_gate(scores, tau, y_fire, y_keep, tok_fire, tok_keep)
        k4_acc = float(y_keep.mean())
        k4_tok = float(tok_keep.mean())
        oracle_fire = y_test.astype(bool)
        o_acc = float(np.where(oracle_fire, y_fire, y_keep).mean())
        o_tok = float(np.where(oracle_fire, tok_fire, tok_keep).mean())
        out.update(
            {
                "tau": tau,
                "acc": acc,
                "tok": tok,
                "n_fire": float(n_fire),
                "k4_acc": k4_acc,
                "k4_tok": k4_tok,
                "d_acc": 100.0 * (acc - k4_acc),
                "d_tok": tok - k4_tok,
                "o_acc": o_acc,
                "o_tok": o_tok,
                "o_d_acc": 100.0 * (o_acc - k4_acc),
                "o_d_tok": o_tok - k4_tok,
            }
        )
    return out


def group_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[row["model"]].append(row)
    return by


def loto_dataset(rows: list[dict[str, Any]], feat_fn, y_key: str) -> dict[str, dict[str, float]]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    out = {}
    for ds, test in by_ds.items():
        train = [x for x in rows if x["dataset"] != ds]
        out[ds] = eval_split(train, test, feat_fn, y_key)
    return out


def oof_model(rows: list[dict[str, Any]], feat_fn, y_key: str, folds: int = 5) -> dict[str, float]:
    if len(rows) < 20:
        return {"auroc": float("nan")}
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for row in rows:
        buckets[int(row["question_idx"]) % folds].append(row)
    ys = []
    ss = []
    confs = []
    y_fire = []
    y_keep = []
    tok_fire = []
    tok_keep = []
    n_fire = 0
    accs = []
    toks = []
    k4_accs = []
    k4_toks = []
    for i in range(folds):
        test = buckets[i]
        train = [x for j, b in enumerate(buckets) if j != i for x in b]
        rec = eval_split(train, test, feat_fn, y_key)
        if rec["auroc"] != rec["auroc"]:
            continue
        y = np.asarray([int(x[y_key]) for x in test])
        x_train = feat_fn(train)
        x_test = feat_fn(test)
        y_train = np.asarray([int(x[y_key]) for x in train], dtype=np.int32)
        scores = logreg(x_train, y_train, x_test)
        ys.append(y)
        ss.append(scores)
        confs.append(np.asarray([rg.finite(x.get("confidence")) for x in test]))
        if y_key == "committed" and "acc" in rec:
            n_fire += int(rec["n_fire"])
            accs.append(rec["acc"] * len(test))
            toks.append(rec["tok"] * len(test))
            k4_accs.append(rec["k4_acc"] * len(test))
            k4_toks.append(rec["k4_tok"] * len(test))
            y_fire.append(np.asarray([int(x["leftover_ok"]) for x in test]))
            y_keep.append(np.asarray([int(x["nofs_ok"]) for x in test]))
            tok_fire.append(np.asarray([int(x["decision_step"]) for x in test]))
            tok_keep.append(np.asarray([int(x["keep_tok"]) for x in test]))
    if not ys:
        return {"auroc": float("nan")}
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    out = {
        "auroc": auroc(y, s),
        "auroc_conf": auroc(y, np.concatenate(confs)),
        "n_test": float(len(y)),
        "pos": float(y.sum()),
    }
    if accs:
        n = sum(len(a) for a in y_fire)
        acc = sum(accs) / n
        tok = sum(toks) / n
        k4_acc = sum(k4_accs) / n
        k4_tok = sum(k4_toks) / n
        y_all = np.concatenate(ys)
        yf = np.concatenate(y_fire)
        yk = np.concatenate(y_keep)
        tf = np.concatenate(tok_fire)
        tk = np.concatenate(tok_keep)
        o_acc = float(np.where(y_all.astype(bool), yf, yk).mean())
        o_tok = float(np.where(y_all.astype(bool), tf, tk).mean())
        out.update(
            {
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
    return out


def line_loto(name: str, rec: dict[str, float]) -> str:
    if rec.get("auroc") != rec.get("auroc"):
        return f"| {name} | — | — | — | — | — |"
    gate = "—"
    if "d_acc" in rec and rec["d_acc"] == rec["d_acc"]:
        gate = (
            f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f} "
            f"（开火 {int(rec['n_fire'])}/{int(rec['n_test'])}）"
        )
    oracle = "—"
    if "o_d_acc" in rec and rec["o_d_acc"] == rec["o_d_acc"]:
        oracle = f"{rec['o_d_acc']:+.1f}pp / {rec['o_d_tok']:+.0f}"
    return (
        f"| {name} | {int(rec['pos'])}/{int(rec['n_test'] - rec['pos'])} | "
        f"{fmt(rec.get('auroc_conf', float('nan')))} | {fmt(rec['auroc'])} | "
        f"{gate} | {oracle} |"
    )


def main() -> None:
    rows = [r for r in load_scored() if r["_has_af"]]
    print(f"scored leftover with A_final: {len(rows)}", flush=True)
    by_model = group_model(rows)
    lines = [
        "# 剩窗批准头（线性探针）",
        "",
        "决策集只有第一扇非高把握同答窗。正类 = 这扇试答已经等于写完终答。",
        "探针看 boxed 末隐状态，标准化后做逻辑回归。把握只当对照，不当特征。",
        "每模型各自一头。留一数据集 = 别人集上训、这集上测。",
        "五折 = 同模型内按题号折。",
        "Acc / token 只在剩窗题上相对「这题继续走密探k4」（交这扇用试答对错，token 用步数近似）。",
        "",
        "## 1. 标签",
        "",
        "| 集 | 剩窗 | 真承诺 | 假平台 | 再等会更好 | 后面换答 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[cell_name(row)].append(row)
    for name in sorted(by_cell):
        xs = by_cell[name]
        n = len(xs)
        pos = sum(1 for x in xs if x["committed"])
        helps = sum(1 for x in xs if x["wait_helps"])
        chg = sum(1 for x in xs if x["will_change"])
        lines.append(f"| {name} | {n} | {pos}（{100.0 * pos / n:.0f}%） | {n - pos} | {helps} | {chg} |")
    n = len(rows)
    pos = sum(1 for x in rows if x["committed"])
    lines.append(
        f"| 全体 | {n} | {pos}（{100.0 * pos / n:.0f}%） | {n - pos} | "
        f"{sum(1 for x in rows if x['wait_helps'])} | "
        f"{sum(1 for x in rows if x['will_change'])} |"
    )

    feats = (
        ("last", "boxed 末", feat_last),
        ("pre", "boxed 前", feat_pre),
        ("delta", "末−前", feat_delta),
    )
    lines += [
        "",
        "## 2. 同模型五折：能不能分开真承诺",
        "",
        "AUROC 越大越像真承诺。对照是窗末把握。",
        "",
        "| 模型 | 真承诺/假平台 | 把握 | boxed 末探针 | boxed 前 | 末−前 | 质心差 |",
        "|---|---|---|---|---|---|---|",
    ]
    oof_last: dict[str, dict[str, float]] = {}
    for model, xs in by_model.items():
        y = np.asarray([int(x["committed"]) for x in xs])
        recs = {name: oof_model(xs, fn, "committed") for name, _, fn in feats}
        oof_last[model] = recs["last"]
        mm = oof_model(xs, feat_last, "committed")
        # mass-mean AUROC via last oof already has logreg; compute mm separately
        scores_mm = []
        ys = []
        buckets: list[list[dict[str, Any]]] = [[] for _ in range(5)]
        for row in xs:
            buckets[int(row["question_idx"]) % 5].append(row)
        for i in range(5):
            test = buckets[i]
            train = [x for j, b in enumerate(buckets) if j != i for x in b]
            if not test or not train:
                continue
            y_tr = np.asarray([int(x["committed"]) for x in train])
            y_te = np.asarray([int(x["committed"]) for x in test])
            if y_tr.min() == y_tr.max() or y_te.min() == y_te.max():
                continue
            scores_mm.append(mass_mean(l2(feat_last(train)), y_tr, l2(feat_last(test))))
            ys.append(y_te)
        mm_auc = auroc(np.concatenate(ys), np.concatenate(scores_mm)) if ys else float("nan")
        last = recs["last"]
        pre = recs["pre"]
        delta = recs["delta"]
        lines.append(
            f"| {MODEL_ZH[model]} | {int(y.sum())}/{int(len(y) - y.sum())} | "
            f"{fmt(last.get('auroc_conf', float('nan')))} | {fmt(last['auroc'])} | "
            f"{fmt(pre['auroc'])} | {fmt(delta['auroc'])} | {fmt(mm_auc)} |"
        )

    lines += [
        "",
        "## 3. 留一数据集（boxed 末探针，标签 = 真承诺）",
        "",
        "别人集上训，这集上测。门：训练上 Acc 不低于「剩窗全留密探」，再省步数。",
        "Δ 是相对这批剩窗题继续走密探。先知 = 只切真承诺。",
        "",
        "| 集 | 真承诺/假平台 | 把握 | 探针 | 探针门 ΔAcc/步 | 先知 ΔAcc/步 |",
        "|---|---|---|---|---|---|",
    ]
    loto_rows = []
    for model, xs in by_model.items():
        recs = loto_dataset(xs, feat_last, "committed")
        for ds, rec in recs.items():
            name = f"{MODEL_ZH[model]} {DS_ZH[ds]}"
            loto_rows.append((name, rec))
            lines.append(line_loto(name, rec))

    lines += [
        "",
        "## 4. 同模型五折上门（boxed 末）",
        "",
        "| 模型 | 把握 AUROC | 探针 AUROC | 探针门 ΔAcc/步 | 先知 ΔAcc/步 |",
        "|---|---|---|---|---|",
    ]
    for model, rec in oof_last.items():
        gate = "—"
        oracle = "—"
        if rec.get("d_acc") == rec.get("d_acc"):
            gate = f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f}（开火 {int(rec['n_fire'])}/{int(rec['n_test'])}）"
        if rec.get("o_d_acc") == rec.get("o_d_acc"):
            oracle = f"{rec['o_d_acc']:+.1f}pp / {rec['o_d_tok']:+.0f}"
        lines.append(
            f"| {MODEL_ZH[model]} | {fmt(rec.get('auroc_conf', float('nan')))} | "
            f"{fmt(rec.get('auroc', float('nan')))} | {gate} | {oracle} |"
        )

    # Extra: leftover_ok / wait_helps / will_change OOF last
    lines += [
        "",
        "## 5. 同一套五折，换标签（boxed 末探针）",
        "",
        "看探针到底在分哪件事。AUROC 对应该列的正类。",
        "",
        "| 模型 | 真承诺 | 试答对金标或写完 | 再等会更好 | 后面换答 |",
        "|---|---|---|---|---|",
    ]
    for model, xs in by_model.items():
        cells = []
        for key in ("committed", "leftover_ok", "wait_helps", "will_change"):
            rec = oof_model(xs, feat_last, key)
            cells.append(fmt(rec.get("auroc", float("nan"))))
        lines.append(f"| {MODEL_ZH[model]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 6. 只看混合窗 / 只看低把握窗（留一数据集，boxed 末）",
        "",
        "混合窗误杀本来就低。低把握才是假平台大头。",
        "",
        "| 集 | 混合 把握/探针 | 低把握 把握/探针 |",
        "|---|---|---|",
    ]
    for model, xs in by_model.items():
        mix = [x for x in xs if x.get("kind") == "mix"]
        low_xs = [x for x in xs if x.get("kind") == "low"]
        mix_loto = loto_dataset(mix, feat_last, "committed") if mix else {}
        low_loto = loto_dataset(low_xs, feat_last, "committed") if low_xs else {}
        datasets = sorted(set(x["dataset"] for x in xs))
        for ds in datasets:
            name = f"{MODEL_ZH[model]} {DS_ZH[ds]}"
            m = mix_loto.get(ds) or {}
            lo = low_loto.get(ds) or {}
            mix_cell = (
                f"{fmt(m.get('auroc_conf', float('nan')))} / {fmt(m.get('auroc', float('nan')))}"
                if m
                else "—"
            )
            low_cell = (
                f"{fmt(lo.get('auroc_conf', float('nan')))} / {fmt(lo.get('auroc', float('nan')))}"
                if lo
                else "—"
            )
            lines.append(f"| {name} | {mix_cell} | {low_cell} |")

    lines += [
        "",
        "读法：探针要在各集留一上都明显高于把握，且门相对密探不伤 Acc，才算开成。",
        "贴 0.5 或只在同分布五折亮、留一灭，都算没过。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
