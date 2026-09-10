#!/usr/bin/env python3
"""A_final lock = same answer as writing full. Acc vs gold stays, tokens drop."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg

TABLE = AE / "tables/afinal_lock_ceiling.md"
MODELS = (("7B", "r1_7b"), ("8B", "nemotron_8b"))
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def first_af_end(rows: list[dict[str, Any]], a_final: Any, want: str) -> int | None:
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        step = int(row["stopped_len"])
        if step < rg.MSS:
            continue
        window = rows[end + 1 - rg.K : end + 1]
        if not all(rg.same(x.get("final_answer"), a_final) for x in window):
            continue
        confs = rg.confs_of(window)
        high = rg.is_high(confs)
        low = (not high) and rg.is_low(confs)
        if want == "any":
            return end
        if want == "high" and high:
            return end
        if want == "low" and low:
            return end
    return None


def eval_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dd.trial_path(model, dataset, seed)
    if dataset in ("aime24", "aime25"):
        gpath = AE / f"results/dense_G_{model}/{dataset}/seed_{seed}/per_sample.json"
    else:
        gpath = AE / f"results/dense_G_{model}/{dataset}/per_sample.json"
    if not puma_path.is_file() or not trials_path.is_file() or not gpath.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gpath)}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    n = orig_acc = puma_acc = any_acc = high_acc = low_acc = 0
    orig_tok = puma_tok = any_tok = high_tok = low_tok = 0.0
    n_any = n_high = n_low = n_any_wrong = 0
    for qi, info in sorted(official.items()):
        trials = by.get(qi)
        gg = gmap.get(qi)
        if not trials or not gg:
            continue
        a_final = gg.get("A_final") or info.get("original_answer")
        rows = rg.usable_rows(trials)
        orig_ok = bool(info.get("original_correct"))
        orig_t = int(info.get("original_tokens") or 0)
        puma_ok = bool(info.get("compressed_correct"))
        puma_t = int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0)
        full = rg.pack(trials, rows, max(len(rows) - 1, 0), "full", original_tokens=orig_t)
        n += 1
        orig_acc += int(orig_ok)
        orig_tok += orig_t
        puma_acc += int(puma_ok)
        puma_tok += puma_t

        def apply(end: int | None) -> tuple[int, float, bool]:
            if end is None:
                return int(orig_ok), full["tokens"], False
            sim = rg.pack(trials, rows, end, "rescue", original_tokens=orig_t)
            return 1 if orig_ok else 0, sim["tokens"], True

        a_ok, a_t, a_hit = apply(first_af_end(rows, a_final, "any"))
        h_ok, h_t, h_hit = apply(first_af_end(rows, a_final, "high"))
        l_ok, l_t, l_hit = apply(first_af_end(rows, a_final, "low"))
        any_acc += a_ok
        any_tok += a_t
        high_acc += h_ok
        high_tok += h_t
        low_acc += l_ok
        low_tok += l_t
        n_any += int(a_hit)
        n_high += int(h_hit)
        n_low += int(l_hit)
        if a_hit and not orig_ok:
            n_any_wrong += 1
    if n == 0:
        return None
    return {
        "n": n,
        "orig_acc": orig_acc / n,
        "orig_tok": orig_tok / n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "any_acc": any_acc / n,
        "any_tok": any_tok / n,
        "high_acc": high_acc / n,
        "high_tok": high_tok / n,
        "low_acc": low_acc / n,
        "low_tok": low_tok / n,
        "n_any": n_any,
        "n_high": n_high,
        "n_low": n_low,
        "n_any_wrong": n_any_wrong,
        "d_any_acc": 100.0 * (any_acc / n - orig_acc / n),
        "d_any_tok": any_tok / n - orig_tok / n,
        "d_low_tok": low_tok / n - orig_tok / n,
        "d_high_tok": high_tok / n - orig_tok / n,
    }


def merge(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(r["n"] for r in recs)
    out: dict[str, Any] = {"n": n}
    for key, val in recs[0].items():
        if key == "n":
            continue
        if key.startswith("n_"):
            out[key] = sum(r[key] for r in recs)
        else:
            out[key] = sum(r[key] * r["n"] for r in recs) / n
    out["d_any_acc"] = 100.0 * (out["any_acc"] - out["orig_acc"])
    out["d_any_tok"] = out["any_tok"] - out["orig_tok"]
    out["d_low_tok"] = out["low_tok"] - out["orig_tok"]
    out["d_high_tok"] = out["high_tok"] - out["orig_tok"]
    return out


def fmt(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    header = (
        "| 集 | 写完全程 | 第一扇锁成写完 | 只高置信锁 | 只低置信锁 | 锁上但金标错 |"
    )
    sep = "|---|---|---|---|---|---:|"
    lines = [
        "# 跟写完终答一样：正确率不变，只省 token",
        "",
        "正类 = 连续 4 步试答已经等于写完全程的终答。交这步试答，对标准答案的对错和写完相同。",
        "不是正确率上限。相对写完，正确率差应为 0，token 为负才是这项目标。",
        "没锁上的题写完。k=4、0.995 只用来分高/低置信窗。AIME 四个 seed。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            if dataset in ("aime24", "aime25"):
                recs = [eval_cell(model, dataset, seed) for seed in dd.AIME_SEEDS]
                recs = [r for r in recs if r]
                if len(recs) != 4:
                    print(f"skip {zh} {ds_zh} seeds={len(recs)}/4", flush=True)
                    continue
                rec = merge(recs)
            else:
                rec = eval_cell(model, dataset, 42)
                if rec is None:
                    print(f"skip {zh} {ds_zh}", flush=True)
                    continue
            row = (
                f"| {zh} {ds_zh} | {fmt(rec['orig_acc'], rec['orig_tok'])} "
                f"| {fmt(rec['any_acc'], rec['any_tok'])}（{rg.fmt_pp(rec['d_any_acc'])} / {rg.fmt_tok(rec['d_any_tok'])}；{rec['n_any']}/{rec['n']}） "
                f"| {fmt(rec['high_acc'], rec['high_tok'])}（{rg.fmt_tok(rec['d_high_tok'])}；{rec['n_high']}/{rec['n']}） "
                f"| {fmt(rec['low_acc'], rec['low_tok'])}（{rg.fmt_tok(rec['d_low_tok'])}；{rec['n_low']}/{rec['n']}） "
                f"| {rec['n_any_wrong']} |"
            )
            print(row, flush=True)
            lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
