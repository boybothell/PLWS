#!/usr/bin/env python3
"""Did closing-think confidence rise only on leftover stoppable windows?"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_cand_collapse as cc
import report_leftover_stability as st

TABLE = AE / "tables/commit_probe.md"
SCORE = AE / "results/commit_probe"
JOBS = (("7B MATH", "r1_7b", "math-500"), ("7B GPQA", "r1_7b", "gpqa-diamond"))


def load_scores(dataset: str) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(SCORE.glob(f"r1_7b_{dataset}_s42_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            out[(int(row["question_idx"]), int(row["decision_step"]))] = row
    return out


def attach(pack: dict[str, Any], scores: dict[tuple[int, int], dict[str, Any]]) -> None:
    for q in pack["questions"]:
        for w in q["windows"]:
            rec = scores.get((int(q["qi"]), int(w["step"])))
            if not rec:
                w["commit_conf"] = float("nan")
                w["lift"] = float("nan")
                w["commit_same"] = float("nan")
                continue
            cc_ = float(rec.get("commit_conf"))
            old = rg.finite(w.get("geo") if "geo" in w else rec.get("old_conf"))
            if "geo" not in w:
                w["geo"] = old
            w["commit_conf"] = cc_
            w["lift"] = cc_ - old if cc_ == cc_ and old == old else float("nan")
            w["commit_same"] = float(rg.same(rec.get("commit_answer"), w["ans"]))
            w["commit_answer"] = rec.get("commit_answer")


def fire_end(q: dict[str, Any], field: str, thr: float) -> int | None:
    for w in q["windows"]:
        val = float(w.get(field, float("nan")))
        if val == val and val >= thr:
            return int(w["end"])
    return None


def eval_door(pack: dict[str, Any], field: str, thr: float) -> dict[str, Any]:
    n = len(pack["questions"])
    acc = tok = host_acc = host_tok = 0.0
    n_fire = n_gain = n_hurt = 0
    for q in pack["questions"]:
        host_acc += int(q["host_ok"])
        host_tok += q["host_tok"]
        end = fire_end(q, field, thr)
        if end is None:
            acc += int(q["host_ok"])
            tok += q["host_tok"]
            continue
        sim = rg.pack(q["trials"], q["rows"], end, "rescue", original_tokens=q["orig_tok"])
        ok = cc.gold_ok(sim["answer"], q["gt"], q["original"], q["orig_ok"])
        acc += int(ok)
        tok += sim["tokens"]
        n_fire += 1
        n_gain += int(ok and not q["host_ok"])
        n_hurt += int((not ok) and q["host_ok"])
    return {
        "n": n,
        "threshold": thr,
        "acc": acc / n,
        "tok": tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "d_host_acc": 100.0 * (acc / n - host_acc / n),
        "d_host_tok": tok / n - host_tok / n,
    }


def pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def cell(rec: dict[str, Any]) -> str:
    if rec["n_fire"] == 0:
        return f"{pair(rec['acc'], rec['tok'])}（不开）"
    return (
        f"{pair(rec['acc'], rec['tok'])}"
        f"（{rg.fmt_pp(rec['d_host_acc'])} / {rg.fmt_tok(rec['d_host_tok'])}；"
        f"{rec['n_fire']}火/{rec['n_gain']}救/{rec['n_hurt']}伤）"
    )


def fmt_auc(v: float) -> str:
    return "—" if v != v else f"{v:.3f}"


def both_win(math: dict[str, Any], gpqa: dict[str, Any]) -> bool:
    return (
        math["acc"] + 1e-12 >= math["host_acc"]
        and gpqa["acc"] + 1e-12 >= gpqa["host_acc"]
        and math["tok"] < math["host_tok"] - 1
        and gpqa["tok"] < gpqa["host_tok"] - 1
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    packs = {}
    recs = {}
    for name, model, dataset in JOBS:
        pack = st.load_cell(model, dataset, 42, {})
        scores = load_scores(dataset)
        attach(pack, scores)
        wins = [w for q in pack["questions"] for w in q["windows"]]
        firsts = [q["windows"][0] for q in pack["questions"] if q["windows"]]
        n_ok = sum(1 for w in wins if w.get("commit_conf") == w.get("commit_conf"))
        rec = {
            "name": name,
            "pack": pack,
            "n_left": sum(1 for q in pack["questions"] if q["windows"]),
            "n_win": len(wins),
            "n_ok": n_ok,
        }
        for field in ("commit_conf", "lift", "geo", "commit_same"):
            rec[f"auc_{field}"] = cc.auroc_of(wins, field)[0]
            rec[f"fauc_{field}"] = cc.auroc_of(firsts, field)[0]
            rec[f"{field}_pos"] = cc.rate(wins, field, True)
            rec[f"{field}_neg"] = cc.rate(wins, field, False)
        packs[name] = pack
        recs[name] = rec
        print(
            f"{name} scored={n_ok}/{len(wins)} "
            f"commit {fmt_auc(rec['auc_commit_conf'])} lift {fmt_auc(rec['auc_lift'])} "
            f"old {fmt_auc(rec['auc_geo'])}",
            flush=True,
        )
    xs = []
    for pack in packs.values():
        xs.extend(
            float(w["commit_conf"])
            for q in pack["questions"]
            for w in q["windows"]
            if w.get("commit_conf") == w.get("commit_conf")
        )
    thrs = sorted(set([0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99] + rg.quantiles(xs, n=21)))
    frozen = None
    for thr in thrs:
        m = eval_door(packs["7B MATH"], "commit_conf", thr)
        g = eval_door(packs["7B GPQA"], "commit_conf", thr)
        if both_win(m, g):
            frozen = (thr, m, g)
            break
    lines = [
        "# 剩窗换收口：先关上思考再打把握",
        "",
        "现有试答已经是思路里接「Final Answer + boxed」。这里同一段思路改成先 `</think>`，再写「The final answer is \\boxed」。",
        "只打 7B MATH / 7B GPQA、密探k4 停点前的低把握连答窗。把握仍是 boxed 里词的几何平均。",
        "正类 = 原试答已经等于金标或写完终答。门开火交原来的试答，不重写。",
        "成功线：同一条门槛，两集正确率都不低于密探k4，而且 token 都更少。做不到就停。",
        "",
        "## 分得开吗",
        "",
        "| 集 | 已打窗 | 新把握 对上/没对上 / AUROC | 新−旧 对上/没对上 / AUROC | 旧把握 AUROC | 新答还是原试答 对上/没对上 |",
        "|---|---:|---|---|---|---|",
    ]
    for name in ("7B MATH", "7B GPQA"):
        rec = recs[name]
        row = (
            f"| {name} | {rec['n_ok']} / {rec['n_win']} "
            f"| {rec['commit_conf_pos']:.2f} / {rec['commit_conf_neg']:.2f} / {fmt_auc(rec['auc_commit_conf'])} "
            f"| {rec['lift_pos']:+.2f} / {rec['lift_neg']:+.2f} / {fmt_auc(rec['auc_lift'])} "
            f"| {fmt_auc(rec['auc_geo'])} "
            f"| {rec['commit_same_pos']:.0%} / {rec['commit_same_neg']:.0%} |"
        )
        print(row, flush=True)
        lines.append(row)
    lines += [
        "",
        "## 当门（相对密探k4）",
        "",
        "| 集 | 密探k4 | 冻住的新把握门 | 同集偷看新把握 |",
        "|---|---|---|---|",
    ]
    if frozen is None:
        lines.append("| 两集同一条 | — | **没有**（正确率不降且都省 token） | 见下 |")
        print("no frozen threshold wins both", flush=True)
    else:
        thr, m, g = frozen
        lines.append(f"| 门槛 {thr:.3f} |  |  |  |")
        for name, rec in (("7B MATH", m), ("7B GPQA", g)):
            host = rec
            lines.append(
                f"| {name} | {pair(host['host_acc'], host['host_tok'])} | {cell(host)} | — |"
            )
        print(f"frozen {thr:.3f}", flush=True)
    for name in ("7B MATH", "7B GPQA"):
        pack = packs[name]
        host = eval_door(pack, "commit_conf", float("inf"))
        xs = [
            float(w["commit_conf"])
            for q in pack["questions"]
            for w in q["windows"]
            if w.get("commit_conf") == w.get("commit_conf")
        ]
        points = [eval_door(pack, "commit_conf", thr) for thr in [float("inf")] + rg.quantiles(xs, n=21)]
        ok = [p for p in points if p["acc"] + 1e-12 >= host["host_acc"]]
        peek = min(ok, key=lambda p: (p["tok"], -p["acc"])) if ok else host
        row = f"| {name} | {pair(host['host_acc'], host['host_tok'])} | 见上 | {cell(peek)} |"
        print(row, flush=True)
        lines.append(row)
    if frozen is None:
        lines += [
            "",
            "**结论：换收口也冻不成两集都涨。这条停。**",
        ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
