#!/usr/bin/env python3
"""Low-conf gold-lock ceiling on the saved dense+regen host."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg

TABLE = AE / "tables/dense_regen_r_oracle.md"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
)
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def first_gold_low(rows: list[dict[str, Any]], gt: Any, host_step: int) -> int | None:
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        window = rows[end + 1 - rg.K : end + 1]
        confs = rg.confs_of(window)
        if rg.is_high(confs) or not rg.is_low(confs):
            continue
        step = int(row["stopped_len"])
        if step < rg.MSS or step >= host_step:
            continue
        if rg.same(row.get("final_answer"), gt):
            return end
    return None


def eval_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    regen_path = dd.regen_stat_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dd.trial_path(model, dataset, seed)
    if not regen_path.is_file() or not puma_path.is_file() or not trials_path.is_file():
        return None
    regen = {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    n = puma_acc = host_acc = acc = 0
    puma_tok = host_tok = tok = 0.0
    n_fire = n_gain = 0
    for qi, info in sorted(official.items()):
        host = regen.get(qi)
        trials = by.get(qi)
        if not host or not trials:
            continue
        gt = info.get("ground_truth")
        rows = rg.usable_rows(trials)
        host_step = int(host.get("stopped_len") or 10**9)
        host_ok = bool(host.get("compressed_correct"))
        host_t = int(host.get("compressed_tokens") or 0) + int(host.get("tokens_trial_answers") or 0)
        puma_ok = bool(info.get("compressed_correct"))
        puma_t = int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0)
        n += 1
        puma_acc += int(puma_ok)
        puma_tok += puma_t
        host_acc += int(host_ok)
        host_tok += host_t
        end = first_gold_low(rows, gt, host_step)
        if end is None:
            acc += int(host_ok)
            tok += host_t
            continue
        n_fire += 1
        sim = rg.pack(trials, rows, end, "rescue", original_tokens=int(info.get("original_tokens") or 0))
        acc += 1
        tok += sim["tokens"]
        if not host_ok:
            n_gain += 1
    if n == 0:
        return None
    return {
        "n": n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
        "acc": acc / n,
        "tok": tok / n,
        "n_fire": n_fire,
        "n_gain": n_gain,
        "d_host_acc": 100.0 * (acc / n - host_acc / n),
        "d_host_tok": tok / n - host_tok / n,
        "d_puma_acc": 100.0 * (acc / n - puma_acc / n),
        "d_puma_tok": tok / n - puma_tok / n,
    }


def merge(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(r["n"] for r in recs)
    def wavg(key: str) -> float:
        return sum(r[key] * r["n"] for r in recs) / n

    acc = wavg("acc")
    host = wavg("host_acc")
    puma = wavg("puma_acc")
    tok = wavg("tok")
    host_tok = wavg("host_tok")
    puma_tok = wavg("puma_tok")
    return {
        "n": n,
        "puma_acc": puma,
        "puma_tok": puma_tok,
        "host_acc": host,
        "host_tok": host_tok,
        "acc": acc,
        "tok": tok,
        "n_fire": sum(r["n_fire"] for r in recs),
        "n_gain": sum(r["n_gain"] for r in recs),
        "d_host_acc": 100.0 * (acc - host),
        "d_host_tok": tok - host_tok,
        "d_puma_acc": 100.0 * (acc - puma),
        "d_puma_tok": tok - puma_tok,
    }


def fmt_pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    header = "| 集 | 官方 PUMA | 密探+重写 | 低置信上限 | 上限比重写 | 上限比PUMA | 提前交 / 救回 |"
    sep = "|---|---|---|---|---|---|---:|"
    lines = [
        "# 密探+重写上的低置信连对上限（正确率只对金标）",
        "",
        "宿主是已存的密探双闸门：k=4、0.995、后路强停，截断后再重写终答。",
        "上限：停点前若有低置信连答（连续 4 步同一答案、把握都 < 0.995）且试答对标准答案，先知交这步试答；否则留重写终答。",
        "交上限试答时不再重写。若那一步也重写，正确率只会更低或持平。AIME 四个 seed。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    for tag, model in MODELS:
        for zh, dataset in DS:
            if dataset in ("aime24", "aime25"):
                recs = [eval_cell(model, dataset, seed) for seed in dd.AIME_SEEDS]
                recs = [r for r in recs if r]
                if len(recs) != 4:
                    print(f"skip {tag} {zh} seeds={len(recs)}/4", flush=True)
                    continue
                rec = merge(recs)
            else:
                rec = eval_cell(model, dataset, 42)
                if rec is None:
                    print(f"skip {tag} {zh}", flush=True)
                    continue
            name = f"{tag} {zh}"
            row = (
                f"| {name} | {fmt_pair(rec['puma_acc'], rec['puma_tok'])} "
                f"| {fmt_pair(rec['host_acc'], rec['host_tok'])} "
                f"| {fmt_pair(rec['acc'], rec['tok'])} "
                f"| {rg.fmt_pp(rec['d_host_acc'])} / {rg.fmt_tok(rec['d_host_tok'])} "
                f"| {rg.fmt_pp(rec['d_puma_acc'])} / {rg.fmt_tok(rec['d_puma_tok'])} "
                f"| {rec['n_fire']} / {rec['n_gain']} |"
            )
            print(row, flush=True)
            lines.append(row)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
