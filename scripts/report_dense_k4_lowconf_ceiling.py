#!/usr/bin/env python3
"""密探k4 + 低置信窗对齐金标或原始CoT 就交试答，不重写。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg

TABLE = AE / "tables/dense_k4/lowconf_af_or_gold.md"
MODELS = (("7B", "r1_7b"), ("8B", "nemotron_8b"))
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def gpath(model: str, dataset: str, seed: int) -> Path:
    if dataset in ("aime24", "aime25", "amc23", "gsm8k") or seed != 42:
        return AE / f"results/dense_G_{model}/{dataset}/seed_{seed}/per_sample.json"
    return AE / f"results/dense_G_{model}/{dataset}/per_sample.json"


def match_full(ans: Any, original_answer: Any, a_final: Any) -> bool:
    return rg.same(ans, original_answer) or rg.same(ans, a_final)


def credit(ans: Any, gt: Any, original_answer: Any, orig_ok: bool) -> int:
    if rg.same(ans, gt):
        return 1
    if rg.same(ans, original_answer):
        return int(orig_ok)
    return 0


def first_low(
    rows: list[dict[str, Any]],
    *,
    gt: Any,
    original_answer: Any,
    a_final: Any,
    host_step: int,
    need_gold: bool,
) -> int | None:
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
        ans = row.get("final_answer")
        hit_gt = rg.same(ans, gt)
        hit_full = match_full(ans, original_answer, a_final)
        if need_gold and not hit_gt:
            continue
        if hit_gt or hit_full:
            return end
    return None


def eval_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    regen_path = dd.regen_stat_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dd.trial_path(model, dataset, seed)
    gp = gpath(model, dataset, seed)
    if not regen_path.is_file() or not puma_path.is_file() or not trials_path.is_file():
        return None
    regen = {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    n = puma_acc = host_acc = rule_acc = gold_acc = safe_acc = 0
    puma_tok = host_tok = rule_tok = gold_tok = safe_tok = 0.0
    n_fire = n_gold = n_af = n_gain = n_hurt = 0
    n_gold_fire = n_safe_fire = n_safe_gain = 0
    for qi, info in sorted(official.items()):
        host = regen.get(qi)
        trials = by.get(qi)
        if not host or not trials:
            continue
        gt = info.get("ground_truth")
        original_answer = info.get("original_answer")
        a_final = (gmap.get(qi) or {}).get("A_final") or original_answer
        orig_ok = bool(info.get("original_correct"))
        orig_t = int(info.get("original_tokens") or 0)
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

        def apply(end: int | None) -> tuple[int, float, bool]:
            if end is None:
                return int(host_ok), host_t, False
            sim = rg.pack(trials, rows, end, "rescue", original_tokens=orig_t)
            return credit(sim["answer"], gt, original_answer, orig_ok), sim["tokens"], True

        r_ok, r_t, r_hit = apply(
            first_low(
                rows, gt=gt, original_answer=original_answer, a_final=a_final,
                host_step=host_step, need_gold=False,
            )
        )
        g_ok, g_t, g_hit = apply(
            first_low(
                rows, gt=gt, original_answer=original_answer, a_final=a_final,
                host_step=host_step, need_gold=True,
            )
        )
        # Acc 相对密探k4 不降：金标，或跟CoT且交出去不会比宿主差
        safe_end = first_low(
            rows, gt=gt, original_answer=original_answer, a_final=a_final,
            host_step=host_step, need_gold=False,
        )
        if safe_end is not None:
            s_ok, s_t, _ = apply(safe_end)
            if s_ok < int(host_ok):
                s_ok, s_t, safe_hit = int(host_ok), host_t, False
            else:
                safe_hit = True
        else:
            s_ok, s_t, safe_hit = int(host_ok), host_t, False

        rule_acc += r_ok
        rule_tok += r_t
        gold_acc += g_ok
        gold_tok += g_t
        safe_acc += s_ok
        safe_tok += s_t
        n_fire += int(r_hit)
        n_gold_fire += int(g_hit)
        n_safe_fire += int(safe_hit)
        if r_hit:
            ans = rows[first_low(
                rows, gt=gt, original_answer=original_answer, a_final=a_final,
                host_step=host_step, need_gold=False,
            )].get("final_answer")
            if rg.same(ans, gt):
                n_gold += 1
            else:
                n_af += 1
        if r_hit and r_ok and not host_ok:
            n_gain += 1
        if r_hit and (not r_ok) and host_ok:
            n_hurt += 1
        if safe_hit and s_ok and not host_ok:
            n_safe_gain += 1
    if n == 0:
        return None
    return {
        "n": n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
        "rule_acc": rule_acc / n,
        "rule_tok": rule_tok / n,
        "gold_acc": gold_acc / n,
        "gold_tok": gold_tok / n,
        "safe_acc": safe_acc / n,
        "safe_tok": safe_tok / n,
        "n_fire": n_fire,
        "n_gold": n_gold,
        "n_af": n_af,
        "n_gain": n_gain,
        "n_hurt": n_hurt,
        "n_gold_fire": n_gold_fire,
        "n_safe_fire": n_safe_fire,
        "n_safe_gain": n_safe_gain,
        "d_rule_host_acc": 100.0 * (rule_acc / n - host_acc / n),
        "d_rule_host_tok": rule_tok / n - host_tok / n,
        "d_rule_puma_acc": 100.0 * (rule_acc / n - puma_acc / n),
        "d_rule_puma_tok": rule_tok / n - puma_tok / n,
        "d_gold_host_acc": 100.0 * (gold_acc / n - host_acc / n),
        "d_gold_host_tok": gold_tok / n - host_tok / n,
        "d_safe_host_acc": 100.0 * (safe_acc / n - host_acc / n),
        "d_safe_host_tok": safe_tok / n - host_tok / n,
        "d_safe_puma_acc": 100.0 * (safe_acc / n - puma_acc / n),
        "d_safe_puma_tok": safe_tok / n - puma_tok / n,
    }


def merge(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(r["n"] for r in recs)
    out: dict[str, Any] = {"n": n}
    for key in recs[0]:
        if key == "n" or key.startswith("d_"):
            continue
        if key.startswith("n_"):
            out[key] = sum(r[key] for r in recs)
        else:
            out[key] = sum(r[key] * r["n"] for r in recs) / n
    out["d_rule_host_acc"] = 100.0 * (out["rule_acc"] - out["host_acc"])
    out["d_rule_host_tok"] = out["rule_tok"] - out["host_tok"]
    out["d_rule_puma_acc"] = 100.0 * (out["rule_acc"] - out["puma_acc"])
    out["d_rule_puma_tok"] = out["rule_tok"] - out["puma_tok"]
    out["d_gold_host_acc"] = 100.0 * (out["gold_acc"] - out["host_acc"])
    out["d_gold_host_tok"] = out["gold_tok"] - out["host_tok"]
    out["d_safe_host_acc"] = 100.0 * (out["safe_acc"] - out["host_acc"])
    out["d_safe_host_tok"] = out["safe_tok"] - out["host_tok"]
    out["d_safe_puma_acc"] = 100.0 * (out["safe_acc"] - out["puma_acc"])
    out["d_safe_puma_tok"] = out["safe_tok"] - out["puma_tok"]
    return out


def pair(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    raise SystemExit(
        "abandoned diagnostic: do not regenerate dense-gate tables. "
        "Canonical comparison is scripts/report_fullcot_puma_plws.py"
    )
    rg.K = 4
    rg.TAU = 0.995
    header = (
        "| 集 | PUMA | 密探k4 | 上限（金标或CoT） | 相对密探k4 | 相对PUMA | "
        "提前交 / 金标 / 只跟CoT / 救回 / 伤 |"
    )
    sep = "|---|---|---|---|---|---|---:|"
    lines = [
        "# 密探k4 + 低置信对齐金标或原始CoT（不重写）",
        "",
        "宿主是已存密探k4：高置信连答 / 后路照旧，截断后重写。",
        "上限：密探k4 停点之前，若有低置信连答窗（连续 4 步同一答案、把握都 < 0.995），",
        "且试答等于金标或等于原始CoT 终答，先知交这步试答，不再重写。否则留密探k4 重写。",
        "Acc 只对金标；交跟原始CoT 同一串时，对错跟官方原始CoT 走。AIME 四个 seed。",
        "",
        header,
        sep,
    ]
    print(header, flush=True)
    print(sep, flush=True)
    extras = [
        "",
        "## 只认金标 / 相对密探k4 Acc 不降",
        "",
        "| 集 | 只认金标 | 金标相对密探k4 | Acc不降（金标或CoT） | Acc不降相对密探k4 | Acc不降相对PUMA |",
        "|---|---|---|---|---|---|",
    ]
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
                f"| {zh} {ds_zh} | {pair(rec['puma_acc'], rec['puma_tok'])} "
                f"| {pair(rec['host_acc'], rec['host_tok'])} "
                f"| {pair(rec['rule_acc'], rec['rule_tok'])} "
                f"| {rg.fmt_pp(rec['d_rule_host_acc'])} / {rg.fmt_tok(rec['d_rule_host_tok'])} "
                f"| {rg.fmt_pp(rec['d_rule_puma_acc'])} / {rg.fmt_tok(rec['d_rule_puma_tok'])} "
                f"| {rec['n_fire']} / {rec['n_gold']} / {rec['n_af']} / {rec['n_gain']} / {rec['n_hurt']} |"
            )
            extra = (
                f"| {zh} {ds_zh} | {pair(rec['gold_acc'], rec['gold_tok'])} "
                f"| {rg.fmt_pp(rec['d_gold_host_acc'])} / {rg.fmt_tok(rec['d_gold_host_tok'])} "
                f"| {pair(rec['safe_acc'], rec['safe_tok'])} "
                f"| {rg.fmt_pp(rec['d_safe_host_acc'])} / {rg.fmt_tok(rec['d_safe_host_tok'])} "
                f"| {rg.fmt_pp(rec['d_safe_puma_acc'])} / {rg.fmt_tok(rec['d_safe_puma_tok'])} |"
            )
            print(row, flush=True)
            lines.append(row)
            extras.append(extra)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines + extras) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
