#!/usr/bin/env python3
"""Two separate gold ceilings: veto wrong high-conf vs fish gold low-conf."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg

TABLE = AE / "tables/two_ceilings_gold.md"
MODELS = (("7B", "r1_7b"), ("8B", "nemotron_8b"))
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)


def windows(rows: list[dict[str, Any]]) -> list[tuple[int, bool, bool]]:
    out = []
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        step = int(row["stopped_len"])
        if step < rg.MSS:
            continue
        confs = rg.confs_of(rows[end + 1 - rg.K : end + 1])
        high = rg.is_high(confs)
        low = (not high) and rg.is_low(confs)
        if high or low:
            out.append((end, high, low))
    return out


def pack_end(trials, rows, end, info) -> dict[str, Any]:
    return rg.pack(
        trials,
        rows,
        end,
        "rescue",
        original_tokens=int(info.get("original_tokens") or 0),
    )


def pack_full(trials, rows, info) -> dict[str, Any]:
    return rg.pack(
        trials,
        rows,
        max(len(rows) - 1, 0),
        "full",
        original_tokens=int(info.get("original_tokens") or 0),
    )


def eval_trial(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dd.trial_path(model, dataset, seed)
    if not puma_path.is_file() or not trials_path.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    n = puma_acc = base_acc = veto_acc = fish_acc = 0
    puma_tok = base_tok = veto_tok = fish_tok = 0.0
    n_high = n_high_wrong = n_veto_gain = n_fish = n_fish_gain = 0
    for qi, info in sorted(official.items()):
        trials = by.get(qi)
        if not trials:
            continue
        gt = info.get("ground_truth")
        rows = rg.usable_rows(trials)
        orig_ok = bool(info.get("original_correct"))
        puma_ok = bool(info.get("compressed_correct"))
        puma_t = int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0)
        evs = windows(rows)
        first_high = next((end for end, high, _ in evs if high), None)
        gold_high = next(
            (end for end, high, _ in evs if high and rg.same(rows[end].get("final_answer"), gt)),
            None,
        )
        gold_low = next(
            (end for end, high, low in evs if low and rg.same(rows[end].get("final_answer"), gt)),
            None,
        )
        full = pack_full(trials, rows, info)
        n += 1
        puma_acc += int(puma_ok)
        puma_tok += puma_t
        # baseline: first high-conf, else full. trial-as-final, no FS
        if first_high is not None:
            sim = pack_end(trials, rows, first_high, info)
            base_acc += int(rg.same(sim["answer"], gt))
            base_tok += sim["tokens"]
            n_high += 1
            if not rg.same(sim["answer"], gt):
                n_high_wrong += 1
        else:
            base_acc += int(orig_ok)
            base_tok += full["tokens"]
        # veto wrong high-conf: first gold high-conf, else full
        if gold_high is not None:
            sim = pack_end(trials, rows, gold_high, info)
            veto_acc += 1
            veto_tok += sim["tokens"]
            if first_high is None or not rg.same(rows[first_high].get("final_answer"), gt):
                n_veto_gain += 1
        else:
            veto_acc += int(orig_ok)
            veto_tok += full["tokens"]
            if first_high is not None and not rg.same(rows[first_high].get("final_answer"), gt) and orig_ok:
                n_veto_gain += 1
        # fish gold low-conf before first high-conf, else same as baseline
        high_step = int(rows[first_high]["stopped_len"]) if first_high is not None else 10**9
        if gold_low is not None and int(rows[gold_low]["stopped_len"]) <= high_step:
            fish_acc += 1
            fish_tok += pack_end(trials, rows, gold_low, info)["tokens"]
            n_fish += 1
            if first_high is None or not rg.same(rows[first_high].get("final_answer"), gt):
                n_fish_gain += 1
        elif first_high is not None:
            sim = pack_end(trials, rows, first_high, info)
            fish_acc += int(rg.same(sim["answer"], gt))
            fish_tok += sim["tokens"]
        else:
            fish_acc += int(orig_ok)
            fish_tok += full["tokens"]
    if n == 0:
        return None
    return {
        "n": n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "base_acc": base_acc / n,
        "base_tok": base_tok / n,
        "veto_acc": veto_acc / n,
        "veto_tok": veto_tok / n,
        "fish_acc": fish_acc / n,
        "fish_tok": fish_tok / n,
        "n_high": n_high,
        "n_high_wrong": n_high_wrong,
        "n_veto_gain": n_veto_gain,
        "n_fish": n_fish,
        "n_fish_gain": n_fish_gain,
        "d_veto": 100.0 * (veto_acc / n - puma_acc / n),
        "d_fish": 100.0 * (fish_acc / n - puma_acc / n),
        "d_veto_base": 100.0 * (veto_acc / n - base_acc / n),
        "d_fish_base": 100.0 * (fish_acc / n - base_acc / n),
    }


def eval_regen(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
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
    n = puma_acc = host_acc = veto_acc = fish_acc = 0
    puma_tok = host_tok = veto_tok = fish_tok = 0.0
    n_early = n_early_wrong = n_veto_gain = n_fish = n_fish_gain = 0
    for qi, info in sorted(official.items()):
        host = regen.get(qi)
        trials = by.get(qi)
        if not host or not trials:
            continue
        gt = info.get("ground_truth")
        rows = rg.usable_rows(trials)
        orig_ok = bool(info.get("original_correct"))
        host_ok = bool(host.get("compressed_correct"))
        host_t = int(host.get("compressed_tokens") or 0) + int(host.get("tokens_trial_answers") or 0)
        puma_ok = bool(info.get("compressed_correct"))
        puma_t = int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0)
        host_step = int(host.get("stopped_len") or 10**9)
        early = str(host.get("stop_reason") or "") not in ("full_reasoning", "")
        full = pack_full(trials, rows, info)
        gold_low = next(
            (
                end
                for end, high, low in windows(rows)
                if low
                and int(rows[end]["stopped_len"]) < host_step
                and rg.same(rows[end].get("final_answer"), gt)
            ),
            None,
        )
        n += 1
        puma_acc += int(puma_ok)
        puma_tok += puma_t
        host_acc += int(host_ok)
        host_tok += host_t
        if early:
            n_early += 1
            if not host_ok:
                n_early_wrong += 1
        # veto: if early rewrite is wrong and writing full is right, write full
        if early and (not host_ok) and orig_ok:
            veto_acc += 1
            veto_tok += full["tokens"]
            n_veto_gain += 1
        else:
            veto_acc += int(host_ok)
            veto_tok += host_t
        # fish gold low-conf before host stop
        if gold_low is not None:
            fish_acc += 1
            fish_tok += pack_end(trials, rows, gold_low, info)["tokens"]
            n_fish += 1
            if not host_ok:
                n_fish_gain += 1
        else:
            fish_acc += int(host_ok)
            fish_tok += host_t
    if n == 0:
        return None
    return {
        "n": n,
        "puma_acc": puma_acc / n,
        "puma_tok": puma_tok / n,
        "host_acc": host_acc / n,
        "host_tok": host_tok / n,
        "veto_acc": veto_acc / n,
        "veto_tok": veto_tok / n,
        "fish_acc": fish_acc / n,
        "fish_tok": fish_tok / n,
        "n_early": n_early,
        "n_early_wrong": n_early_wrong,
        "n_veto_gain": n_veto_gain,
        "n_fish": n_fish,
        "n_fish_gain": n_fish_gain,
        "d_veto_host": 100.0 * (veto_acc / n - host_acc / n),
        "d_fish_host": 100.0 * (fish_acc / n - host_acc / n),
        "d_veto_puma": 100.0 * (veto_acc / n - puma_acc / n),
        "d_fish_puma": 100.0 * (fish_acc / n - puma_acc / n),
    }


def merge(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(r["n"] for r in recs)

    def w(key: str) -> float:
        return sum(r[key] * r["n"] for r in recs) / n

    out = {key: w(key) for key in recs[0] if key != "n" and not key.startswith("n_")}
    out["n"] = n
    for key in recs[0]:
        if key.startswith("n_") and key != "n":
            out[key] = sum(r[key] for r in recs)
    # recompute deltas from weighted acc
    if "base_acc" in recs[0]:
        out["d_veto"] = 100.0 * (out["veto_acc"] - out["puma_acc"])
        out["d_fish"] = 100.0 * (out["fish_acc"] - out["puma_acc"])
        out["d_veto_base"] = 100.0 * (out["veto_acc"] - out["base_acc"])
        out["d_fish_base"] = 100.0 * (out["fish_acc"] - out["base_acc"])
    else:
        out["d_veto_host"] = 100.0 * (out["veto_acc"] - out["host_acc"])
        out["d_fish_host"] = 100.0 * (out["fish_acc"] - out["host_acc"])
        out["d_veto_puma"] = 100.0 * (out["veto_acc"] - out["puma_acc"])
        out["d_fish_puma"] = 100.0 * (out["fish_acc"] - out["puma_acc"])
    return out


def load_all(fn) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            if dataset in ("aime24", "aime25"):
                recs = [fn(model, dataset, seed) for seed in dd.AIME_SEEDS]
                recs = [r for r in recs if r]
                if len(recs) != 4:
                    print(f"skip {zh} {ds_zh} seeds={len(recs)}/4", flush=True)
                    continue
                rec = merge(recs)
            else:
                rec = fn(model, dataset, 42)
                if rec is None:
                    print(f"skip {zh} {ds_zh}", flush=True)
                    continue
            out.append((f"{zh} {ds_zh}", rec))
            print(f"ok {zh} {ds_zh} n={rec['n']}", flush=True)
    return out


def fmt(acc: float, tok: float) -> str:
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    print("=== 试答即终答 ===", flush=True)
    trial = load_all(eval_trial)
    print("=== 密探+重写 ===", flush=True)
    regen = load_all(eval_regen)
    lines = [
        "# 两个上限分开算（正确率只对金标）",
        "",
        "挡住高置信交错：高置信连答若试答/重写对不上标准答案，就当没看见，继续往后；有对的高置信再停，否则写完。",
        "捞低置信对：高置信门照旧（包括会停错的），只额外在停点前交第一扇试答对金标的低置信连答。",
        "两件事分开，不是叠在一起。k=4、0.995。AIME 四个 seed。",
        "",
        "## 只交试答（不开后路强停）",
        "",
        "| 集 | 官方 PUMA | 只开 0.995 | 挡住高置信交错 | 捞低置信对 | 高置信停错 | 挡住救回 | 低置信捞回 |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for name, rec in trial:
        lines.append(
            f"| {name} | {fmt(rec['puma_acc'], rec['puma_tok'])} "
            f"| {fmt(rec['base_acc'], rec['base_tok'])} "
            f"| {fmt(rec['veto_acc'], rec['veto_tok'])}（{rg.fmt_pp(rec['d_veto_base'])}） "
            f"| {fmt(rec['fish_acc'], rec['fish_tok'])}（{rg.fmt_pp(rec['d_fish_base'])}） "
            f"| {rec['n_high_wrong']}/{rec['n_high']} "
            f"| {rec['n_veto_gain']} | {rec['n_fish_gain']} |"
        )
    lines += [
        "",
        "## 密探 + 截断再写终答",
        "",
        "挡住 = 早停后重写仍错、且写完全程是对的，就改成写完。捞 = 停点前交对金标的低置信试答。",
        "",
        "| 集 | 官方 PUMA | 密探+重写 | 挡住高置信交错 | 捞低置信对 | 早停仍错 | 挡住救回 | 低置信捞回 |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for name, rec in regen:
        lines.append(
            f"| {name} | {fmt(rec['puma_acc'], rec['puma_tok'])} "
            f"| {fmt(rec['host_acc'], rec['host_tok'])} "
            f"| {fmt(rec['veto_acc'], rec['veto_tok'])}（{rg.fmt_pp(rec['d_veto_host'])}） "
            f"| {fmt(rec['fish_acc'], rec['fish_tok'])}（{rg.fmt_pp(rec['d_fish_host'])}） "
            f"| {rec['n_early_wrong']}/{rec['n_early']} "
            f"| {rec['n_veto_gain']} | {rec['n_fish_gain']} |"
        )
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[7:]), flush=True)
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
