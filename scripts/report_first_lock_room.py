#!/usr/bin/env python3
"""第一次可停（不看把握）分三档：相对宿主的提升空间。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_cand_collapse as cc
import report_dense_k4_lowconf_ceiling as low
import report_k4_angles as ang

TABLE = AE / "tables/first_lock_room.md"
MODELS = (
    ("7B", "r1_7b"),
    ("8B", "nemotron_8b"),
    ("14B", "r1_14b"),
    ("32B", "r1_32b"),
    ("30B", "qwen3_30b_a3b"),
    ("Qwen3-4B", "qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b"),
)
DS = (
    ("MATH", "math-500"),
    ("奥赛", "olympiadbench"),
    ("GPQA", "gpqa-diamond"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
)
KINDS = ("high", "mix", "low")
KIND_ZH = {"high": "高把握", "mix": "混合", "low": "低把握"}


def dense_trial_path(model: str, dataset: str, seed: int) -> Path:
    seeded = AE / f"results/dense_G_{model}/{dataset}/seed_{seed}/dense_puma/trial_answers.json"
    flat = AE / f"results/dense_G_{model}/{dataset}/dense_puma/trial_answers.json"
    if seeded.is_file():
        return seeded
    if flat.is_file():
        return flat
    if dataset in ("aime24", "aime25", "amc23", "gsm8k") or seed != 42:
        return seeded
    return flat


def window_kind(confs: list[float]) -> str:
    if rg.is_high(confs):
        return "high"
    if rg.is_low(confs):
        return "low"
    return "mix"


def first_ok(
    rows: list[dict[str, Any]], gt: Any, original: Any, a_final: Any
) -> dict[str, Any] | None:
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        if int(row["stopped_len"]) < rg.MSS:
            continue
        ans = row.get("final_answer")
        if not cc.stoppable(ans, gt, original, a_final):
            continue
        window = rows[end + 1 - rg.K : end + 1]
        confs = rg.confs_of(window)
        return {
            "end": end,
            "step": int(row["stopped_len"]),
            "ans": ans,
            "kind": window_kind(confs),
        }
    return None


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = dense_trial_path(model, dataset, seed)
    if not puma_path.is_file() or not trials_path.is_file():
        return None
    official = {int(r["question_idx"]): r for r in dd.load_json(puma_path)}
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)}
        if regen_path.is_file()
        else {}
    )
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    questions = []
    n_regen = 0
    for qi, info in sorted(official.items()):
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        gt = info.get("ground_truth")
        original = info.get("original_answer")
        a_final = (gmap.get(qi) or {}).get("A_final") or original
        orig_ok = bool(info.get("original_correct"))
        orig_tok = int(info.get("original_tokens") or 0)
        puma_ok = bool(info.get("compressed_correct"))
        puma_tok = int(info.get("compressed_tokens") or 0) + int(
            info.get("tokens_trial_answers") or 0
        )
        host = regen.get(qi)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        if host:
            n_regen += 1
            host_ok = bool(host.get("compressed_correct"))
            host_tok = int(host.get("compressed_tokens") or 0) + int(
                host.get("tokens_trial_answers") or 0
            )
            host_step = int(host.get("stopped_len") or sim["step"])
            host_name = "密探k4"
        else:
            host_ok = puma_ok
            host_tok = puma_tok
            host_step = int(sim["step"])
            host_name = "PUMA"
        lock = first_ok(rows, gt, original, a_final)
        fire = None
        if lock is not None:
            packed = rg.pack(
                trials, rows, lock["end"], "rescue", original_tokens=orig_tok
            )
            fire = {
                **lock,
                "ok": low.credit(packed["answer"], gt, original, orig_ok),
                "tok": packed["tokens"],
            }
        questions.append(
            {
                "host_ok": host_ok,
                "host_tok": host_tok,
                "host_step": host_step,
                "host_name": host_name,
                "puma_ok": puma_ok,
                "puma_tok": puma_tok,
                "k4_tok": sim["tokens"],
                "fire": fire,
            }
        )
    if not questions:
        return None
    return {"questions": questions, "n_regen": n_regen}


def merge(packs: list[dict[str, Any]]) -> dict[str, Any]:
    qs = []
    n_regen = 0
    for pack in packs:
        qs.extend(pack["questions"])
        n_regen += pack["n_regen"]
    return {"questions": qs, "n_regen": n_regen}


def apply_door(qs: list[dict[str, Any]], want: str | None) -> dict[str, Any]:
    acc = tok = fire = gain = hurt = earlier = later = 0
    lags: list[float] = []
    for q in qs:
        hit = q["fire"]
        use = hit is not None and (want is None or hit["kind"] == want)
        if use:
            ok, t = hit["ok"], hit["tok"]
            fire += 1
            gain += int(ok and not q["host_ok"])
            hurt += int((not ok) and q["host_ok"])
            if hit["step"] < q["host_step"]:
                earlier += 1
                lags.append(q["host_step"] - hit["step"])
            elif hit["step"] > q["host_step"]:
                later += 1
        else:
            ok, t = q["host_ok"], q["host_tok"]
        acc += int(ok)
        tok += t
    n = len(qs)
    return {
        "acc": acc / n,
        "tok": tok / n,
        "fire": fire,
        "gain": gain,
        "hurt": hurt,
        "earlier": earlier,
        "later": later,
        "lag_p50": rg.p50(lags) if lags else float("nan"),
    }


def score(pack: dict[str, Any]) -> dict[str, Any]:
    qs = pack["questions"]
    n = len(qs)
    rec: dict[str, Any] = {
        "n": n,
        "n_regen": pack["n_regen"],
        "host_name": "密探k4" if pack["n_regen"] == n else "PUMA",
        "host_acc": sum(q["host_ok"] for q in qs) / n,
        "host_tok": sum(q["host_tok"] for q in qs) / n,
        "puma_acc": sum(q["puma_ok"] for q in qs) / n,
        "puma_tok": sum(q["puma_tok"] for q in qs) / n,
        "k4_tok": sum(q["k4_tok"] for q in qs) / n,
        "n_hit": sum(1 for q in qs if q["fire"]),
        "n_none": sum(1 for q in qs if not q["fire"]),
    }
    for kind in KINDS:
        rec[f"n_{kind}"] = sum(
            1 for q in qs if q["fire"] and q["fire"]["kind"] == kind
        )
    rec["all"] = apply_door(qs, None)
    for kind in KINDS:
        rec[kind] = apply_door(qs, kind)
    return rec


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = []
        seeds = []
        for seed in dd.AIME_SEEDS:
            pack = load_cell(model, dataset, seed)
            if pack:
                packs.append(pack)
                seeds.append(seed)
        if not packs:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
        pack = merge(packs)
        name = f"{zh} {ds_zh}" if len(seeds) == 4 else f"{zh} {ds_zh}（{len(seeds)} seed）"
    else:
        pack = load_cell(model, dataset, 42)
        if pack is None:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
        name = f"{zh} {ds_zh}"
    rec = score(pack)
    rec["name"] = name
    print(
        f"{name} n={rec['n']} host={rec['host_name']} "
        f"high/mix/low={rec['n_high']}/{rec['n_mix']}/{rec['n_low']}",
        flush=True,
    )
    return rec


def door_cell(rec: dict[str, Any], key: str) -> str:
    door = rec[key]
    d_acc = 100.0 * (door["acc"] - rec["host_acc"])
    d_tok = door["tok"] - rec["host_tok"]
    lag = (
        f"，早中位 {door['lag_p50']:.0f} 步"
        if door["earlier"] and door["lag_p50"] == door["lag_p50"]
        else ""
    )
    late = f"，晚于宿主 {door['later']}" if door["later"] else ""
    return (
        f"{rg.fmt_pct(door['acc'])} / {door['tok']:.0f}"
        f"（{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)}；"
        f"开火 {door['fire']}，救回 {door['gain']} / 伤 {door['hurt']}"
        f"{lag}{late}）"
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    lines = [
        "# 第一次可停分三档：提升空间",
        "",
        "开火不看把握。第一次连续 4 步同一试答，且已经等于金标或写完终答（`A_final` / 官方终答），就交这步试答，不重写。",
        "步数 < 10 仍不许停。把握只用来事后分桶，不决定能不能停。",
        "",
        "- **高把握**：窗里第一次 ≥ 0.995，后面 ≥ 第一次 − 0.03（密探连答停会跳的那种）。",
        "- **低把握**：四步都 < 0.995。",
        "- **混合**：同答已经齐、也对上了，但既不是全低，也凑不齐高把握锁。",
        "",
        "「只交某一档」= 第一次可停恰好是这档才开火，其余题留宿主。",
        "宿主：这格若已有密探k4 重写，就留密探k4；没有重写就留官方 PUMA。",
        "Acc 只对金标。交跟写完同一串时，对错跟官方写完走。AIME 默认四个 seed；缺 seed 的格子会标明。",
        "只认密探轨迹（`dense_G_*`），官方 PUMA 稀疏探点不当第一次可停。",
        "",
        "## 1. 第一次可停先到哪一档",
        "",
        "| 集 | 题数 | 宿主 | 有可停 | 高把握先到 | 混合先到 | 低把握先到 | 从没可停 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n']} | {rec['host_name']} "
            f"| {rec['n_hit']}（{ang.pct(rec['n_hit'], rec['n'])}） "
            f"| {rec['n_high']} | {rec['n_mix']} | {rec['n_low']} | {rec['n_none']} |"
        )
    lines += [
        "",
        "## 2. 三种都交：第一次可停就停",
        "",
        "这是总上限：能锁上的题都交试答，锁不上的留宿主。括号相对宿主。",
        "",
        "| 集 | 宿主 | 三种都交 |",
        "|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rg.fmt_pct(rec['host_acc'])} / {rec['host_tok']:.0f} "
            f"| {door_cell(rec, 'all')} |"
        )
    lines += [
        "",
        "## 3. 只交其中一档",
        "",
        "看单独做高把握 / 混合 / 低把握门，天花板有多高。括号相对宿主。",
        "",
        "| 集 | 只交高把握先到 | 只交混合先到 | 只交低把握先到 |",
        "|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {door_cell(rec, 'high')} "
            f"| {door_cell(rec, 'mix')} | {door_cell(rec, 'low')} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "救回 / 伤是相对宿主的对错变化，不是相对 PUMA。",
        "早中位 = 开火题比宿主早几步（只统计比宿主早的那些）。",
        "高把握先到的题，密探k4 多半已经能停，空间通常只剩几步。",
        "混合 / 低把握先到才是「值不值得另做一扇门」的地方。",
        "14B / 32B / 30B / Qwen3 没有密探k4 重写，宿主是 PUMA，Acc 空间相对 PUMA。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
