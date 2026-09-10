#!/usr/bin/env python3
"""密探k4 底：混合窗形状 / 武装后降门槛 / 频率，不是再确认一次。"""
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
import report_first_lock_room as room
import report_k4_hyps as h
import report_k4_second_lock as sl

TABLE = AE / "tables/k4_hyps3.md"
FREEZE = ("7B MATH", "7B GPQA", "8B MATH", "8B GPQA")
RUNTIME = (
    ("mix0", "第一扇混合"),
    ("mix_up", "混合且把握抬"),
    ("mix_end", "混合且最后最高"),
    ("mix_99", "混合且最后 ≥0.99"),
    ("mix_min90", "混合且四步 ≥0.90"),
    ("arm98", "武装后 τ=0.98"),
    ("arm99", "武装后 τ=0.99"),
    ("arm95", "武装后四步 ≥0.95"),
    ("freq50", "剩窗且该答已过半"),
    ("maj8", "近 8 步多数=该答"),
)
ORACLE = (
    ("mix_up_or", "先知混合抬"),
    ("arm98_or", "先知武装 0.98"),
    ("freq50_or", "先知过半"),
)


def windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        if int(row["stopped_len"]) < rg.MSS:
            continue
        window = rows[end + 1 - rg.K : end + 1]
        confs = rg.confs_of(window)
        if not confs or any(c != c for c in confs):
            continue
        out.append(
            {
                "end": end,
                "step": int(row["stopped_len"]),
                "ans": row.get("final_answer"),
                "c": confs[-1],
                "c0": confs[0],
                "cmin": min(confs),
                "cmax": max(confs),
                "kind": sl.kind_of(confs),
                "confs": confs,
            }
        )
    return out


def near_high(confs: list[float], tau: float) -> bool:
    if len(confs) < rg.K or any(c != c for c in confs):
        return False
    if confs[0] < tau:
        return False
    return all(c >= confs[0] - rg.EPS for c in confs[1:])


def frac_same(rows: list[dict[str, Any]], end: int, ans: Any) -> float:
    used = rows[: end + 1]
    if not used:
        return 0.0
    hit = sum(1 for x in used if rg.same(x.get("final_answer"), ans))
    return hit / len(used)


def maj8(rows: list[dict[str, Any]], end: int, ans: Any) -> bool:
    used = rows[max(0, end + 1 - 8) : end + 1]
    if len(used) < 8:
        return False
    hit = sum(1 for x in used if rg.same(x.get("final_answer"), ans))
    return hit >= 5


def pick(rows: list[dict[str, Any]], wins: list[dict[str, Any]], host_step: int) -> dict[str, Any]:
    out: dict[str, Any] = {k: None for k, _ in RUNTIME}
    left = [w for w in wins if w["kind"] in ("low", "mix") and w["step"] < host_step]
    mix = left[0] if left and left[0]["kind"] == "mix" else None
    if mix is not None:
        out["mix0"] = mix
        if mix["c"] >= mix["c0"]:
            out["mix_up"] = mix
        if mix["c"] >= mix["cmax"] - 1e-12:
            out["mix_end"] = mix
        if mix["c"] >= 0.99:
            out["mix_99"] = mix
        if mix["cmin"] >= 0.90:
            out["mix_min90"] = mix
    if left:
        a0 = left[0]["ans"]
        for w in wins:
            if w["step"] < host_step and rg.same(w["ans"], a0) and near_high(w["confs"], 0.98):
                out["arm98"] = w
                break
        for w in wins:
            if w["step"] < host_step and rg.same(w["ans"], a0) and near_high(w["confs"], 0.99):
                out["arm99"] = w
                break
        for w in left:
            if rg.same(w["ans"], a0) and w["cmin"] >= 0.95:
                out["arm95"] = w
                break
    for w in left:
        if frac_same(rows, w["end"], w["ans"]) >= 0.5:
            out["freq50"] = w
            break
    for w in left:
        if maj8(rows, w["end"], w["ans"]):
            out["maj8"] = w
            break
    return out


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    puma_path = dd.puma_stat_path(model, dataset, seed)
    trials_path = room.dense_trial_path(model, dataset, seed)
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
        host = regen.get(qi)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        if host:
            n_regen += 1
            host_ok = bool(host.get("compressed_correct"))
            host_tok = int(host.get("compressed_tokens") or 0) + int(
                host.get("tokens_trial_answers") or 0
            )
            host_step = int(host.get("stopped_len") or sim["step"])
        else:
            host_ok = bool(info.get("compressed_correct"))
            host_tok = int(info.get("compressed_tokens") or 0) + int(
                info.get("tokens_trial_answers") or 0
            )
            host_step = int(sim["step"])
        raw = pick(rows, windows(rows), host_step)
        fires = {
            name: h.pack_fire(
                trials,
                rows,
                raw[name],
                orig_tok=orig_tok,
                gt=gt,
                original=original,
                a_final=a_final,
                orig_ok=orig_ok,
                need_pos=False,
            )
            for name, _ in RUNTIME
        }
        for name, src in (("mix_up_or", "mix_up"), ("arm98_or", "arm98"), ("freq50_or", "freq50")):
            fires[name] = h.pack_fire(
                trials,
                rows,
                raw[src],
                orig_tok=orig_tok,
                gt=gt,
                original=original,
                a_final=a_final,
                orig_ok=orig_ok,
                need_pos=True,
            )
        questions.append(
            {
                "host_ok": host_ok,
                "host_tok": host_tok,
                "host_step": host_step,
                "armed": {name: raw[name] is not None for name, _ in RUNTIME},
                "fires": fires,
            }
        )
    if not questions:
        return None
    return {"questions": questions, "n_regen": n_regen}


def score(pack: dict[str, Any]) -> dict[str, Any]:
    qs = pack["questions"]
    n = len(qs)
    rec: dict[str, Any] = {
        "n": n,
        "n_regen": pack["n_regen"],
        "host_name": "密探k4" if pack["n_regen"] == n else "PUMA",
        "host_acc": sum(int(q["host_ok"]) for q in qs) / n,
        "host_tok": sum(q["host_tok"] for q in qs) / n,
    }
    for name, _ in RUNTIME:
        rec[f"n_{name}"] = sum(1 for q in qs if q["armed"][name])
        rec[name] = h.apply_door(qs, name)
    for name, _ in ORACLE:
        rec[name] = h.apply_door(qs, name)
    return rec


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = [p for p in (load_cell(model, dataset, s) for s in dd.AIME_SEEDS) if p]
        if not packs:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
        pack = room.merge(packs)
        name = f"{zh} {ds_zh}" if len(packs) == 4 else f"{zh} {ds_zh}（{len(packs)} seed）"
    else:
        pack = load_cell(model, dataset, 42)
        if pack is None:
            print(f"skip {zh} {ds_zh}", flush=True)
            return None
        name = f"{zh} {ds_zh}"
    rec = score(pack)
    rec["name"] = name
    bits = " ".join(
        f"{k}={rg.fmt_pp(100.0 * (rec[k]['acc'] - rec['host_acc']))}" for k, _ in RUNTIME
    )
    print(f"{name} n={rec['n']} {bits}", flush=True)
    return rec


def pick_frozen(recs: list[dict[str, Any]]) -> tuple[str, str] | None:
    by = {r["name"]: r for r in recs}
    if not all(x in by for x in FREEZE):
        return None
    best = None
    for name, zh in RUNTIME:
        cells = [by[x][name] for x in FREEZE]
        hosts = [by[x] for x in FREEZE]
        if any(c["acc"] + 1e-12 < h["host_acc"] for c, h in zip(cells, hosts)):
            continue
        saved = sum(c["tok"] - h["host_tok"] for c, h in zip(cells, hosts))
        if best is None or saved < best[0]:
            best = (saved, name, zh)
    return None if best is None else (best[1], best[2])


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in (("7B", "r1_7b"), ("8B", "nemotron_8b")):
        for ds_zh, dataset in room.DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    frozen = pick_frozen(recs)
    lines = [
        "# 密探k4 底：混合形状 / 武装降门槛 / 频率",
        "",
        "不再做「再确认一次同答」。三类没试过的：",
        "",
        "- **混合窗形状**：上一轮第一扇混合只差 8B MATH −0.6。这里拆成把握抬 / 最后最高 / 最后 ≥0.99 / 四步都 ≥0.90。",
        "- **武装后降门槛**：先出现剩窗答案 A，之后 A 用更低的高把握锁（0.98 / 0.99 / 四步 ≥0.95）才停。不交低把握试答。",
        "- **频率**：剩窗时，到此为止该答案已占一半探点，或近 8 步至少 5 次是它。不是连续 8 步。",
        "",
        "对照密探k4。开火交试答。冻法仍是 7B/8B 的 MATH+GPQA。",
        "",
    ]
    if frozen:
        lines += [f"**冻法：** `{frozen[1]}`。", ""]
    else:
        lines += ["**冻不成。**", ""]
    lines += [
        "## 1. 能开火的题数",
        "",
        "| 集 | 混合 | 抬 | 最后最高 | 最后≥0.99 | 四步≥0.90 | 武装0.98 | 武装0.99 | 四步≥0.95 | 过半 | 近8多数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n_mix0']} | {rec['n_mix_up']} | {rec['n_mix_end']} "
            f"| {rec['n_mix_99']} | {rec['n_mix_min90']} | {rec['n_arm98']} | {rec['n_arm99']} "
            f"| {rec['n_arm95']} | {rec['n_freq50']} | {rec['n_maj8']} |"
        )
    lines += [
        "",
        "## 2. 混合窗形状",
        "",
        "| 集 | 宿主 | 全混合 | 把握抬 | 最后最高 | 最后≥0.99 | 四步≥0.90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rg.fmt_pct(rec['host_acc'])} / {rec['host_tok']:.0f} "
            f"| {h.door_cell(rec, 'mix0')} | {h.door_cell(rec, 'mix_up')} "
            f"| {h.door_cell(rec, 'mix_end')} | {h.door_cell(rec, 'mix_99')} "
            f"| {h.door_cell(rec, 'mix_min90')} |"
        )
    lines += [
        "",
        "## 3. 武装后降门槛 / 频率",
        "",
        "| 集 | 武装 τ=0.98 | 武装 τ=0.99 | 武装四步≥0.95 | 该答已过半 | 近 8 步多数 |",
        "|---|---|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {h.door_cell(rec, 'arm98')} | {h.door_cell(rec, 'arm99')} "
            f"| {h.door_cell(rec, 'arm95')} | {h.door_cell(rec, 'freq50')} "
            f"| {h.door_cell(rec, 'maj8')} |"
        )
    lines += [
        "",
        "## 4. 先知",
        "",
        "| 集 | 先知混合抬 | 先知武装 0.98 | 先知过半 |",
        "|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {h.door_cell(rec, 'mix_up_or')} "
            f"| {h.door_cell(rec, 'arm98_or')} | {h.door_cell(rec, 'freq50_or')} |"
        )
    lines += ["", "## 读法", "", "混合形状是为了修 8B MATH 那 3 题假混合。武装降门槛不交低把握试答。", ""]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)
    print(f"冻：{frozen[1]}" if frozen else "冻不成", flush=True)


if __name__ == "__main__":
    main()
