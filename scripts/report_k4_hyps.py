#!/usr/bin/env python3
"""密探k4 底：几条还没做过的运行时假说。"""
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
import report_k4_second_lock as sl

TABLE = AE / "tables/k4_hyps.md"
FREEZE = ("7B MATH", "7B GPQA", "8B MATH", "8B GPQA")
RUNTIME = (
    ("mix0", "第一扇混合"),
    ("back", "换答后再锁回"),
    ("block2", "非重叠第二块"),
    ("third", "第三扇同答"),
    ("k8", "连续 8 步同答"),
    ("late30", "剩窗步数 ≥30"),
    ("late40", "剩窗步数 ≥40"),
    ("sec90", "第二扇且把握 ≥0.90"),
)
ORACLE = (
    ("mix0_or", "先知混合"),
    ("back_or", "先知锁回"),
    ("k8_or", "先知 8 步"),
    ("late40_or", "先知 ≥40"),
)


def leftover(w: dict[str, Any]) -> bool:
    return w["kind"] in ("low", "mix")


def saw_other(rows: list[dict[str, Any]], lo: int, hi: int, ans: Any) -> bool:
    for row in rows:
        step = int(row["stopped_len"])
        if step <= lo or step >= hi:
            continue
        if not rg.same(row.get("final_answer"), ans):
            return True
    return False


def first_k8(rows: list[dict[str, Any]], host_step: int) -> dict[str, Any] | None:
    for end in range(7, len(rows)):
        window = rows[end - 7 : end + 1]
        steps = [int(x["stopped_len"]) for x in window]
        if steps != list(range(steps[0], steps[0] + 8)):
            continue
        step = steps[-1]
        if step < rg.MSS or step >= host_step:
            continue
        ans = window[0].get("final_answer")
        if not all(rg.same(ans, x.get("final_answer")) for x in window):
            continue
        last4 = window[-4:]
        confs = rg.confs_of(last4)
        if not confs or any(c != c for c in confs) or rg.is_high(confs):
            continue
        return {
            "end": end,
            "step": step,
            "ans": ans,
            "c": confs[-1],
            "kind": sl.kind_of(confs),
        }
    return None


def pick_hyps(
    rows: list[dict[str, Any]],
    wins: list[dict[str, Any]],
    host_step: int,
) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {k: None for k, _ in RUNTIME + ORACLE}
    left = [w for w in wins if leftover(w) and w["step"] < host_step]
    if left:
        first = left[0]
        if first["kind"] == "mix":
            out["mix0"] = first
        same = [w for w in left if rg.same(w["ans"], first["ans"])]
        for w in same[1:]:
            if w["step"] >= first["step"] + 4 and out["block2"] is None:
                out["block2"] = w
            if w["c"] >= 0.90 and out["sec90"] is None:
                out["sec90"] = w
            if saw_other(rows, first["step"], w["step"], first["ans"]) and out["back"] is None:
                out["back"] = w
        if len(same) >= 3:
            out["third"] = same[2]
        for w in left:
            if w["step"] >= 30 and out["late30"] is None:
                out["late30"] = w
            if w["step"] >= 40 and out["late40"] is None:
                out["late40"] = w
    out["k8"] = first_k8(rows, host_step)
    return out


def pack_fire(
    trials: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    hit: dict[str, Any] | None,
    *,
    orig_tok: int,
    gt: Any,
    original: Any,
    a_final: Any,
    orig_ok: bool,
    need_pos: bool,
) -> dict[str, Any] | None:
    if hit is None:
        return None
    pos = cc.stoppable(hit["ans"], gt, original, a_final)
    if need_pos and not pos:
        return None
    packed = rg.pack(trials, rows, hit["end"], "rescue", original_tokens=orig_tok)
    return {
        "step": hit["step"],
        "kind": hit["kind"],
        "pos": pos,
        "ok": low.credit(packed["answer"], gt, original, orig_ok),
        "tok": packed["tokens"],
    }


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
            host_name = "密探k4"
        else:
            host_ok = bool(info.get("compressed_correct"))
            host_tok = int(info.get("compressed_tokens") or 0) + int(
                info.get("tokens_trial_answers") or 0
            )
            host_step = int(sim["step"])
            host_name = "PUMA"
        wins = sl.same_windows(rows)
        raw = pick_hyps(rows, wins, host_step)
        fires = {}
        for name, _ in RUNTIME:
            fires[name] = pack_fire(
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
        alias = {
            "mix0_or": "mix0",
            "back_or": "back",
            "k8_or": "k8",
            "late40_or": "late40",
        }
        for name, src in alias.items():
            fires[name] = pack_fire(
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
                "host_name": host_name,
                "armed": {name: raw[name] is not None for name, _ in RUNTIME},
                "fires": fires,
            }
        )
    if not questions:
        return None
    return {"questions": questions, "n_regen": n_regen}


def apply_door(qs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    acc = tok = fire = gain = hurt = earlier = 0
    lags: list[float] = []
    for q in qs:
        hit = q["fires"][name]
        if hit is not None:
            ok, t = hit["ok"], hit["tok"]
            fire += 1
            gain += int(ok and not q["host_ok"])
            hurt += int((not ok) and q["host_ok"])
            if hit["step"] < q["host_step"]:
                earlier += 1
                lags.append(q["host_step"] - hit["step"])
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
        "lag_p50": rg.p50(lags) if lags else float("nan"),
    }


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
        rec[name] = apply_door(qs, name)
    for name, _ in ORACLE:
        rec[name] = apply_door(qs, name)
    return rec


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = [load_cell(model, dataset, seed) for seed in dd.AIME_SEEDS]
        packs = [p for p in packs if p]
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
        f"{k}={rg.fmt_pp(100.0 * (rec[k]['acc'] - rec['host_acc']))}"
        for k, _ in RUNTIME
    )
    print(f"{name} n={rec['n']} {bits}", flush=True)
    return rec


def door_cell(rec: dict[str, Any], key: str) -> str:
    door = rec[key]
    d_acc = 100.0 * (door["acc"] - rec["host_acc"])
    d_tok = door["tok"] - rec["host_tok"]
    lag = (
        f"，早中位 {door['lag_p50']:.0f}"
        if door["earlier"] and door["lag_p50"] == door["lag_p50"]
        else ""
    )
    return (
        f"{rg.fmt_pct(door['acc'])} / {door['tok']:.0f}"
        f"（{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)}；"
        f"开火 {door['fire']}，救回 {door['gain']} / 伤 {door['hurt']}{lag}）"
    )


def pick_frozen(recs: list[dict[str, Any]]) -> tuple[str, str] | None:
    by = {r["name"]: r for r in recs}
    if not all(name in by for name in FREEZE):
        return None
    best: tuple[float, str] | None = None
    for name, zh in RUNTIME:
        cells = [by[x][name] for x in FREEZE]
        hosts = [by[x] for x in FREEZE]
        if any(c["acc"] + 1e-12 < h["host_acc"] for c, h in zip(cells, hosts)):
            continue
        saved = sum(c["tok"] - h["host_tok"] for c, h in zip(cells, hosts))
        if best is None or saved < best[0]:
            best = (saved, name)
    if best is None:
        return None
    return best[1], dict(RUNTIME)[best[1]]


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in room.MODELS:
        for ds_zh, dataset in room.DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    frozen = pick_frozen(recs)
    lines = [
        "# 密探k4 底：别的假说",
        "",
        "对照仍是密探k4（无重写则 PUMA）。开火交试答，不重写。步数 < 10 不算。",
        "",
        "- **第一扇混合**：第一次同答窗既不是高把握锁、也不是四步都 < 0.995。",
        "- **换答后再锁回**：先有一扇剩窗，中间换过别的试答，再锁回同一答案。",
        "- **非重叠第二块**：同一答案再锁，且至少隔 4 步（不是紧挨着的下一扇）。",
        "- **第三扇同答**：同一答案第三次剩窗。",
        "- **连续 8 步同答**：k=8，最后 4 步仍不是高把握锁。",
        "- **剩窗步数 ≥30 / 40**：跳过早到的剩窗，取第一扇够晚的。",
        "- **第二扇且把握 ≥0.90**：第二扇同答，最后一步把握够高。",
        "",
    ]
    if frozen:
        lines += [f"**冻法：** 7B/8B MATH+GPQA Acc 都不低于宿主，最省 token 的是 `{frozen[1]}`。", ""]
    else:
        lines += ["**冻不成。** 7B/8B MATH+GPQA 没有一条运行时门四格 Acc 都不掉。", ""]
    lines += [
        "## 1. 能开火的题数",
        "",
        "| 集 | 题数 | 宿主 | 混合 | 锁回 | 第二块 | 第三扇 | 8 步 | ≥30 | ≥40 | 二扇≥0.90 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n']} | {rec['host_name']} "
            f"| {rec['n_mix0']} | {rec['n_back']} | {rec['n_block2']} "
            f"| {rec['n_third']} | {rec['n_k8']} | {rec['n_late30']} "
            f"| {rec['n_late40']} | {rec['n_sec90']} |"
        )
    lines += [
        "",
        "## 2. 运行时（不问对错）",
        "",
        "| 集 | 宿主 | 第一扇混合 | 换答后再锁回 | 非重叠第二块 | 第三扇同答 |",
        "|---|---|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rg.fmt_pct(rec['host_acc'])} / {rec['host_tok']:.0f} "
            f"| {door_cell(rec, 'mix0')} "
            f"| {door_cell(rec, 'back')} "
            f"| {door_cell(rec, 'block2')} "
            f"| {door_cell(rec, 'third')} |"
        )
    lines += [
        "",
        "| 集 | 连续 8 步 | 剩窗 ≥30 | 剩窗 ≥40 | 第二扇把握 ≥0.90 |",
        "|---|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {door_cell(rec, 'k8')} "
            f"| {door_cell(rec, 'late30')} "
            f"| {door_cell(rec, 'late40')} "
            f"| {door_cell(rec, 'sec90')} |"
        )
    lines += [
        "",
        "## 3. 先知：同样的门，只停对上的",
        "",
        "| 集 | 先知混合 | 先知锁回 | 先知 8 步 | 先知 ≥40 |",
        "|---|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {door_cell(rec, 'mix0_or')} "
            f"| {door_cell(rec, 'back_or')} "
            f"| {door_cell(rec, 'k8_or')} "
            f"| {door_cell(rec, 'late40_or')} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "冻法只看 7B/8B 的 MATH 和 GPQA。先知格不是门，只看这条假说天花板还在不在。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)
    print(f"冻：{frozen[1]}" if frozen else "冻不成", flush=True)


if __name__ == "__main__":
    main()
