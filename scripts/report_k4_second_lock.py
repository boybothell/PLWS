#!/usr/bin/env python3
"""密探k4 底：第一扇同答只记账，第二扇同答才考虑停。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_cand_collapse as cc
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room

TABLE = AE / "tables/k4_second_lock.md"
FREEZE = ("7B MATH", "7B GPQA", "8B MATH", "8B GPQA")
VARIANTS = (
    ("second", "第二扇同答"),
    ("dc_ge", "第二扇且把握不降"),
    ("dc_gt", "第二扇且把握抬"),
    ("dc_01", "第二扇且把握 +0.01"),
    ("dc_02", "第二扇且把握 +0.02"),
    ("dc_05", "第二扇且把握 +0.05"),
    ("dt_8", "第二扇且隔 ≥8 步"),
    ("low2", "第二扇仍全低把握"),
    ("oracle", "先知：第二扇同答可停才停"),
)


def kind_of(confs: list[float]) -> str:
    if rg.is_high(confs):
        return "high"
    if rg.is_low(confs):
        return "low"
    return "mix"


def same_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "kind": kind_of(confs),
            }
        )
    return out


def pred_for(name: str) -> Callable[[dict[str, Any], dict[str, Any]], bool]:
    if name == "second":
        return lambda _s0, _w: True
    if name == "dc_ge":
        return lambda s0, w: w["c"] >= s0["c"]
    if name == "dc_gt":
        return lambda s0, w: w["c"] > s0["c"]
    if name == "dc_01":
        return lambda s0, w: w["c"] >= s0["c"] + 0.01
    if name == "dc_02":
        return lambda s0, w: w["c"] >= s0["c"] + 0.02
    if name == "dc_05":
        return lambda s0, w: w["c"] >= s0["c"] + 0.05
    if name == "dt_8":
        return lambda s0, w: w["step"] >= s0["step"] + 8
    if name == "low2":
        return lambda _s0, w: w["kind"] == "low"
    raise KeyError(name)


def pick(
    wins: list[dict[str, Any]],
    host_step: int,
    name: str,
    *,
    gt: Any,
    original: Any,
    a_final: Any,
) -> dict[str, Any] | None:
    if not wins:
        return None
    s0 = wins[0]
    if s0["kind"] == "high":
        return None
    for w in wins[1:]:
        if w["step"] >= host_step:
            break
        if w["kind"] == "high":
            continue
        if not rg.same(w["ans"], s0["ans"]):
            continue
        if name == "oracle":
            if cc.stoppable(w["ans"], gt, original, a_final):
                return {**w, "s0": s0}
            continue
        if pred_for(name)(s0, w):
            return {**w, "s0": s0}
    return None


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
        wins = same_windows(rows)
        s0 = wins[0] if wins else None
        later_same = 0
        later_left = 0
        if s0 and s0["kind"] != "high":
            for w in wins[1:]:
                if w["step"] >= host_step:
                    break
                if not rg.same(w["ans"], s0["ans"]):
                    continue
                later_same += 1
                if w["kind"] != "high":
                    later_left += 1
        fires = {}
        for name, _ in VARIANTS:
            hit = pick(wins, host_step, name, gt=gt, original=original, a_final=a_final)
            if hit is None:
                fires[name] = None
                continue
            packed = rg.pack(trials, rows, hit["end"], "rescue", original_tokens=orig_tok)
            fires[name] = {
                "step": hit["step"],
                "kind": hit["kind"],
                "s0_kind": hit["s0"]["kind"],
                "dc": hit["c"] - hit["s0"]["c"],
                "dt": hit["step"] - hit["s0"]["step"],
                "pos": cc.stoppable(hit["ans"], gt, original, a_final),
                "ok": low.credit(packed["answer"], gt, original, orig_ok),
                "tok": packed["tokens"],
            }
        questions.append(
            {
                "host_ok": host_ok,
                "host_tok": host_tok,
                "host_step": host_step,
                "host_name": host_name,
                "s0_kind": None if s0 is None else s0["kind"],
                "later_same": later_same,
                "later_left": later_left,
                "fires": fires,
            }
        )
    if not questions:
        return None
    return {"questions": questions, "n_regen": n_regen}


def apply_door(qs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    acc = tok = fire = gain = hurt = earlier = 0
    lags: list[float] = []
    dcs: list[float] = []
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
            if hit["dc"] == hit["dc"]:
                dcs.append(hit["dc"])
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
        "dc_p50": rg.p50(dcs) if dcs else float("nan"),
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
        "n_arm": sum(1 for q in qs if q["s0_kind"] in ("low", "mix")),
        "n_later": sum(1 for q in qs if q["later_same"]),
        "n_left": sum(1 for q in qs if q["later_left"]),
        "n_low0": sum(1 for q in qs if q["s0_kind"] == "low"),
        "n_mix0": sum(1 for q in qs if q["s0_kind"] == "mix"),
    }
    for name, _ in VARIANTS:
        rec[name] = apply_door(qs, name)
    return rec


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = []
        for seed in dd.AIME_SEEDS:
            pack = load_cell(model, dataset, seed)
            if pack:
                packs.append(pack)
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
    door = rec["second"]
    print(
        f"{name} n={rec['n']} host={rec['host_name']} "
        f"arm={rec['n_arm']} later={rec['n_later']} "
        f"second={rg.fmt_pp(100.0 * (door['acc'] - rec['host_acc']))}/"
        f"{rg.fmt_tok(door['tok'] - rec['host_tok'])} fire={door['fire']}",
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
    for name, zh in VARIANTS:
        if name == "oracle":
            continue
        cells = [by[x][name] for x in FREEZE]
        hosts = [by[x] for x in FREEZE]
        if any(c["acc"] + 1e-12 < h["host_acc"] for c, h in zip(cells, hosts)):
            continue
        saved = sum(c["tok"] - h["host_tok"] for c, h in zip(cells, hosts))
        if best is None or saved < best[0]:
            best = (saved, name)
    if best is None:
        return None
    zh = dict(VARIANTS)[best[1]]
    return best[1], zh


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
        "# 密探k4 + 第二扇同答",
        "",
        "第一扇连续 4 步同答只记账，不停。高把握第一扇交给密探k4。",
        "之后再次连续 4 步、还是同一个试答、窗还不是高把握锁、且早于宿主停点，才交这步试答，不重写。",
        "没开火留宿主：有密探k4 重写就留它，否则留官方 PUMA。Acc 只对金标。",
        "先知格：同样只看第二扇同答，但对上金标或写完终答才停。",
        "",
    ]
    if frozen:
        lines += [
            f"**冻法：** 7B/8B 的 MATH+GPQA 上 Acc 都不低于宿主，再选最省 token 的："
            f"`{frozen[1]}`。",
            "",
        ]
    else:
        lines += [
            "**冻不成。** 7B/8B 的 MATH+GPQA 上，没有一条运行时门能四格 Acc 都不低于宿主。",
            "",
        ]
    lines += [
        "## 1. 有多少题能走到第二扇",
        "",
        "武装 = 第一扇不是高把握。再锁 = 宿主停点前还有同答窗。剩窗 = 再锁时仍不是高把握。",
        "",
        "| 集 | 题数 | 宿主 | 武装 | 再锁 | 剩窗 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n']} | {rec['host_name']} "
            f"| {rec['n_arm']} | {rec['n_later']} | {rec['n_left']} |"
        )
    lines += [
        "",
        "## 2. 运行时门（不问对错）",
        "",
        "| 集 | 宿主 | 第二扇同答 | 把握不降 | 把握抬 |",
        "|---|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rg.fmt_pct(rec['host_acc'])} / {rec['host_tok']:.0f} "
            f"| {door_cell(rec, 'second')} "
            f"| {door_cell(rec, 'dc_ge')} "
            f"| {door_cell(rec, 'dc_gt')} |"
        )
    lines += [
        "",
        "## 3. 更严的过滤 / 先知上限",
        "",
        "| 集 | +0.01 | +0.02 | 隔 ≥8 步 | 第二扇仍低把握 | 先知可停 |",
        "|---|---|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {door_cell(rec, 'dc_01')} "
            f"| {door_cell(rec, 'dc_02')} "
            f"| {door_cell(rec, 'dt_8')} "
            f"| {door_cell(rec, 'low2')} "
            f"| {door_cell(rec, 'oracle')} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "对照是密探k4（没有重写则是 PUMA），不是论文 ASAG。",
        "第二扇同答是运行时门：对错都停。先知格才只停对上的。",
        "冻法只看 7B/8B 的 MATH 和 GPQA。奥赛 / AIME / 更大模型是迁移。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)
    if frozen:
        print(f"冻：{frozen[1]}", flush=True)
    else:
        print("冻不成", flush=True)


if __name__ == "__main__":
    main()
