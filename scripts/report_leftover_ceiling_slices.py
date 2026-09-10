#!/usr/bin/env python3
"""跨模型跨集：剩窗各切片上限，看哪一块值得做。"""
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
import report_k4_hyps as hy
import report_k4_second_lock as sl

TABLE = AE / "tables/leftover_ceiling_slices.md"
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
SLICES = (
    ("commit", "剩窗且等于写完终答"),
    ("ok", "剩窗且对金标或写完"),
    ("mix_ok", "混合剩窗且对"),
    ("low_ok", "低把握剩窗且对"),
    ("never_ok", "再也没有高把握、且对"),
    ("never_commit", "再也没有高把握、且等于写完"),
)


def fmt_pair(acc: float, tok: float) -> str:
    return f"{100.0 * acc:.1f}% / {tok:.0f}"


def load_cell(model: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not trials_path.is_file():
        return []
    puma_path = dd.puma_stat_path(model, dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    regen_path = dd.regen_stat_path(model, dataset, seed)
    regen = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)} if regen_path.is_file() else {}
    )
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    if official:
        qis = sorted(official)
    elif gmap:
        qis = sorted(set(gmap) & set(by))
    else:
        qis = sorted(by)
    out = []
    for qi in qis:
        trials = by.get(qi)
        if not trials:
            continue
        rows = rg.usable_rows(trials)
        if not rows:
            continue
        info = official.get(qi) or {}
        g = gmap.get(qi) or {}
        last = max(trials, key=lambda x: int(x["stopped_len"]))
        gt = info.get("ground_truth") or g.get("ground_truth")
        original = info.get("original_answer") or g.get("A_final") or last.get("final_answer")
        a_final = g.get("A_final") or original
        orig_ok = bool(info.get("original_correct")) if "original_correct" in info else bool(
            low.credit(original, gt, original, True)
        )
        orig_tok = int(info.get("original_tokens") or last.get("count_reasoning_tokens") or 0)
        sim = dd.simulate(trials, original_tokens=orig_tok)
        host = regen.get(qi)
        if host:
            host_ok = bool(host.get("compressed_correct"))
            host_tok = int(host.get("compressed_tokens") or 0) + int(
                host.get("tokens_trial_answers") or 0
            )
            host_name = "密探k4"
        elif official:
            host_ok = bool(info.get("compressed_correct"))
            host_tok = int(info.get("compressed_tokens") or 0) + int(
                info.get("tokens_trial_answers") or 0
            )
            host_name = "PUMA"
        else:
            host_ok = orig_ok
            host_tok = float(orig_tok or sim["tokens"])
            host_name = "写完"
        wins = sl.same_windows(rows)
        first = wins[0] if wins else None
        left = next((w for w in wins if hy.leftover(w)), None)
        high = next((w for w in wins if w["kind"] == "high"), None)
        high_after = (
            next((w for w in wins if w["kind"] == "high" and w["step"] > left["step"]), None)
            if left
            else None
        )
        if left is not None:
            packed = rg.pack(trials, rows, left["end"], "rescue", original_tokens=orig_tok)
            left_ok = bool(low.credit(left["ans"], gt, original, orig_ok))
            committed = bool(rg.same(left["ans"], a_final))
            left_tok = packed["tokens"]
        else:
            left_ok = False
            committed = False
            left_tok = host_tok
        out.append(
            {
                "first_kind": first["kind"] if first else "none",
                "has_left": left is not None,
                "left_kind": left["kind"] if left else None,
                "left_step": int(left["step"]) if left else None,
                "left_ok": left_ok,
                "committed": committed,
                "high_any": high is not None,
                "high_after": high_after is not None,
                "same_as_high": bool(
                    high_after is not None and rg.same(left["ans"], high_after["ans"])
                )
                if left is not None
                else False,
                "will_change": bool(
                    high_after is not None and not rg.same(left["ans"], high_after["ans"])
                )
                if left is not None
                else False,
                "never_high": bool(left is not None and high_after is None),
                "host_ok": host_ok,
                "host_tok": host_tok,
                "host_name": host_name,
                "left_tok": left_tok,
                "n_steps": int(last.get("stopped_len") or 0),
            }
        )
    return out


def apply(qs: list[dict[str, Any]], pred) -> dict[str, float]:
    acc = tok = fire = gain = hurt = 0
    lags: list[float] = []
    for q in qs:
        if pred(q):
            ok, t = q["left_ok"], q["left_tok"]
            fire += 1
            gain += int(ok and not q["host_ok"])
            hurt += int((not ok) and q["host_ok"])
            if q["left_step"] is not None:
                lags.append(float(q["n_steps"] - q["left_step"]))
        else:
            ok, t = q["host_ok"], q["host_tok"]
        acc += int(ok)
        tok += float(t)
    n = max(len(qs), 1)
    host_acc = sum(int(q["host_ok"]) for q in qs) / n
    host_tok = sum(float(q["host_tok"]) for q in qs) / n
    return {
        "acc": acc / n,
        "tok": tok / n,
        "fire": fire,
        "gain": gain,
        "hurt": hurt,
        "d_acc": 100.0 * (acc / n - host_acc),
        "d_tok": tok / n - host_tok,
        "lag_p50": rg.p50(lags) if lags else float("nan"),
        "n": float(len(qs)),
        "host_acc": host_acc,
        "host_tok": host_tok,
    }


def pred_of(name: str):
    if name == "commit":
        return lambda q: q["has_left"] and q["committed"]
    if name == "ok":
        return lambda q: q["has_left"] and q["left_ok"]
    if name == "mix_ok":
        return lambda q: q["has_left"] and q["left_kind"] == "mix" and q["left_ok"]
    if name == "low_ok":
        return lambda q: q["has_left"] and q["left_kind"] == "low" and q["left_ok"]
    if name == "never_ok":
        return lambda q: q["never_high"] and q["left_ok"]
    if name == "never_commit":
        return lambda q: q["never_high"] and q["committed"]
    raise KeyError(name)


def worth(rec: dict[str, float]) -> str:
    if rec["d_acc"] >= 2.0 and rec["d_tok"] <= -400:
        return "Acc+Tok"
    if rec["d_acc"] >= 2.0:
        return "只Acc"
    if rec["d_tok"] <= -500:
        return "只Tok"
    return "窄"


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    cells: list[tuple[str, str, list[dict[str, Any]]]] = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            seeds = (42, 0, 1, 123) if dataset in ("aime24", "aime25") else (42,)
            qs: list[dict[str, Any]] = []
            for seed in seeds:
                qs.extend(load_cell(model, dataset, seed))
            if qs:
                cells.append((f"{zh} {ds_zh}", qs[0]["host_name"] if qs else "—", qs))
                print(f"{zh} {ds_zh} n={len(qs)} leftover={sum(1 for q in qs if q['has_left'])}", flush=True)

    lines = [
        "# 剩窗切片上限：跨模型跨数据集",
        "",
        "只认密探轨迹。剩窗 = 第一扇低把握或混合连答窗。",
        "后面有没有高把握，看同一条轨迹后头还会不会出现高把握连答。",
        "上限都是先知：只切该切片，交试答不重写，其余题留宿主（有密探k4 重写就留它，否则 PUMA，再没有就写完）。",
        "「值得」= 相对宿主 Acc ≥ +2pp 且 token ≤ −400，或至少一边够大且不随模型变大消失。",
        "",
        "## 1. 剩窗之后会不会来高把握",
        "",
        "高把握先到的题没有剩窗。这里只看有剩窗的题。",
        "",
        "| 集 | 题 | 剩窗 | 混合/低把握 | 后面高把握同答 | 后面高把握换答 | 再也没有高把握 | 宿主 |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for name, host, qs in cells:
        n = len(qs)
        left = [q for q in qs if q["has_left"]]
        mix = sum(1 for q in left if q["left_kind"] == "mix")
        low_n = sum(1 for q in left if q["left_kind"] == "low")
        same = sum(1 for q in left if q["same_as_high"])
        chg = sum(1 for q in left if q["will_change"])
        never = sum(1 for q in left if q["never_high"])
        L = max(len(left), 1)
        lines.append(
            f"| {name} | {n} | {len(left)}（{100.0 * len(left) / n:.0f}%） | "
            f"{mix}/{low_n} | {same}（{100.0 * same / L:.0f}%） | "
            f"{chg}（{100.0 * chg / L:.0f}%） | {never}（{100.0 * never / L:.0f}%） | {host} |"
        )

    lines += [
        "",
        "## 2. GPQA 会不会随模型变大变成高把握题",
        "",
        "先到高把握 = 第一扇连答就是高把握，密探k4 已经能停。",
        "剩窗后再也没有高把握 = 写完才停，这才是大 token。",
        "",
        "| 模型 | 题 | 先到高把握 | 有剩窗 | 剩窗后再也没有高把握 | 剩窗后换答再锁 | 低把握剩窗且对 上限 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, host, qs in cells:
        if "GPQA" not in name:
            continue
        n = len(qs)
        first_high = sum(1 for q in qs if q["first_kind"] == "high")
        left = [q for q in qs if q["has_left"]]
        never = sum(1 for q in left if q["never_high"])
        chg = sum(1 for q in left if q["will_change"])
        rec = apply(qs, pred_of("low_ok"))
        lines.append(
            f"| {name} | {n} | {first_high}（{100.0 * first_high / n:.0f}%） | "
            f"{len(left)} | {never}（{100.0 * never / n:.0f}% 题） | {chg} | "
            f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f}（开火 {int(rec['fire'])}） |"
        )

    lines += [
        "",
        "## 3. 各切片先知上限（相对宿主）",
        "",
        "Δ 是 Acc / token。开火写在括号里。值得一列看这一格够不够大。",
        "",
        "| 集 | 宿主 Acc/Tok | 剩窗=写完终答 | 剩窗且对 | 混合且对 | 低把握且对 | 再也没有高把握且对 |",
        "|---|---|---|---|---|---|---|",
    ]
    slice_rows: dict[str, list[tuple[str, dict[str, float]]]] = {k: [] for k, _ in SLICES}
    for name, host, qs in cells:
        recs = {k: apply(qs, pred_of(k)) for k, _ in SLICES}
        for k, rec in recs.items():
            slice_rows[k].append((name, rec))
        def cell(rec: dict[str, float]) -> str:
            return (
                f"{rec['d_acc']:+.1f}pp / {rec['d_tok']:+.0f} "
                f"（{int(rec['fire'])}，{worth(rec)}）"
            )
        host_acc = recs["commit"]["host_acc"]
        host_tok = recs["commit"]["host_tok"]
        lines.append(
            f"| {name} | {fmt_pair(host_acc, host_tok)} | "
            f"{cell(recs['commit'])} | {cell(recs['ok'])} | "
            f"{cell(recs['mix_ok'])} | {cell(recs['low_ok'])} | "
            f"{cell(recs['never_ok'])} |"
        )

    lines += [
        "",
        "## 4. 哪一块跨模型还在",
        "",
        "按数据集看：这一切片在几个模型上算「值得」，会不会随变大变窄。",
        "",
        "| 数据集 | 切片 | 值得的模型 | 最窄的一格 | 最大 Acc | 最多省 token |",
        "|---|---|---|---|---:|---:|",
    ]
    by_ds: dict[str, list[str]] = defaultdict(list)
    for name, host, qs in cells:
        ds = name.split(" ", 1)[1]
        by_ds[ds].append(name)
    order = ["MATH", "奥赛", "GPQA", "AIME24", "AIME25"]
    for ds in order:
        names = by_ds.get(ds) or []
        if not names:
            continue
        for key, zh in SLICES:
            recs = [(n, r) for n, r in slice_rows[key] if n.split(" ", 1)[1] == ds]
            if not recs:
                continue
            good = [n.split(" ", 1)[0] for n, r in recs if worth(r) != "窄"]
            worst = min(recs, key=lambda x: (x[1]["d_acc"], -x[1]["d_tok"]))
            best_acc = max(r["d_acc"] for _, r in recs)
            best_tok = min(r["d_tok"] for _, r in recs)
            lines.append(
                f"| {ds} | {zh} | "
                f"{', '.join(good) if good else '没有'} | "
                f"{worst[0]} {worst[1]['d_acc']:+.1f}/{worst[1]['d_tok']:+.0f} | "
                f"{best_acc:+.1f} | {best_tok:+.0f} |"
            )

    lines += [
        "",
        "## 读法",
        "",
        "混合且对：多半腿已经在高把握锁前两三步，省不了多少。",
        "低把握且对 / 再也没有高把握且对：同一摊题时，就是写完才停的那些。",
        "剩窗=写完终答：含「后面高把握同答」的啰嗦题，Acc 通常平，token 少。",
        "先看第 2 节 GPQA 会不会随变大消失，再看第 4 节哪一块跨模型还在。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
