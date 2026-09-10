#!/usr/bin/env python3
"""第一扇同答窗：对上率、现成尺子能不能分开、第一次锁上就停会伤多少。"""
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

TABLE = AE / "tables/first_same_sep.md"
FIELDS = (
    ("geo", "把握"),
    ("geo_min", "最差把握"),
    ("geo_rise", "把握升降"),
    ("neg_dist", "越少换答"),
    ("rel", "相对步数"),
)


def first_same(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for end, row in enumerate(rows):
        if not rg.window_ok(rows, end):
            continue
        if int(row["stopped_len"]) < rg.MSS:
            continue
        window = rows[end + 1 - rg.K : end + 1]
        confs = rg.confs_of(window)
        n_last = max(int(rows[-1]["stopped_len"]), 1)
        return {
            "end": end,
            "step": int(row["stopped_len"]),
            "ans": row.get("final_answer"),
            "kind": room.window_kind(confs),
            "geo": confs[-1] if confs and confs[-1] == confs[-1] else float("nan"),
            "geo_min": min(confs) if confs and all(c == c for c in confs) else float("nan"),
            "geo_rise": (confs[-1] - confs[0]) if confs and all(c == c for c in confs) else float("nan"),
            "neg_dist": -float(ang_distinct(rows[: end + 1])),
            "rel": int(row["stopped_len"]) / n_last,
        }
    return None


def ang_distinct(rows: list[dict[str, Any]]) -> int:
    seen = []
    for row in rows:
        ans = row.get("final_answer")
        if not any(rg.same(ans, x) for x in seen):
            seen.append(ans)
    return len(seen)


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
        if host:
            n_regen += 1
            host_ok = bool(host.get("compressed_correct"))
            host_tok = int(host.get("compressed_tokens") or 0) + int(
                host.get("tokens_trial_answers") or 0
            )
        else:
            host_ok = bool(info.get("compressed_correct"))
            host_tok = int(info.get("compressed_tokens") or 0) + int(
                info.get("tokens_trial_answers") or 0
            )
        lock = first_same(rows)
        fire = None
        if lock is not None:
            packed = rg.pack(trials, rows, lock["end"], "rescue", original_tokens=orig_tok)
            fire = {
                **lock,
                "pos": cc.stoppable(lock["ans"], gt, original, a_final),
                "ok": low.credit(packed["answer"], gt, original, orig_ok),
                "tok": packed["tokens"],
            }
        questions.append({"host_ok": host_ok, "host_tok": host_tok, "fire": fire})
    if not questions:
        return None
    return {"questions": questions, "n_regen": n_regen}


def apply_always(qs: list[dict[str, Any]], want: str | None) -> dict[str, Any]:
    acc = tok = fire = gain = hurt = 0
    for q in qs:
        hit = q["fire"]
        use = hit is not None and (want is None or hit["kind"] == want)
        if use:
            ok, t = hit["ok"], hit["tok"]
            fire += 1
            gain += int(ok and not q["host_ok"])
            hurt += int((not ok) and q["host_ok"])
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
    }


def score(pack: dict[str, Any]) -> dict[str, Any]:
    qs = pack["questions"]
    n = len(qs)
    locks = [q["fire"] for q in qs if q["fire"]]
    rec: dict[str, Any] = {
        "n": n,
        "host_name": "密探k4" if pack["n_regen"] == n else "PUMA",
        "host_acc": sum(q["host_ok"] for q in qs) / n,
        "host_tok": sum(q["host_tok"] for q in qs) / n,
        "n_lock": len(locks),
        "n_pos": sum(1 for w in locks if w["pos"]),
        "all": apply_always(qs, None),
    }
    for kind in room.KINDS:
        sub = [w for w in locks if w["kind"] == kind]
        rec[f"n_{kind}"] = len(sub)
        rec[f"pos_{kind}"] = sum(1 for w in sub if w["pos"])
        rec[kind] = apply_always(qs, kind)
        rec[f"sig_{kind}"] = {}
        for field, _ in FIELDS:
            rec[f"sig_{kind}"][field] = cc.auroc_of(sub, field)[0]
    rec["sig_all"] = {field: cc.auroc_of(locks, field)[0] for field, _ in FIELDS}
    rec["sig_low"] = rec["sig_low"]
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
    print(
        f"{name} lock={rec['n_lock']}/{rec['n']} pos={rec['n_pos']} "
        f"low {rec['pos_low']}/{rec['n_low']}",
        flush=True,
    )
    return rec


def pct(n: int, d: int) -> str:
    return "—" if d <= 0 else f"{100.0 * n / d:.0f}%"


def fmt_auroc(v: float) -> str:
    return "—" if v != v else f"{v:.2f}"


def door_cell(rec: dict[str, Any], key: str) -> str:
    door = rec[key]
    d_acc = 100.0 * (door["acc"] - rec["host_acc"])
    d_tok = door["tok"] - rec["host_tok"]
    return (
        f"{rg.fmt_pct(door['acc'])} / {door['tok']:.0f}"
        f"（{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)}；"
        f"开火 {door['fire']}，救回 {door['gain']} / 伤 {door['hurt']}）"
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in room.MODELS:
        for ds_zh, dataset in room.DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    lines = [
        "# 第一扇同答窗：分得开吗",
        "",
        "运行时不知道金标、也不知道写完终答。能看见的是：第一次连续 4 步同一试答。",
        "正类 = 这步试答已经等于金标或写完终答。步数 < 10 仍不算。",
        "「第一次锁上就停」= 看见第一扇同答窗就交试答，不重写；没锁上留宿主。",
        "宿主：有密探k4 重写就留它，否则留 PUMA。",
        "低把握 = 四步都 < 0.995。AUROC 越大越像正类，0.50 是猜。",
        "",
        "## 1. 第一扇同答窗有多少其实可停",
        "",
        "| 集 | 题数 | 有第一扇 | 全档对上 | 高把握对上 | 混合对上 | 低把握对上 |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n']} | {rec['n_lock']} "
            f"| {rec['n_pos']}/{rec['n_lock']}（{pct(rec['n_pos'], rec['n_lock'])}） "
            f"| {rec['pos_high']}/{rec['n_high']}（{pct(rec['pos_high'], rec['n_high'])}） "
            f"| {rec['pos_mix']}/{rec['n_mix']}（{pct(rec['pos_mix'], rec['n_mix'])}） "
            f"| {rec['pos_low']}/{rec['n_low']}（{pct(rec['pos_low'], rec['n_low'])}） |"
        )
    lines += [
        "",
        "## 2. 不问对错、第一扇就停",
        "",
        "| 集 | 宿主 | 第一扇都停 | 只停第一扇低把握 |",
        "|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rg.fmt_pct(rec['host_acc'])} / {rec['host_tok']:.0f} "
            f"| {door_cell(rec, 'all')} | {door_cell(rec, 'low')} |"
        )
    lines += [
        "",
        "## 3. 第一扇低把握窗上，现成尺子 AUROC",
        "",
        "| 集 | 低把握第一扇 对/错 | 把握 | 最差把握 | 把握升降 | 越少换答 | 相对步数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in recs:
        s = rec["sig_low"]
        lines.append(
            f"| {rec['name']} | {rec['pos_low']}/{rec['n_low'] - rec['pos_low']} "
            f"| {fmt_auroc(s['geo'])} | {fmt_auroc(s['geo_min'])} "
            f"| {fmt_auroc(s['geo_rise'])} | {fmt_auroc(s['neg_dist'])} "
            f"| {fmt_auroc(s['rel'])} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "先知「低把握先到就停」只停对上的那些。这里是运行时：第一扇低把握同答就停，对错都停。",
        "对上率低、或 AUROC 贴 0.5，就是分不开。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"写成 {TABLE}", flush=True)


if __name__ == "__main__":
    main()
