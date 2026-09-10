#!/usr/bin/env python3
"""k4 无后路：第一扇同答窗谁先到，低把握先到会不会切掉后面的高把握锁。"""
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
import report_k4_full_no_fs as nofs
import report_k4_second_lock as sl

TABLE = AE / "tables/k4_nofs_window_order.md"
SIGS = (
    ("geo", "把握"),
    ("geo_min", "最差把握"),
    ("geo_rise", "把握升降"),
    ("neg_dist", "越少换答"),
    ("rel", "相对步数"),
)


def windows_of(
    rows: list[dict[str, Any]], gt: Any, original: Any, a_final: Any
) -> list[dict[str, Any]]:
    n_last = max(int(rows[-1]["stopped_len"]), 1)
    out = []
    for win in sl.same_windows(rows):
        confs = rg.confs_of(rows[win["end"] + 1 - rg.K : win["end"] + 1])
        seen = []
        for row in rows[: win["end"] + 1]:
            ans = row.get("final_answer")
            if not any(rg.same(ans, x) for x in seen):
                seen.append(ans)
        out.append(
            {
                **win,
                "pos": cc.stoppable(win["ans"], gt, original, a_final),
                "geo": confs[-1],
                "geo_min": min(confs),
                "geo_rise": confs[-1] - confs[0],
                "neg_dist": -float(len(seen)),
                "rel": win["step"] / n_last,
            }
        )
    return out


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    pack = nofs.load_cell(model, dataset, seed)
    if pack is None:
        return None
    trials_path = room.dense_trial_path(model, dataset, seed)
    puma_path = dd.puma_stat_path(model, dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    gp = low.gpath(model, dataset, seed)
    gmap = {int(r["question_idx"]): r for r in dd.load_json(gp)} if gp.is_file() else {}
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in dd.load_json(trials_path):
        by[int(row["question_idx"])].append(row)
    if official:
        qis = sorted(official)
    elif gmap:
        qis = sorted(set(gmap) & set(by))
    else:
        qis = sorted(by)
    qmap = {}
    used = 0
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
        orig_ok = bool(info.get("original_correct")) if info else bool(
            low.credit(original, gt, original, True)
        )
        orig_tok = int(info.get("original_tokens") or last.get("count_reasoning_tokens") or 0)
        wins = windows_of(rows, gt, original, a_final)
        first = wins[0] if wins else None
        first_ok = next((w for w in wins if w["pos"]), None)
        first_high = next((w for w in wins if w["kind"] == "high"), None)
        leftover = first if first is not None and first["kind"] != "high" else None
        fire = None
        if leftover is not None:
            packed = rg.pack(trials, rows, leftover["end"], "rescue", original_tokens=orig_tok)
            fire = {
                **leftover,
                "ok": low.credit(packed["answer"], gt, original, orig_ok),
                "tok": packed["tokens"],
            }
        qmap[used] = {
            "wins": wins,
            "first": first,
            "first_ok": first_ok,
            "first_high": first_high,
            "leftover": leftover,
            "fire": fire,
            "orig_ok": orig_ok,
        }
        used += 1
    qs = pack["questions"]
    if used != len(qs):
        print(f"warn {model} {dataset} seed={seed}: windows {used} vs qs {len(qs)}", flush=True)
        return None
    for i, q in enumerate(qs):
        q.update(qmap[i])
        first = q["first"]
        high = q["first_high"]
        leftover = q["leftover"]
        if q["nofs_branch"] == "consec":
            q["fate"] = "consec"
        elif q["first_ok"] is not None:
            q["fate"] = "full_ok"
        else:
            q["fate"] = "full_none"
        q["left_then_high"] = bool(
            leftover is not None and high is not None and leftover["step"] < high["step"]
        )
        q["left_same_high"] = bool(
            q["left_then_high"] and rg.same(leftover["ans"], high["ans"])
        )
        q["first_kind"] = first["kind"] if first else "none"
        q["ok_kind"] = q["first_ok"]["kind"] if q["first_ok"] else "none"
    return pack


def merge(packs: list[dict[str, Any]]) -> dict[str, Any]:
    qs = []
    for pack in packs:
        qs.extend(pack["questions"])
    return {"questions": qs}


def auroc_field(xs: list[dict[str, Any]], field: str, pred) -> float:
    pos = [float(x[field]) for x in xs if pred(x)]
    neg = [float(x[field]) for x in xs if not pred(x)]
    return rg.auroc(pos, neg)


def apply_door(qs: list[dict[str, Any]], pred) -> dict[str, float]:
    acc = tok = fire = gain = hurt = puma_gain = puma_hurt = 0
    puma_n = 0
    puma_base = puma_door = 0.0
    puma_tok_base = puma_tok_door = 0.0
    for q in qs:
        hit = q["fire"]
        use = hit is not None and pred(q)
        if use:
            ok, t = hit["ok"], hit["tok"]
            fire += 1
            gain += int(ok and not q["nofs_ok"])
            hurt += int((not ok) and q["nofs_ok"])
        else:
            ok, t = q["nofs_ok"], q["nofs_tok"]
        acc += int(ok)
        tok += t
        if q.get("has_puma"):
            puma_n += 1
            puma_base += int(q["puma_nofs_ok"])
            puma_tok_base += q["puma_nofs_tok"]
            puma_door += int(ok)
            puma_tok_door += t
            puma_gain += int(ok and not q["puma_nofs_ok"])
            puma_hurt += int((not ok) and q["puma_nofs_ok"])
    n = len(qs)
    return {
        "acc": acc / n,
        "tok": tok / n,
        "fire": fire,
        "gain": gain,
        "hurt": hurt,
        "puma_n": puma_n,
        "puma_acc": (puma_door / puma_n) if puma_n else float("nan"),
        "puma_tok": (puma_tok_door / puma_n) if puma_n else float("nan"),
        "puma_d_acc": (100.0 * (puma_door - puma_base) / puma_n) if puma_n else float("nan"),
        "puma_d_tok": ((puma_tok_door - puma_tok_base) / puma_n) if puma_n else float("nan"),
        "puma_gain": puma_gain,
        "puma_hurt": puma_hurt,
        "puma_base_acc": (puma_base / puma_n) if puma_n else float("nan"),
        "puma_base_tok": (puma_tok_base / puma_n) if puma_n else float("nan"),
    }


def score(pack: dict[str, Any]) -> dict[str, Any]:
    qs = pack["questions"]
    n = len(qs)
    consec = [q for q in qs if q["fate"] == "consec"]
    full_ok = [q for q in qs if q["fate"] == "full_ok"]
    full_none = [q for q in qs if q["fate"] == "full_none"]
    left_high = [q for q in consec if q["left_then_high"]]
    left_same = [q for q in left_high if q["left_same_high"]]
    left_diff = [q for q in left_high if not q["left_same_high"]]
    leftovers = [q["leftover"] for q in qs if q["leftover"] is not None]
    later_high = [q["leftover"] for q in qs if q["left_then_high"]]
    never_high = [q["leftover"] for q in qs if q["leftover"] is not None and q["first_high"] is None]
    full_left = [q["leftover"] for q in qs if q["fate"].startswith("full") and q["leftover"] is not None]

    def kind_counts(group: list[dict[str, Any]], key: str = "first_kind") -> dict[str, int]:
        return {k: sum(1 for q in group if q[key] == k) for k in ("high", "mix", "low", "none")}

    rec: dict[str, Any] = {
        "n": n,
        "n_consec": len(consec),
        "n_full_ok": len(full_ok),
        "n_full_none": len(full_none),
        "consec_k": kind_counts(consec),
        "full_ok_first": kind_counts(full_ok),
        "full_ok_lock": kind_counts(full_ok, "ok_kind"),
        "full_none_k": kind_counts(full_none),
        "n_left_high": len(left_high),
        "n_left_same": len(left_same),
        "n_left_diff": len(left_diff),
        "n_left_diff_hurt": sum(1 for q in left_diff if q["fire"] and (not q["fire"]["ok"]) and q["nofs_ok"]),
        "n_left_diff_ok": sum(1 for q in left_diff if q["fire"] and q["fire"]["ok"]),
        "n_full_ok_first_is_lock": sum(
            1 for q in full_ok if q["first"] is not None and q["first"]["pos"]
        ),
        "n_full_ok_later_lock": sum(
            1
            for q in full_ok
            if q["first"] is not None and not q["first"]["pos"] and q["first_ok"] is not None
        ),
        "nofs_acc": sum(q["nofs_ok"] for q in qs) / n,
        "nofs_tok": sum(q["nofs_tok"] for q in qs) / n,
        "puma_has": any(q.get("has_puma") for q in qs),
    }
    rec["later_high_sig"] = {f: auroc_field(leftovers, f, lambda w, later=set(id(x) for x in later_high): id(w) in later) for f, _ in SIGS} if leftovers and later_high and never_high else {f: float("nan") for f, _ in SIGS}
    if leftovers and later_high and never_high:
        rec["later_high_sig"] = {f: rg.auroc(
            [float(w[f]) for w in later_high],
            [float(w[f]) for w in never_high],
        ) for f, _ in SIGS}
        rec["n_later_high"] = len(later_high)
        rec["n_never_high"] = len(never_high)
    else:
        rec["later_high_sig"] = {f: float("nan") for f, _ in SIGS}
        rec["n_later_high"] = len(later_high)
        rec["n_never_high"] = len(never_high)
    if full_left:
        rec["full_pos_sig"] = {f: cc.auroc_of(full_left, f)[0] for f, _ in SIGS}
        rec["n_full_left_pos"] = sum(1 for w in full_left if w["pos"])
        rec["n_full_left_neg"] = sum(1 for w in full_left if not w["pos"])
    else:
        rec["full_pos_sig"] = {f: float("nan") for f, _ in SIGS}
        rec["n_full_left_pos"] = 0
        rec["n_full_left_neg"] = 0
    rec["always"] = apply_door(qs, lambda q: True)
    rec["full_left"] = apply_door(qs, lambda q: q["fate"].startswith("full"))
    rec["full_left_pos"] = apply_door(qs, lambda q: q["fate"] == "full_ok" and q["first"] is not None and q["first"]["pos"])
    rec["full_ok_any"] = apply_door(
        qs,
        lambda q: q["fate"] == "full_ok" and q["leftover"] is not None,
    )
    return rec


def load_named(zh: str, model: str, ds_zh: str, dataset: str) -> dict[str, Any] | None:
    if dataset in ("aime24", "aime25"):
        packs = []
        for seed in (42, 0, 1, 123):
            pack = load_cell(model, dataset, seed)
            if pack:
                packs.append(pack)
        if not packs:
            return None
        pack = merge(packs)
        name = f"{zh} {ds_zh}" if len(packs) == 4 else f"{zh} {ds_zh}（{len(packs)} seed）"
    else:
        pack = load_cell(model, dataset, 42)
        if pack is None:
            return None
        name = f"{zh} {ds_zh}"
    rec = score(pack)
    rec["name"] = name
    print(
        f"{name} consec={rec['n_consec']} left→high={rec['n_left_high']} "
        f"same={rec['n_left_same']} diff={rec['n_left_diff']} "
        f"full_ok={rec['n_full_ok']} none={rec['n_full_none']}",
        flush=True,
    )
    return rec


def kind_cell(counts: dict[str, int]) -> str:
    return f"{counts['high']}/{counts['mix']}/{counts['low']}/{counts['none']}"


def fmt_auroc(v: float) -> str:
    return "—" if v != v else f"{v:.2f}"


def door_cell(rec: dict[str, Any], key: str) -> str:
    door = rec[key]
    if rec["puma_has"] and door["puma_n"]:
        return (
            f"{rg.fmt_pct(door['puma_acc'])} / {door['puma_tok']:.0f}"
            f"（{rg.fmt_pp(door['puma_d_acc'])} / {rg.fmt_tok(door['puma_d_tok'])}；"
            f"开火 {door['fire']}，救回 {door['puma_gain']} / 伤 {door['puma_hurt']}）"
        )
    d_acc = 100.0 * (door["acc"] - rec["nofs_acc"])
    d_tok = door["tok"] - rec["nofs_tok"]
    return (
        f"{rg.fmt_pct(door['acc'])} / {door['tok']:.0f}"
        f"（{rg.fmt_pp(d_acc)} / {rg.fmt_tok(d_tok)} vs k4无后路；"
        f"开火 {door['fire']}，救回 {door['gain']} / 伤 {door['hurt']}）"
    )


def main() -> None:
    rg.K = 4
    rg.TAU = 0.995
    recs = []
    for zh, model in nofs.MODELS:
        for ds_zh, dataset in nofs.DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
    lines = [
        "# k4 无后路：第一扇窗谁先到，会不会误杀高把握锁",
        "",
        "底是密探k4 去掉后路。连答停 = 已经出现过高把握四步同答锁。写完 = 从没出现这把锁。",
        "第一扇 = 第一次连续 4 步同一试答、步数 ≥10，不问对错。剩窗 = 第一扇不是高把握。",
        "对照默认官方 PUMA 无后路。没有官方 PUMA 的格子相对 k4 无后路。",
        "高/混/低/无 = 第一扇档；无 = 从来没有四步同答窗。",
        "",
        "## 1. 三组题的第一扇是哪一档",
        "",
        "| 集 | 连答停 | 连答停第一扇 高/混/低/无 | 写完已可停 | 可停题第一扇 | 可停题第一扇可停窗 | 写完无可停 | 无可停第一扇 |",
        "|---|---:|---|---:|---|---|---:|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n_consec']} | {kind_cell(rec['consec_k'])} "
            f"| {rec['n_full_ok']} | {kind_cell(rec['full_ok_first'])} "
            f"| {kind_cell(rec['full_ok_lock'])} "
            f"| {rec['n_full_none']} | {kind_cell(rec['full_none_k'])} |"
        )
    lines += [
        "",
        "## 2. 连答停的题：低/混合有没有先到",
        "",
        "左→高 = 第一扇不是高把握，后面才出现高把握锁。同答 = 剩窗试答和后来高把握锁同一串。",
        "异答 = 剩窗已经换了答案；这时若看见剩窗就停，才会切掉后面那把高把握锁。",
        "伤 = 异答剩窗交出去是错的，而 k4 无后路连答停是对的。",
        "",
        "| 集 | 连答停 | 左→高 | 其中同答 | 其中异答 | 异答开火会伤 | 写完已可停里第一扇已经可停 | 第一扇还不对、后面才可停 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n_consec']} | {rec['n_left_high']} "
            f"| {rec['n_left_same']} | {rec['n_left_diff']} | {rec['n_left_diff_hurt']} "
            f"| {rec['n_full_ok_first_is_lock']} | {rec['n_full_ok_later_lock']} |"
        )
    lines += [
        "",
        "## 3. 现成尺子",
        "",
        "「后面会来高把握」：正类 = 剩窗之后真的出现了高把握锁；负类 = 这题再也没有高把握锁。",
        "「写完剩窗可停」：只看写完题的剩窗，正类 = 这扇已经等于金标或写完终答。",
        "AUROC 越大越像正类，0.50 是猜。",
        "",
        "| 集 | 左→高 / 再也没有 | 把握 | 最差把握 | 升降 | 越少换答 | 相对步数 | 写完剩窗对/错 | 把握 | 最差把握 | 升降 | 越少换答 | 相对步数 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in recs:
        a = rec["later_high_sig"]
        b = rec["full_pos_sig"]
        lines.append(
            f"| {rec['name']} | {rec['n_later_high']}/{rec['n_never_high']} "
            f"| {fmt_auroc(a['geo'])} | {fmt_auroc(a['geo_min'])} | {fmt_auroc(a['geo_rise'])} "
            f"| {fmt_auroc(a['neg_dist'])} | {fmt_auroc(a['rel'])} "
            f"| {rec['n_full_left_pos']}/{rec['n_full_left_neg']} "
            f"| {fmt_auroc(b['geo'])} | {fmt_auroc(b['geo_min'])} | {fmt_auroc(b['geo_rise'])} "
            f"| {fmt_auroc(b['neg_dist'])} | {fmt_auroc(b['rel'])} |"
        )
    lines += [
        "",
        "## 4. 若看见剩窗就停，相对 PUMA 无后路",
        "",
        "不问对错。括号相对官方 PUMA 无后路。",
        "",
        "| 集 | PUMA 无后路 | k4 无后路 | 第一扇剩窗都停 | 只停写完题的剩窗 | 只停写完且第一扇已可停 |",
        "|---|---|---|---|---|---|",
    ]
    for rec in recs:
        puma = (
            f"{rg.fmt_pct(rec['always']['puma_base_acc'])} / {rec['always']['puma_base_tok']:.0f}"
            if rec["puma_has"] and rec["always"]["puma_n"]
            else "—"
        )
        k4 = f"{rg.fmt_pct(rec['nofs_acc'])} / {rec['nofs_tok']:.0f}"
        lines.append(
            f"| {rec['name']} | {puma} | {k4} "
            f"| {door_cell(rec, 'always')} "
            f"| {door_cell(rec, 'full_left')} "
            f"| {door_cell(rec, 'full_left_pos')} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "连答停的题里，若第一扇已经是高把握，剩窗门碰不到它，不存在误杀。",
        "误杀只发生在「低/混合先到、后面才高把握、而且剩窗答案已经变了」。",
        "写完已可停的题里，第一扇几乎不会是高把握：高把握一到，k4 无后路就会连答停，不会写完。",
        "「只停写完题的剩窗」是先知：运行时不知道后面还会不会来高把握锁。",
        "",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
