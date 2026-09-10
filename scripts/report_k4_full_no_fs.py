#!/usr/bin/env python3
"""密探k4 去掉后路之后，写完的题还有多少可停空间。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE / "scripts"))

import replay_default_dense_gate as dd
import replay_rescue_R_gate as rg
import report_dense_k4_lowconf_ceiling as low
import report_first_lock_room as room

TABLE = AE / "tables/k4_full_no_fs.md"
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


def full_tokens(trials: list[dict[str, Any]], orig_tok: int) -> tuple[int, int]:
    last = max(trials, key=lambda x: int(x["stopped_len"]))
    usable = [
        x
        for x in trials
        if dd.should_probe(int(x["stopped_len"]))
        and str(x.get("final_answer") or "")
        and dd.finite(x.get("confidence")) == dd.finite(x.get("confidence"))
    ]
    trial_tok = sum(dd.trial_answer_tokens(x) for x in usable)
    return (orig_tok or 0) + trial_tok, int(last["stopped_len"])


def load_cell(model: str, dataset: str, seed: int) -> dict[str, Any] | None:
    trials_path = room.dense_trial_path(model, dataset, seed)
    if not trials_path.is_file():
        return None
    puma_path = dd.puma_stat_path(model, dataset, seed)
    regen_path = dd.regen_stat_path(model, dataset, seed)
    official = (
        {int(r["question_idx"]): r for r in dd.load_json(puma_path)} if puma_path.is_file() else {}
    )
    regen = (
        {int(r["question_idx"]): r for r in dd.load_json(regen_path)} if regen_path.is_file() else {}
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
    questions = []
    used_regen = 0
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
        sim = dd.simulate(trials, original_tokens=orig_tok)
        host = regen.get(qi)
        if host:
            used_regen += 1
            k4_tok = int(host.get("compressed_tokens") or 0) + int(host.get("tokens_trial_answers") or 0)
            k4_ok = bool(host.get("compressed_correct"))
            consec_tok = k4_tok if sim["branch"] == "consec" else sim["tokens"]
            consec_ok = k4_ok if sim["branch"] == "consec" else bool(
                low.credit(sim["answer"], gt, original, orig_ok)
            )
            consec_step = int(host.get("stopped_len") or sim["step"])
        else:
            if sim["branch"] == "full":
                k4_tok, _ = full_tokens(trials, orig_tok)
                k4_ok = orig_ok
            else:
                k4_tok = sim["tokens"]
                k4_ok = bool(low.credit(sim["answer"], gt, original, orig_ok))
            consec_tok = sim["tokens"]
            consec_ok = bool(low.credit(sim["answer"], gt, original, orig_ok))
            consec_step = int(sim["step"])
        if sim["branch"] == "consec":
            nofs_branch = "consec"
            nofs_tok = consec_tok
            nofs_ok = consec_ok
            nofs_step = consec_step
        else:
            nofs_branch = "full"
            nofs_tok, nofs_step = full_tokens(trials, orig_tok)
            nofs_ok = orig_ok
        lock = room.first_ok(rows, gt, original, a_final)
        fire = None
        if lock is not None:
            packed = rg.pack(trials, rows, lock["end"], "rescue", original_tokens=orig_tok)
            fire = {
                **lock,
                "ok": low.credit(packed["answer"], gt, original, orig_ok),
                "tok": packed["tokens"],
            }
        if info:
            puma_ok = bool(info.get("compressed_correct"))
            puma_tok = int(info.get("compressed_tokens") or 0) + int(info.get("tokens_trial_answers") or 0)
            puma_reason = str(info.get("stop_reason") or "")
            if puma_reason == "forced_stop_redundancy":
                puma_nofs_ok = orig_ok
                puma_nofs_tok = orig_tok + int(info.get("tokens_trial_answers") or 0)
            else:
                puma_nofs_ok = puma_ok
                puma_nofs_tok = puma_tok
            has_puma = True
        else:
            puma_ok = puma_tok = puma_nofs_ok = puma_nofs_tok = float("nan")
            has_puma = False
        questions.append(
            {
                "k4_ok": k4_ok,
                "k4_tok": k4_tok,
                "k4_branch": sim["branch"],
                "nofs_ok": nofs_ok,
                "nofs_tok": nofs_tok,
                "nofs_step": nofs_step,
                "nofs_branch": nofs_branch,
                "orig_ok": orig_ok,
                "has_puma": has_puma,
                "puma_ok": puma_ok,
                "puma_tok": puma_tok,
                "puma_nofs_ok": puma_nofs_ok,
                "puma_nofs_tok": puma_nofs_tok,
                "fire": fire,
            }
        )
    if not questions:
        return None
    return {"questions": questions, "regen": used_regen > 0}


def merge(packs: list[dict[str, Any]]) -> dict[str, Any]:
    qs = []
    for pack in packs:
        qs.extend(pack["questions"])
    return {"questions": qs}


def apply(qs: list[dict[str, Any]], *, only_full: bool, want: str | None) -> dict[str, Any]:
    acc = tok = fire = earlier = 0
    lags: list[float] = []
    delivered: list[tuple[dict[str, Any], bool, float]] = []
    for q in qs:
        hit = q["fire"]
        use = (
            hit is not None
            and (not only_full or q["nofs_branch"] == "full")
            and (want is None or hit["kind"] == want)
        )
        if use:
            ok, t = hit["ok"], hit["tok"]
            fire += 1
            if hit["step"] < q["nofs_step"]:
                earlier += 1
                lags.append(q["nofs_step"] - hit["step"])
        else:
            ok, t = q["nofs_ok"], q["nofs_tok"]
        delivered.append((q, bool(ok), float(t)))
        acc += int(ok)
        tok += t
    n = len(qs)

    def versus(ok_key: str, tok_key: str) -> dict[str, float]:
        keep = [(q, ok, t) for q, ok, t in delivered if q.get("has_puma") or ok_key.startswith("nofs") or ok_key.startswith("k4")]
        if ok_key.startswith("puma"):
            keep = [(q, ok, t) for q, ok, t in delivered if q.get("has_puma")]
        if not keep:
            return {"acc": float("nan"), "tok": float("nan"), "d_acc": float("nan"), "d_tok": float("nan"), "gain": 0, "hurt": 0, "n": 0}
        base_acc = sum(q[ok_key] for q, _ok, _t in keep) / len(keep)
        base_tok = sum(q[tok_key] for q, _ok, _t in keep) / len(keep)
        door_acc = sum(ok for _q, ok, _t in keep) / len(keep)
        door_tok = sum(t for _q, _ok, t in keep) / len(keep)
        return {
            "acc": door_acc,
            "tok": door_tok,
            "d_acc": 100.0 * (door_acc - base_acc),
            "d_tok": door_tok - base_tok,
            "gain": sum(int(ok and not q[ok_key]) for q, ok, _t in keep),
            "hurt": sum(int((not ok) and q[ok_key]) for q, ok, _t in keep),
            "n": len(keep),
        }

    return {
        "acc": acc / n,
        "tok": tok / n,
        "fire": fire,
        "earlier": earlier,
        "lag_p50": rg.p50(lags) if lags else float("nan"),
        "vs_nofs": versus("nofs_ok", "nofs_tok"),
        "vs_puma": versus("puma_ok", "puma_tok"),
        "vs_puma_nofs": versus("puma_nofs_ok", "puma_nofs_tok"),
    }


def score(pack: dict[str, Any]) -> dict[str, Any]:
    qs = pack["questions"]
    n = len(qs)
    full = [q for q in qs if q["nofs_branch"] == "full"]
    puma_qs = [q for q in qs if q.get("has_puma")]
    rec: dict[str, Any] = {
        "n": n,
        "k4_acc": sum(q["k4_ok"] for q in qs) / n,
        "k4_tok": sum(q["k4_tok"] for q in qs) / n,
        "nofs_acc": sum(q["nofs_ok"] for q in qs) / n,
        "nofs_tok": sum(q["nofs_tok"] for q in qs) / n,
        "has_puma": bool(puma_qs),
        "puma_acc": sum(q["puma_ok"] for q in puma_qs) / len(puma_qs) if puma_qs else float("nan"),
        "puma_tok": sum(q["puma_tok"] for q in puma_qs) / len(puma_qs) if puma_qs else float("nan"),
        "puma_nofs_acc": sum(q["puma_nofs_ok"] for q in puma_qs) / len(puma_qs) if puma_qs else float("nan"),
        "puma_nofs_tok": sum(q["puma_nofs_tok"] for q in puma_qs) / len(puma_qs) if puma_qs else float("nan"),
        "n_consec": sum(1 for q in qs if q["nofs_branch"] == "consec"),
        "n_full": len(full),
        "n_was_fs": sum(1 for q in full if q["k4_branch"] == "fs"),
        "n_was_full": sum(1 for q in full if q["k4_branch"] == "full"),
        "n_ok": sum(1 for q in full if q["fire"]),
        "n_high": sum(1 for q in full if q["fire"] and q["fire"]["kind"] == "high"),
        "n_mix": sum(1 for q in full if q["fire"] and q["fire"]["kind"] == "mix"),
        "n_low": sum(1 for q in full if q["fire"] and q["fire"]["kind"] == "low"),
        "n_none": sum(1 for q in full if not q["fire"]),
        "all": apply(qs, only_full=True, want=None),
        "high": apply(qs, only_full=True, want="high"),
        "mix": apply(qs, only_full=True, want="mix"),
        "low": apply(qs, only_full=True, want="low"),
    }
    return rec


def cell_vs(rec: dict[str, Any], vs_key: str) -> str:
    door = rec["all"]
    vs = door[vs_key]
    if vs["n"] == 0 or vs["acc"] != vs["acc"]:
        return "—"
    lag = (
        f"，早中位 {door['lag_p50']:.0f} 步"
        if door["earlier"] and door["lag_p50"] == door["lag_p50"]
        else ""
    )
    return (
        f"{rg.fmt_pct(vs['acc'])} / {vs['tok']:.0f}"
        f"（{rg.fmt_pp(vs['d_acc'])} / {rg.fmt_tok(vs['d_tok'])}；"
        f"开火 {door['fire']}，救回 {vs['gain']} / 伤 {vs['hurt']}{lag}）"
    )


def fmt_base(acc: float, tok: float) -> str:
    if acc != acc:
        return "—"
    return f"{rg.fmt_pct(acc)} / {tok:.0f}"


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
    rec["regen"] = any(p.get("regen") for p in (packs if dataset in ("aime24", "aime25") else [pack]))
    return rec


def main() -> None:
    recs = []
    for zh, model in MODELS:
        for ds_zh, dataset in DS:
            rec = load_named(zh, model, ds_zh, dataset)
            if rec:
                recs.append(rec)
                print(
                    f"{rec['name']} full={rec['n_full']}/{rec['n']} "
                    f"lock={rec['n_ok']} none={rec['n_none']}",
                    flush=True,
                )
    lines = [
        "# 密探k4 去掉后路：写完的题还有多少空间",
        "",
        "密探k4 只留连答锁（k=4、第一次 ≥0.995、后面 ≥第一次−0.03），后路强停关掉。",
        "没锁上的题整条写完，交写完终答。7B/8B 连答停仍用重写交卷；14B/32B/30B/Qwen3 没有重写，连答停交试答。",
        "可停 = 连续 4 步同一试答，且等于金标或写完终答，步数 ≥10。",
        "「写完上开火」= 只对 k4 去掉后路之后仍写完的题，在第一次可停交试答、不重写；连答停的题不动。",
        "官方 PUMA 无后路：后路强停的题改成写完（复用原终答 / 原 token）。Qwen3 MATH 没有官方 PUMA。",
        "Acc 只对金标。缺密探轨迹的格子不写。",
        "",
        "## 1. 去掉后路之后，谁在写完",
        "",
        "| 集 | 题 | 仍连答停 | 将写完 | 其中现在是后路 | 其中现在已写完 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n']} | {rec['n_consec']} "
            f"| {rec['n_full']}（{100.0 * rec['n_full'] / rec['n']:.0f}%） "
            f"| {rec['n_was_fs']} | {rec['n_was_full']} |"
        )
    lines += [
        "",
        "## 2. 写完的题里，已经能锁的有多少",
        "",
        "| 集 | 写完 | 已可停 | 高把握 | 混合 | 低把握 | 无可停 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} | {rec['n_full']} | {rec['n_ok']} "
            f"| {rec['n_high']} | {rec['n_mix']} | {rec['n_low']} | {rec['n_none']} |"
        )
    lines += [
        "",
        "## 3. 写完里可停都交，对照 PUMA / k4",
        "",
        "括号是相对该列底的 Acc 百分点差 / token 差。救回 / 伤也相对该底。",
        "",
        "| 集 | PUMA 有后路 | PUMA 无后路 | k4 无后路 | 可停都交 vs PUMA有后路 | 可停都交 vs PUMA无后路 | 可停都交 vs k4无后路 |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in recs:
        lines.append(
            f"| {rec['name']} "
            f"| {fmt_base(rec['puma_acc'], rec['puma_tok'])} "
            f"| {fmt_base(rec['puma_nofs_acc'], rec['puma_nofs_tok'])} "
            f"| {fmt_base(rec['nofs_acc'], rec['nofs_tok'])} "
            f"| {cell_vs(rec, 'vs_puma')} "
            f"| {cell_vs(rec, 'vs_puma_nofs')} "
            f"| {cell_vs(rec, 'vs_nofs')} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "去掉后路，写完 = 现在的写完 + 现在的后路。后路题会变长，无后路的 token 往往更贵。",
        "「可停都交」只切 k4 无后路之后仍写完、且已经能锁的题。连答停的题不动。",
        "相对官方 PUMA：连答停那批若 k4 重写/试答和官方不一样，Acc 也会动。",
        "",
        "缺的格子：30B 奥赛（密探没写完）、Qwen3-4B AIME25、Qwen3-8B 奥赛、Qwen3-4B AIME24 缺 seed 123、Qwen3-8B AIME25 缺 seed 42。Qwen3-4B/8B MATH 没有官方 PUMA，写完终答用轨迹 `A_final`。",
        "",
    ]
    TABLE.write_text("\n".join(lines))
    print(f"wrote {TABLE}", flush=True)


if __name__ == "__main__":
    main()
